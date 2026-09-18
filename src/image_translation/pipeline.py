"""Pipeline orchestration – coordinates modules without owning low-level logic."""

from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional

from .config import AppConfig
from .diagnostics import build_image_diagnostic, write_batch_summary, write_image_diagnostic
from .imaging.inpaint_quality import assess_inpaint_quality
from .input import AppInput, InputType
from .models.image_job import ImageJob, JobStatus
from .models.processing_result import ProcessingResult
from .models.text_region import TextAction
from .pipeline_context import PipelineContext
from .quality_control import validate_final_image, validate_region_metadata
from .translation import classify_regions
from .utilities import files, folders, images
from .utilities.image_format import ImagePayload, finalize_output, processing_view
from .utilities.output_paths import (
    resolve_output_file_path,
    resolve_preserved_original_path,
)

logger = logging.getLogger(__name__)


class JobStartDecision(str, Enum):
    proceed = "proceed"
    skip = "skip"
    conflict = "conflict"


def run_pipeline(
    app_input: AppInput,
    config: AppConfig,
    context: Optional[PipelineContext] = None,
) -> ProcessingResult:
    """Execute the full image translation pipeline."""
    result = ProcessingResult()
    ctx = context or PipelineContext(config)

    image_paths = _enumerate_images(app_input, config)
    if not image_paths:
        logger.warning("No images found in: %s", app_input.input_path)
        if config.output.save_metadata:
            write_batch_summary(app_input.output_folder, result, config)
        return result

    logger.info("Found %d image(s).", len(image_paths))

    for source_path in image_paths:
        output_path = resolve_output_file_path(
            source_path,
            app_input.output_folder,
            app_input.input_folder,
            config.output,
        )
        preserved_path = resolve_preserved_original_path(
            source_path,
            app_input.output_folder,
            app_input.input_folder,
        )
        job = ImageJob(
            source_path=source_path,
            output_path=output_path,
            preserved_original_path=preserved_path,
        )

        decision, reason = _evaluate_job_start(job, config)
        if decision == JobStartDecision.skip:
            job.status = JobStatus.skipped
            job.error = reason
            result.add_job(job)
            result.skipped += 1
            _write_job_diagnostic(app_input, job, config, ctx, started_at=time.monotonic())
            logger.info("[SKIP] %s - %s", source_path.name, reason)
            continue
        if decision == JobStartDecision.conflict:
            job.status = JobStatus.conflict
            job.error = reason
            result.add_job(job)
            result.conflicts += 1
            result.errors.append(f"{source_path.name}: {reason}")
            _write_job_diagnostic(app_input, job, config, ctx, started_at=time.monotonic())
            logger.error("[CONFLICT] %s - %s", source_path.name, reason)
            if not config.general.continue_on_error:
                break
            continue

        job.status = JobStatus.processing
        result.add_job(job)
        started_at = time.monotonic()

        try:
            _process_single_image(app_input, job, config, ctx, started_at)
            if job.qc_warnings:
                result.partial += 1
                result.warnings.extend(
                    f"{source_path.name}: {warning}" for warning in job.qc_warnings
                )
                job.mark_partial()
                logger.info("[PARTIAL] %s", source_path.name)
            else:
                result.succeeded += 1
                job.mark_completed()
                logger.info("[OK] %s", source_path.name)
            _write_job_diagnostic(
                app_input, job, config, ctx, started_at, layouts=job.layouts
            )
        except Exception as exc:
            job.mark_failed(str(exc))
            result.failed += 1
            result.errors.append(f"{source_path.name}: {exc}")
            logger.error("[ERROR] %s - %s", source_path.name, exc)
            _write_job_diagnostic(
                app_input, job, config, ctx, started_at, error=str(exc)
            )
            if not config.general.continue_on_error:
                break

    if config.output.save_metadata:
        write_batch_summary(app_input.output_folder, result, config)
    return result


def _evaluate_job_start(job: ImageJob, config: AppConfig) -> tuple[JobStartDecision, Optional[str]]:
    """Determine whether to skip, conflict, or proceed for a job."""
    if config.output.overwrite_existing:
        return JobStartDecision.proceed, None

    if job.output_path.exists():
        return JobStartDecision.skip, "localized output already exists"

    if (
        config.output.preserve_original
        and job.preserved_original_path is not None
        and job.preserved_original_path.exists()
        and not files.bytes_equal(job.source_path, job.preserved_original_path)
    ):
        return (
            JobStartDecision.conflict,
            "preserved original differs from source and overwrite is disabled",
        )

    return JobStartDecision.proceed, None


def _ensure_preserved_original(job: ImageJob, config: AppConfig) -> None:
    """Write or reuse the byte-identical preserved original copy."""
    if not config.output.preserve_original or job.preserved_original_path is None:
        return

    preserved = job.preserved_original_path
    if preserved.exists():
        if files.bytes_equal(job.source_path, preserved):
            return
        raise RuntimeError(
            "preserved original differs from source and overwrite is disabled"
        )

    files.ensure_parent_folder(preserved)
    files.copy_file_atomic(job.source_path, preserved)


def _process_single_image(
    app_input: AppInput,
    job: ImageJob,
    config: AppConfig,
    ctx: PipelineContext,
    started_at: float,
) -> None:
    _ensure_preserved_original(job, config)

    payload = images.load_image(job.source_path)
    img_bgr, source_alpha = processing_view(payload)
    w, h = images.read_dimensions(job.source_path)
    job.image_width = w
    job.image_height = h

    ocr_engine = ctx.ocr()
    regions = ocr_engine.detect(img_bgr) if ocr_engine is not None else []
    job.text_regions = regions

    if not regions:
        files.ensure_parent_folder(job.output_path)
        images.save_image_atomic(job.output_path, finalize_output(img_bgr, source_alpha, payload))
        _write_job_diagnostic(app_input, job, config, ctx, started_at)
        return

    classify_regions(regions, config.translation, config.classification)

    translator = ctx.translator()
    translate_regions = [r for r in regions if r.action == TextAction.translate]
    if translate_regions:
        if translator is None:
            raise RuntimeError(
                "Translation is required for translatable regions but translator is disabled"
            )
        translations = translator.translate_batch(
            translate_regions,
            config.translation.target_language,
        )
        if len(translations) != len(translate_regions):
            raise RuntimeError(
                f"Translation batch cardinality mismatch: expected "
                f"{len(translate_regions)}, got {len(translations)}"
            )
        for region, translation in zip(translate_regions, translations):
            region.translation = translation

    metadata_qc = validate_region_metadata(regions, config.translation)
    job.metadata_qc = metadata_qc.to_dict()
    if not metadata_qc.passed:
        raise RuntimeError(
            "; ".join(
                issue.message
                for issue in metadata_qc.issues
                if issue.severity.value == "error"
            )
        )

    job.qc_warnings = [
        issue.message
        for issue in metadata_qc.issues
        if issue.severity.value == "warning"
    ]

    mask = None
    cleaned = img_bgr
    processor = ctx.image_processor()
    if processor is not None:
        mask = processor.generate_mask(img_bgr, regions)
        cleaned = processor.remove_text(img_bgr, mask)

    if mask is not None:
        inpaint_report = assess_inpaint_quality(
            img_bgr,
            cleaned,
            mask,
            flatness_threshold=config.quality_control.inpaint_flatness_threshold,
            seam_threshold=config.quality_control.inpaint_seam_threshold,
            spill_threshold=config.quality_control.inpaint_spill_threshold,
        )
        job.inpaint_qc = inpaint_report
        for item in inpaint_report.get("issues", []):
            job.qc_warnings.append(item["message"])

    reviser = ctx.image_reviser()
    layouts: List[dict] = []
    if reviser is not None:
        final_bgr, layouts = reviser.revise(cleaned, img_bgr, regions)
        for layout in layouts:
            for warning in layout.get("style_warnings", []):
                job.qc_warnings.append(f"Style review: {warning}")
    else:
        final_bgr = cleaned

    if config.quality_control.enable_final_pixel_checks:
        final_ocr_engine = (
            ctx.final_ocr_engine()
            if config.quality_control.enable_final_ocr
            else None
        )
        final_qc = validate_final_image(
            img_bgr,
            final_bgr,
            regions,
            layouts,
            config.quality_control,
            final_ocr_engine=final_ocr_engine,
        )
        job.final_qc = final_qc.to_dict()
        job.qc_warnings.extend(
            issue.message
            for issue in final_qc.issues
            if issue.severity.value == "warning"
        )
        if not final_qc.passed:
            raise RuntimeError(
                "; ".join(
                    issue.message
                    for issue in final_qc.issues
                    if issue.severity.value == "error"
                )
            )

    final_payload = finalize_output(final_bgr, source_alpha, payload)
    files.ensure_parent_folder(job.output_path)
    images.save_image_atomic(job.output_path, final_payload)
    job.layouts = layouts
    _save_artifacts(job, config, img_bgr, mask, cleaned, layouts)


def _save_artifacts(job, config, source_image, mask, cleaned_image, layouts):
    output_dir = job.output_path.parent
    if config.output.save_masks and mask is not None:
        images.save_image_atomic(output_dir / "masks" / f"{job.output_path.stem}.png", mask)

    if config.output.save_cleaned_images and cleaned_image is not None:
        images.save_image_atomic(
            output_dir / "cleaned" / f"{job.output_path.stem}.png", cleaned_image
        )


def _write_job_diagnostic(app_input, job, config, ctx, started_at, layouts=None, error=None):
    if not config.output.save_metadata:
        return
    diagnostic = build_image_diagnostic(
        job,
        config,
        started_at=started_at,
        backend_names=ctx.backend_names(),
        metadata_qc=getattr(job, "metadata_qc", None),
        final_qc=getattr(job, "final_qc", None),
        inpaint_qc=getattr(job, "inpaint_qc", None),
        layouts=layouts,
        error=error,
    )
    write_image_diagnostic(app_input.output_folder, job, diagnostic, config)


def _enumerate_images(app_input: AppInput, config: AppConfig) -> List[Path]:
    exts = set(config.input.extensions)
    if app_input.input_type == InputType.SINGLE_IMAGE:
        return [app_input.single_image_path]
    return folders.enumerate_images(
        app_input.input_folder,
        exts,
        recursive=config.input.recursive,
        exclude_suffix=config.output.suffix,
    )
