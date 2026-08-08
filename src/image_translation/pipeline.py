"""Pipeline orchestration – coordinates modules without owning low-level logic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set

from .config import AppConfig
from .imaging import HybridImageProcessor
from .input import AppInput, InputType
from .models.image_job import ImageJob, JobStatus
from .models.processing_result import ProcessingResult
from .models.text_region import TextAction
from .ocr import OcrEngine, PaddleOcrEngine
from .revision import ImageReviser
from .translation import Translator, NoopTranslator, classify_regions
from .utilities import files, folders, images, json_utils

logger = logging.getLogger(__name__)


def run_pipeline(app_input: AppInput, config: AppConfig) -> ProcessingResult:
    """Execute the full image translation pipeline.

    Args:
        app_input: Resolved application input.
        config: Validated configuration.

    Returns:
        ProcessingResult with per-image job status.
    """
    result = ProcessingResult()

    # Enumerate images
    image_paths = _enumerate_images(app_input, config)
    if not image_paths:
        logger.warning("No images found in: %s", app_input.input_path)
        return result

    logger.info("Found %d image(s).", len(image_paths))

    # Lazy-initialize heavy services
    ocr_engine: Optional[OcrEngine] = None
    image_processor: Optional[HybridImageProcessor] = None
    image_reviser: Optional[ImageReviser] = None
    translator: Optional[Translator] = None

    for source_path in image_paths:
        output_path = folders.derive_output_path(
            source_path, app_input.output_folder, app_input.input_folder
        )

        # Skip if output exists and overwrite is disabled
        if output_path.exists() and not config.output.overwrite_existing:
            logger.info("[SKIP] %s - output already exists", source_path.name)
            job = ImageJob(
                source_path=source_path,
                output_path=output_path,
                status=JobStatus.completed,
            )
            result.add_job(job)
            result.skipped += 1
            continue

        job = ImageJob(
            source_path=source_path,
            output_path=output_path,
            status=JobStatus.processing,
        )
        result.add_job(job)

        try:
            _process_single_image(
                job=job,
                config=config,
                ocr_engine=ocr_engine,
                image_processor=image_processor,
                image_reviser=image_reviser,
                translator=translator,
            )
            # Store initialized engines for reuse
            if ocr_engine is None:
                ocr_engine = _init_ocr(config)
            if image_processor is None:
                image_processor = _init_imaging(config)
            if image_reviser is None:
                image_reviser = _init_revision(config)
            if translator is None:
                translator = _init_translator()

        except Exception as exc:
            job.mark_failed(str(exc))
            logger.error("[ERROR] %s - %s", source_path.name, exc)
            result.failed += 1
            result.errors.append(f"{source_path.name}: {exc}")

            if not config.general.continue_on_error:
                break
            continue

        job.mark_completed()
        result.succeeded += 1
        logger.info("[OK] %s", source_path.name)

    return result


def _process_single_image(
    job: ImageJob,
    config: AppConfig,
    ocr_engine: Optional[OcrEngine] = None,
    image_processor: Optional[HybridImageProcessor] = None,
    image_reviser: Optional[ImageReviser] = None,
    translator: Optional[Translator] = None,
) -> None:
    """Process one image through OCR → classify → translate → clean → revise."""
    # Load image and get dimensions
    img = images.load_image(job.source_path)
    w, h = images.read_dimensions(job.source_path)
    job.image_width = w
    job.image_height = h

    # 1. OCR
    if config.ocr.enabled and ocr_engine is None:
        ocr_engine = _init_ocr(config)
    if ocr_engine is not None:
        regions = ocr_engine.detect(img)
        job.text_regions = regions
    else:
        regions = job.text_regions

    if not regions:
        # No text found – just copy the image
        images.save_image(job.output_path, img)
        _save_artifacts(job, config, img, None, None)
        return

    # 2. Classify
    classify_regions(regions, config.translation)

    # 3. Translate
    if config.translation.enabled and translator is None:
        translator = _init_translator()
    if translator is not None:
        translate_regions = [r for r in regions if r.action == TextAction.translate]
        for region in translate_regions:
            region.translation = translator.translate(region, config.translation.target_language)

    # Count review regions
    review_count = sum(1 for r in regions if r.action == TextAction.review)
    if review_count > 0:
        logger.info("[REVIEW] %s - %d text region(s) require review", job.source_name, review_count)

    # 4. Generate mask and remove text
    mask = None
    cleaned = img
    if config.imaging.enabled and image_processor is None:
        image_processor = _init_imaging(config)
    if image_processor is not None:
        mask = image_processor.generate_mask(img, regions)
        cleaned = image_processor.remove_text(img, mask)

    # 5. Revise (render translated text)
    if config.revision.enabled and image_reviser is None:
        image_reviser = _init_revision(config)
    if image_reviser is not None:
        final = image_reviser.revise(cleaned, regions)
    else:
        final = cleaned

    # 6. Save final image
    files.ensure_parent_folder(job.output_path)
    images.save_image(job.output_path, final)

    # 7. Save artifacts
    _save_artifacts(job, config, img, mask, cleaned)


def _save_artifacts(
    job: ImageJob,
    config: AppConfig,
    source_image,
    mask,
    cleaned_image,
) -> None:
    """Save metadata, mask, and cleaned intermediate images."""
    output_dir = job.output_path.parent

    # Metadata
    if config.output.save_metadata:
        meta_dir = output_dir / "metadata"
        meta_path = meta_dir / f"{job.output_path.stem}.json"
        json_utils.save_json(meta_path, job.to_dict())

    # Mask
    if config.output.save_masks and mask is not None:
        mask_dir = output_dir / "masks"
        mask_path = mask_dir / f"{job.output_path.stem}.png"
        images.save_image(mask_path, mask)

    # Cleaned intermediate
    if config.output.save_cleaned_images and cleaned_image is not None:
        cleaned_dir = output_dir / "cleaned"
        cleaned_path = cleaned_dir / f"{job.output_path.stem}.png"
        images.save_image(cleaned_path, cleaned_image)


def _enumerate_images(app_input: AppInput, config: AppConfig) -> List[Path]:
    """List all image files to process."""
    exts = set(config.input.extensions)
    if app_input.input_type == InputType.SINGLE_IMAGE:
        return [app_input.single_image_path]
    return folders.enumerate_images(
        app_input.input_folder,
        exts,
        recursive=config.input.recursive,
    )


# ---- Lazy initializers ----

_ocr_instance: Optional[OcrEngine] = None
_imaging_instance: Optional[HybridImageProcessor] = None
_revision_instance: Optional[ImageReviser] = None
_translator_instance: Optional[Translator] = None


def _init_ocr(config: AppConfig) -> OcrEngine:
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOcrEngine(config.ocr)
    return _ocr_instance


def _init_imaging(config: AppConfig) -> HybridImageProcessor:
    global _imaging_instance
    if _imaging_instance is None:
        _imaging_instance = HybridImageProcessor(config.imaging)
    return _imaging_instance


def _init_revision(config: AppConfig) -> ImageReviser:
    global _revision_instance
    if _revision_instance is None:
        _revision_instance = ImageReviser(config.revision)
    return _revision_instance


def _init_translator() -> Translator:
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = NoopTranslator()
    return _translator_instance
