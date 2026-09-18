"""Deterministic pixel processing from agent manifests."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .blur import apply_feathered_blur
from .contract import (
    BatchManifest,
    ImageManifest,
    ReconstructionMethod,
    RegionAction,
    RegionManifest,
    validate_source_hash,
)
from .diagnostics import build_batch_summary, build_image_diagnostic
from .imaging import assess_inpaint_quality, create_text_mask, inpaint_navier_stokes
from .io import (
    bytes_equal,
    copy_file_atomic,
    ensure_parent_folder,
    finalize_output,
    load_image,
    processing_view,
    save_image_atomic,
    save_json_atomic,
)
from .io_paths import resolve_output_file_path, resolve_preserved_original_path
from .qc import PixelQcOptions, validate_final_image
from .regions import ProcessRegion, from_manifest
from .revision import ImageReviser
from .validation import validate_image_manifest

logger = logging.getLogger(__name__)


@dataclass
class ProcessingOptions:
    output_root: Path
    mask_expansion_pixels: int = 3
    preserve_original: bool = True
    overwrite_existing: bool = False
    continue_on_error: bool = True
    enable_pixel_checks: bool = True
    promote: bool = False
    preserved_region_tolerance: float = 12.0
    clip_tolerance_pixels: int = 2


@dataclass
class ImageProcessingResult:
    source_path: Path
    status: str
    candidate_output_path: Optional[Path] = None
    preserved_original_path: Optional[Path] = None
    promoted_output_path: Optional[Path] = None
    diagnostic: Dict = field(default_factory=dict)
    error: Optional[str] = None


def process_batch(
    manifest: BatchManifest,
    options: ProcessingOptions,
) -> Dict:
    output_root = options.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    image_results: List[Dict] = []
    for image_manifest in manifest.images:
        started_at = time.monotonic()
        source_path = image_manifest.source_path.resolve()
        result = ImageProcessingResult(source_path=source_path, status="failure")

        try:
            if not source_path.is_file():
                raise FileNotFoundError(f"source image not found: {source_path}")

            validate_source_hash(source_path, image_manifest.source_hash)

            payload = load_image(source_path)
            img_bgr, source_alpha = processing_view(payload)
            height, width = img_bgr.shape[:2]

            validation = validate_image_manifest(image_manifest, width, height)
            if not validation.passed:
                raise ValueError(
                    "; ".join(
                        issue.message
                        for issue in validation.issues
                        if issue.severity == "error"
                    )
                )

            output_path = resolve_output_file_path(source_path, output_root)
            preserved_path = resolve_preserved_original_path(source_path, output_root)
            result.candidate_output_path = output_path
            result.preserved_original_path = preserved_path

            if output_path.exists() and not options.overwrite_existing:
                result.status = "skipped"
                result.diagnostic = build_image_diagnostic(
                    status="skipped",
                    source_path=source_path,
                    candidate_output_path=output_path,
                    preserved_original_path=preserved_path,
                    promoted_output_path=None,
                    regions=[],
                    methods=[],
                    validation_issues=[],
                    warnings=["localized output already exists"],
                    error=None,
                    started_at=started_at,
                    image_width=0,
                    image_height=0,
                )
                image_results.append(_result_summary(result))
                _write_image_diagnostic(metadata_dir, source_path, result.diagnostic)
                continue

            _ensure_preserved_original(source_path, preserved_path, options.preserve_original)

            regions = [from_manifest(region) for region in image_manifest.regions]
            warnings: List[str] = [
                issue.message
                for issue in validation.issues
                if issue.severity == "warning"
            ]

            cleaned, _mask, methods, inpaint_warnings = _reconstruct_background(
                img_bgr,
                image_manifest.regions,
                options.mask_expansion_pixels,
            )
            warnings.extend(inpaint_warnings)

            reviser = ImageReviser()
            final_bgr, layouts = reviser.revise(cleaned, regions)

            if options.enable_pixel_checks:
                qc_result = validate_final_image(
                    img_bgr,
                    final_bgr,
                    regions,
                    layouts,
                    PixelQcOptions(
                        preserved_region_tolerance=options.preserved_region_tolerance,
                        clip_tolerance_pixels=options.clip_tolerance_pixels,
                    ),
                )
                errors = [issue for issue in qc_result.issues if issue.severity.value == "error"]
                warnings.extend(
                    issue.message
                    for issue in qc_result.issues
                    if issue.severity.value == "warning"
                )
                if errors:
                    raise ValueError("; ".join(issue.message for issue in errors))

            final_payload = finalize_output(final_bgr, source_alpha, payload)
            ensure_parent_folder(output_path)
            save_image_atomic(output_path, final_payload)

            status = _resolve_status(image_manifest.regions, methods, warnings)
            promoted_path = None
            if options.promote:
                promoted_path = _promote_output(source_path, output_path, preserved_path)
                result.promoted_output_path = promoted_path

            result.status = status
            result.diagnostic = build_image_diagnostic(
                status=status,
                source_path=source_path,
                candidate_output_path=output_path,
                preserved_original_path=preserved_path,
                promoted_output_path=promoted_path,
                regions=[region.model_dump(mode="json") for region in image_manifest.regions],
                methods=methods,
                validation_issues=[issue.__dict__ for issue in validation.issues],
                warnings=warnings,
                error=None,
                started_at=started_at,
                image_width=width,
                image_height=height,
            )
        except Exception as exc:
            logger.exception("Failed to process %s", source_path)
            result.status = "failure"
            result.error = str(exc)
            result.diagnostic = build_image_diagnostic(
                status="failure",
                source_path=source_path,
                candidate_output_path=result.candidate_output_path,
                preserved_original_path=result.preserved_original_path,
                promoted_output_path=None,
                regions=[region.model_dump(mode="json") for region in image_manifest.regions],
                methods=[],
                validation_issues=[],
                warnings=[],
                error=str(exc),
                started_at=started_at,
                image_width=0,
                image_height=0,
            )
            if not options.continue_on_error:
                image_results.append(_result_summary(result))
                _write_image_diagnostic(metadata_dir, source_path, result.diagnostic)
                break

        image_results.append(_result_summary(result))
        _write_image_diagnostic(metadata_dir, source_path, result.diagnostic)

    summary = build_batch_summary(image_results)
    save_json_atomic(output_root / "summary.json", summary)
    return summary


def _ensure_preserved_original(
    source_path: Path,
    preserved_path: Path,
    preserve: bool,
) -> None:
    if not preserve:
        return
    if preserved_path.exists():
        if bytes_equal(source_path, preserved_path):
            return
        raise RuntimeError(
            "preserved original differs from source and overwrite is disabled"
        )
    ensure_parent_folder(preserved_path)
    copy_file_atomic(source_path, preserved_path)


def _reconstruct_background(
    image: np.ndarray,
    regions: List[RegionManifest],
    mask_expansion_pixels: int,
):
    editable = [
        from_manifest(region)
        for region in regions
        if region.action in (RegionAction.translate, RegionAction.remove)
    ]
    if not editable:
        return image.copy(), None, [], []

    mask = create_text_mask(image, editable, expansion_pixels=mask_expansion_pixels)
    cleaned = inpaint_navier_stokes(image, mask)
    methods: List[Dict] = [{"method": "inpaint", "backend": "opencv_navier_stokes"}]
    warnings: List[str] = []

    blur_regions = [
        region
        for region in regions
        if region.action in (RegionAction.translate, RegionAction.remove)
        and region.reconstruction
        and region.reconstruction.method == ReconstructionMethod.blur_fallback
    ]
    if blur_regions:
        blur_mask = create_text_mask(
            image,
            [from_manifest(region) for region in blur_regions],
            expansion_pixels=mask_expansion_pixels,
        )
        sigma = (
            blur_regions[0].reconstruction.blur_sigma
            if blur_regions[0].reconstruction
            else 12.0
        )
        cleaned = apply_feathered_blur(cleaned, blur_mask, sigma=sigma)
        methods.append(
            {
                "method": "blur_fallback",
                "sigma": sigma,
                "region_ids": [region.id for region in blur_regions],
            }
        )
        warnings.append(
            "Localized Gaussian blur fallback was applied for one or more regions"
        )

    inpaint_report = assess_inpaint_quality(image, cleaned, mask)
    for item in inpaint_report.get("issues", []):
        warnings.append(item["message"])

    return cleaned, mask, methods, warnings


def _resolve_status(regions, methods, warnings) -> str:
    if any(region.action == RegionAction.review for region in regions):
        return "review"
    if any(method.get("method") == "blur_fallback" for method in methods):
        return "partial"
    if warnings:
        return "partial"
    return "success"


def _promote_output(source_path: Path, candidate_path: Path, preserved_path: Path) -> Path:
    if not preserved_path.exists():
        copy_file_atomic(source_path, preserved_path)
    elif not bytes_equal(source_path, preserved_path):
        raise RuntimeError("cannot promote: preserved original differs from source")

    copy_file_atomic(candidate_path, source_path)
    return source_path


def _result_summary(result: ImageProcessingResult) -> Dict:
    return {
        "source_path": str(result.source_path),
        "status": result.status,
        "candidate_output_path": (
            str(result.candidate_output_path) if result.candidate_output_path else None
        ),
        "preserved_original_path": (
            str(result.preserved_original_path) if result.preserved_original_path else None
        ),
        "promoted_output_path": (
            str(result.promoted_output_path) if result.promoted_output_path else None
        ),
        "error": result.error,
    }


def _write_image_diagnostic(metadata_dir: Path, source_path: Path, diagnostic: Dict) -> Path:
    path = metadata_dir / f"{source_path.stem}.json"
    save_json_atomic(path, diagnostic)
    return path
