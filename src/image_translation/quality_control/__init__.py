"""Quality control for localized product images."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from ..config.models import QcConfig
from ..models.text_region import TextAction, TextRegion
from ..revision.render_validation import validate_rendered_layout
from .final_ocr import validate_final_image_ocr
from .types import QcIssue, QcResult, QcSeverity

if TYPE_CHECKING:
    from ..ocr.base import OcrEngine

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
_BRACKETED_SOURCE_RE = re.compile(r"^\[[^\]]+\]$")


def _merge_results(*results: QcResult) -> QcResult:
    issues: List[QcIssue] = []
    evidence: Dict[str, Any] = {}
    for result in results:
        issues.extend(result.issues)
        if result.evidence:
            evidence.update(result.evidence)
    passed = not any(i.severity == QcSeverity.error for i in issues)
    return QcResult(passed=passed, issues=issues, evidence=evidence)


def validate_region_metadata(
    regions: List[TextRegion],
    translation_config=None,
) -> QcResult:
    """Validate OCR/classification/translation metadata before rendering."""
    issues: List[QcIssue] = []

    for region in regions:
        text = region.source_text.strip()
        if not text:
            continue

        if region.action == TextAction.review:
            issues.append(
                QcIssue(
                    severity=QcSeverity.warning,
                    code="region_requires_review",
                    message=f"Region requires manual review: {text[:80]}",
                    region_id=region.id,
                )
            )
            continue

        if region.action != TextAction.translate:
            continue

        translation = region.translation or {}
        translated = (
            translation.get("translated_text")
            or translation.get("compact_text")
            or ""
        ).strip()

        if not translated:
            issues.append(
                QcIssue(
                    severity=QcSeverity.error,
                    code="missing_translation",
                    message=f"No English translation for: {text[:80]}",
                    region_id=region.id,
                )
            )
            continue

        if region.confidence < 0.5:
            issues.append(
                QcIssue(
                    severity=QcSeverity.warning,
                    code="low_ocr_confidence",
                    message=(
                        f"Low OCR confidence ({region.confidence:.2f}) for: {text[:80]}"
                    ),
                    region_id=region.id,
                )
            )

    base = QcResult(
        passed=not any(i.severity == QcSeverity.error for i in issues),
        issues=issues,
    )

    from ..config.models import TranslationConfig
    from ..translation.validation import validate_translations

    cfg = translation_config if translation_config is not None else TranslationConfig()
    return _merge_results(base, validate_translations(regions, cfg))


def validate_final_image(
    source_image: np.ndarray,
    final_image: np.ndarray,
    regions: List[TextRegion],
    layouts: List[Dict],
    config: QcConfig,
    final_ocr_engine: Optional["OcrEngine"] = None,
) -> QcResult:
    """Validate rendered output pixels, glyph geometry, OCR, and preserved regions."""
    issues: List[QcIssue] = []
    evidence: Dict[str, Any] = {}

    if source_image.shape[:2] != final_image.shape[:2]:
        issues.append(
            QcIssue(
                severity=QcSeverity.error,
                code="dimension_mismatch",
                message=(
                    f"Output dimensions {final_image.shape[:2]} "
                    f"!= source {source_image.shape[:2]}"
                ),
            )
        )

    region_by_id = {region.id: region for region in regions}
    for layout in layouts:
        region = region_by_id.get(str(layout.get("region_id", "")))
        if region is None:
            continue
        for item in validate_rendered_layout(
            layout,
            region.polygon,
            clip_tolerance_pixels=config.clip_tolerance_pixels,
            min_alpha_coverage=config.rendered_text_min_alpha_coverage,
            max_outside_polygon_ratio=config.max_glyph_outside_polygon_ratio,
        ):
            severity = QcSeverity(item["severity"])
            issues.append(
                QcIssue(
                    severity=severity,
                    code=item["code"],
                    message=item["message"],
                    region_id=item.get("region_id", ""),
                )
            )

    for region in regions:
        if region.action != TextAction.translate:
            continue
        translated = (
            (region.translation or {}).get("translated_text")
            or (region.translation or {}).get("compact_text")
            or ""
        ).strip()
        if _BRACKETED_SOURCE_RE.match(translated):
            issues.append(
                QcIssue(
                    severity=QcSeverity.error,
                    code="placeholder_in_output",
                    message=f"Placeholder text would render for: {region.source_text[:40]}",
                    region_id=region.id,
                )
            )
        if not translated or not translated.replace(" ", "").isascii():
            issues.append(
                QcIssue(
                    severity=QcSeverity.error,
                    code="missing_english_output",
                    message=f"No readable English for region: {region.source_text[:40]}",
                    region_id=region.id,
                )
            )

    for region in regions:
        if region.action not in (TextAction.preserve, TextAction.review):
            continue
        diff = _region_mean_abs_diff(source_image, final_image, region.polygon)
        if diff is not None and diff > config.preserved_region_tolerance:
            issues.append(
                QcIssue(
                    severity=QcSeverity.error,
                    code="preserved_region_changed",
                    message=(
                        f"Preserved region changed by {diff:.1f} px mean diff: "
                        f"{region.source_text[:40]}"
                    ),
                    region_id=region.id,
                )
            )

    if (
        config.enable_final_ocr
        and final_ocr_engine is not None
        and any(r.action == TextAction.translate for r in regions)
    ):
        ocr_result, ocr_evidence = validate_final_image_ocr(
            final_image,
            regions,
            final_ocr_engine,
            min_confidence=config.final_ocr_min_confidence,
            coverage_threshold=config.final_ocr_cjk_coverage_threshold,
            overlap_threshold=config.final_ocr_overlap_threshold,
        )
        issues.extend(ocr_result.issues)
        evidence["final_ocr"] = ocr_evidence

    passed = not any(i.severity == QcSeverity.error for i in issues)
    return QcResult(passed=passed, issues=issues, evidence=evidence)


validate_image_regions = validate_region_metadata

__all__ = [
    "QcIssue",
    "QcResult",
    "QcSeverity",
    "validate_final_image",
    "validate_image_regions",
    "validate_region_metadata",
]


def _region_mean_abs_diff(
    source: np.ndarray, final: np.ndarray, polygon: list
) -> Optional[float]:
    crop_s = _region_crop(source, polygon)
    crop_f = _region_crop(final, polygon)
    if crop_s is None or crop_f is None or crop_s.shape != crop_f.shape:
        return None
    return float(np.mean(np.abs(crop_s.astype(np.float32) - crop_f.astype(np.float32))))


def _region_crop(image: np.ndarray, polygon: list) -> Optional[np.ndarray]:
    poly = np.array(polygon, dtype=np.int32)
    xs, ys = poly[:, 0], poly[:, 1]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    h, w = image.shape[:2]
    if x2 <= x1 or y2 <= y1:
        return None
    return image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
