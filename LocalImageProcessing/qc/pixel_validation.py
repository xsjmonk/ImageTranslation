"""Pixel-level output validation without OCR."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

import numpy as np

from ..contract import RegionAction
from ..regions import ProcessRegion
from ..revision.render_validation import validate_rendered_layout


class QcSeverity(str, Enum):
    error = "error"
    warning = "warning"


@dataclass
class QcIssue:
    severity: QcSeverity
    code: str
    message: str
    region_id: str = ""


@dataclass
class QcResult:
    passed: bool
    issues: List[QcIssue] = field(default_factory=list)


@dataclass
class PixelQcOptions:
    preserved_region_tolerance: float = 12.0
    clip_tolerance_pixels: int = 2
    rendered_text_min_alpha_coverage: float = 0.005
    max_glyph_outside_polygon_ratio: float = 0.05


def validate_final_image(
    source_image: np.ndarray,
    final_image: np.ndarray,
    regions: List[ProcessRegion],
    layouts: List[Dict],
    options: PixelQcOptions,
) -> QcResult:
    issues: List[QcIssue] = []

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
            clip_tolerance_pixels=options.clip_tolerance_pixels,
            min_alpha_coverage=options.rendered_text_min_alpha_coverage,
            max_outside_polygon_ratio=options.max_glyph_outside_polygon_ratio,
        ):
            issues.append(
                QcIssue(
                    severity=QcSeverity(item["severity"]),
                    code=item["code"],
                    message=item["message"],
                    region_id=item.get("region_id", ""),
                )
            )

    for region in regions:
        if region.action != RegionAction.translate:
            continue
        if not region.translated_text or not region.translated_text.replace(" ", "").isascii():
            issues.append(
                QcIssue(
                    severity=QcSeverity.error,
                    code="missing_english_output",
                    message=f"No readable English for region: {region.source_text[:40]}",
                    region_id=region.id,
                )
            )

    for region in regions:
        if region.action not in (RegionAction.preserve, RegionAction.review):
            continue
        diff = _region_mean_abs_diff(source_image, final_image, region.polygon)
        if diff is not None and diff > options.preserved_region_tolerance:
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

    passed = not any(issue.severity == QcSeverity.error for issue in issues)
    return QcResult(passed=passed, issues=issues)


def _region_mean_abs_diff(
    source: np.ndarray,
    final: np.ndarray,
    polygon: list,
) -> float | None:
    crop_s = _region_crop(source, polygon)
    crop_f = _region_crop(final, polygon)
    if crop_s is None or crop_f is None or crop_s.shape != crop_f.shape:
        return None
    return float(np.mean(np.abs(crop_s.astype(np.float32) - crop_f.astype(np.float32))))


def _region_crop(image: np.ndarray, polygon: list) -> np.ndarray | None:
    poly = np.array(polygon, dtype=np.int32)
    xs, ys = poly[:, 0], poly[:, 1]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    h, w = image.shape[:2]
    if x2 <= x1 or y2 <= y1:
        return None
    return image[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]
