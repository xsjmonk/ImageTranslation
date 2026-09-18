"""Post-render validation using actual glyph alpha masks and oriented geometry."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..utilities.geometry import alpha_outside_polygon_ratio, oriented_layout_polygon


def measure_alpha_bounds(layer_rgba: np.ndarray) -> Optional[Dict[str, float]]:
    """Measure tight bounds of non-transparent pixels in an RGBA layer."""
    if layer_rgba is None or layer_rgba.ndim != 3 or layer_rgba.shape[2] < 4:
        return None
    alpha = layer_rgba[:, :, 3]
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return None
    return {
        "x1": float(xs.min()),
        "y1": float(ys.min()),
        "x2": float(xs.max()) + 1.0,
        "y2": float(ys.max()) + 1.0,
        "alpha_pixels": int(len(xs)),
        "alpha_coverage": float(len(xs)) / float(alpha.size),
    }


def validate_rendered_layout(
    layout: dict,
    polygon: List[List[float]],
    *,
    clip_tolerance_pixels: float = 2.0,
    min_alpha_coverage: float = 0.005,
    max_outside_polygon_ratio: float = 0.05,
) -> List[dict]:
    """Validate a rendered region using measured alpha mask geometry."""
    issues: List[dict] = []
    layer = layout.get("render_layer")
    bounds = layout.get("render_bounds")

    if layer is None and bounds is None:
        issues.append(
            {
                "severity": "error",
                "code": "missing_rendered_text",
                "message": "No rendered glyph alpha detected in translated region",
                "region_id": layout.get("region_id", ""),
            }
        )
        return issues

    if bounds and bounds.get("alpha_coverage", 0.0) < min_alpha_coverage:
        issues.append(
            {
                "severity": "error",
                "code": "missing_rendered_text",
                "message": "Rendered English is not visibly present in the intended region",
                "region_id": layout.get("region_id", ""),
            }
        )

    if layer is not None:
        stroke_margin = float(layout.get("stroke_margin", 0))
        allowed = _allowed_layout_polygon(
            layout,
            polygon,
            clip_tolerance_pixels + stroke_margin,
        )
        outside_ratio, alpha_pixels = alpha_outside_polygon_ratio(
            layer,
            allowed,
            margin=clip_tolerance_pixels,
        )
        layout["polygon_outside_ratio"] = outside_ratio
        layout["polygon_alpha_pixels"] = alpha_pixels
        if alpha_pixels > 0 and outside_ratio > max_outside_polygon_ratio:
            issues.append(
                {
                    "severity": "error",
                    "code": "text_clipped",
                    "message": (
                        "Rendered glyph alpha escapes the oriented source region "
                        f"({outside_ratio:.1%} outside polygon)"
                    ),
                    "region_id": layout.get("region_id", ""),
                }
            )

    if layout.get("fit_warning"):
        issues.append(
            {
                "severity": "warning",
                "code": "text_fit_warning",
                "message": layout["fit_warning"],
                "region_id": layout.get("region_id", ""),
            }
        )

    return issues


def _allowed_layout_polygon(
    layout: dict,
    polygon: List[List[float]],
    tolerance: float,
) -> List[List[float]]:
    center = layout.get("center")
    width = layout.get("width")
    height = layout.get("height")
    if center is None or width is None or height is None:
        return polygon
    return oriented_layout_polygon(
        tuple(center),
        float(width),
        float(height),
        float(layout.get("angle", 0.0)),
        margin=tolerance,
    )
