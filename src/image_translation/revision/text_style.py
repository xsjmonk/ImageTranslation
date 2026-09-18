"""Text style estimation and structured rendering attributes."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TextStyle:
    color_rgb: Tuple[int, int, int] = (255, 255, 255)
    opacity: float = 1.0
    bold: bool = False
    italic_oblique: float = 0.0
    stroke_width: int = 0
    stroke_color_rgb: Tuple[int, int, int] = (0, 0, 0)
    shadow_offset: Tuple[int, int] = (0, 0)
    shadow_blur: int = 0
    shadow_color_rgb: Tuple[int, int, int] = (0, 0, 0)
    shadow_opacity: float = 0.0
    gradient_top_rgb: Optional[Tuple[int, int, int]] = None
    gradient_bottom_rgb: Optional[Tuple[int, int, int]] = None
    alignment: str = "center"
    line_spacing: float = 1.2
    confidence: float = 0.0
    recovered_effects: List[str] = field(default_factory=list)
    review_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_style_from_crop(image_bgr: np.ndarray, bbox: Optional[list]) -> TextStyle:
    """Estimate foreground color, stroke, shadow, and weight from a source crop."""
    style = TextStyle()
    if bbox is None or image_bgr is None:
        style.review_warnings.append("missing_source_crop")
        return style

    x1, y1, x2, y2 = bbox
    h, w = image_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        style.review_warnings.append("invalid_source_crop")
        return style

    crop = image_bgr[y1:y2, x1:x2]
    gray = crop.mean(axis=2)
    dark_mask = gray < 128
    bright_mask = gray > 180

    if np.any(dark_mask):
        pixels = crop[dark_mask]
        style.color_rgb = tuple(int(v) for v in pixels.mean(axis=0)[::-1])
        style.bold = float(np.std(gray[dark_mask])) > 35.0
        style.recovered_effects.append("foreground_color")
        if style.bold:
            style.recovered_effects.append("weight")
    elif np.any(bright_mask):
        pixels = crop[bright_mask]
        style.color_rgb = tuple(int(v) for v in pixels.mean(axis=0)[::-1])
        style.recovered_effects.append("foreground_color")
    else:
        style.review_warnings.append("foreground_color_uncertain")

    stroke = _estimate_stroke(crop, gray, dark_mask if np.any(dark_mask) else bright_mask)
    if stroke > 0:
        style.stroke_width = stroke
        style.stroke_color_rgb = _estimate_stroke_color(crop, gray)
        style.recovered_effects.append("stroke")

    shadow = _estimate_shadow(crop, gray)
    if shadow is not None:
        style.shadow_offset = shadow["offset"]
        style.shadow_blur = shadow["blur"]
        style.shadow_color_rgb = shadow["color"]
        style.shadow_opacity = shadow["opacity"]
        style.recovered_effects.append("shadow")

    gradient = _estimate_simple_gradient(crop, gray)
    if gradient is not None:
        style.gradient_top_rgb, style.gradient_bottom_rgb = gradient
        style.recovered_effects.append("gradient")

    if not style.recovered_effects:
        style.review_warnings.append("style_recovery_low_confidence")

    style.confidence = min(1.0, len(style.recovered_effects) * 0.25)
    return style


def _estimate_stroke(crop: np.ndarray, gray: np.ndarray, text_mask: np.ndarray) -> int:
    try:
        import cv2
    except ImportError:
        return 0
    if not np.any(text_mask):
        return 0
    edges = cv2.Canny(gray.astype(np.uint8), 40, 120)
    dilated = cv2.dilate(text_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    halo = (dilated > 0) & (~text_mask) & (edges > 0)
    if float(np.mean(halo)) > 0.02:
        return 2 if float(np.mean(halo)) > 0.05 else 1
    return 0


def _estimate_stroke_color(crop: np.ndarray, gray: np.ndarray) -> Tuple[int, int, int]:
    edge_mask = gray < 90
    if not np.any(edge_mask):
        return (0, 0, 0)
    pixels = crop[edge_mask]
    return tuple(int(v) for v in pixels.mean(axis=0)[::-1])


def _estimate_shadow(crop: np.ndarray, gray: np.ndarray) -> Optional[dict]:
    if crop.shape[0] < 6 or crop.shape[1] < 6:
        return None
    lower = gray[int(gray.shape[0] * 0.55) :, :]
    upper = gray[: int(gray.shape[0] * 0.45), :]
    if float(np.mean(lower)) + 8 < float(np.mean(upper)):
        return {
            "offset": (1, 2),
            "blur": 2,
            "color": (0, 0, 0),
            "opacity": 0.45,
        }
    return None


def _estimate_simple_gradient(
    crop: np.ndarray, gray: np.ndarray
) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    top = gray[: max(1, gray.shape[0] // 3), :]
    bottom = gray[min(gray.shape[0] - 1, gray.shape[0] * 2 // 3) :, :]
    if abs(float(np.mean(top)) - float(np.mean(bottom))) < 18:
        return None
    top_rgb = tuple(int(v) for v in crop[: top.shape[0], :].mean(axis=(0, 1))[::-1])
    bottom_rgb = tuple(int(v) for v in crop[-bottom.shape[0] :, :].mean(axis=(0, 1))[::-1])
    return top_rgb, bottom_rgb
