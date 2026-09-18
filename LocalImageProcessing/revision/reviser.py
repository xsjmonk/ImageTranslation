"""Composite supplied English onto cleaned images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..contract import RegionAction
from ..regions import ProcessRegion
from .layout import LayoutEngine
from .render_validation import measure_alpha_bounds
from .text_renderer import TextRenderer
from .text_style import TextStyle


@dataclass
class RevisionOptions:
    minimum_font_size: int = 12
    maximum_font_size: int = 48
    allow_multiline: bool = True
    line_spacing: float = 1.2
    clip_tolerance_pixels: int = 2
    font_path: Optional[str] = None
    bold_font_path: Optional[str] = None


class ImageReviser:
    def __init__(self, options: RevisionOptions | None = None) -> None:
        opts = options or RevisionOptions()
        self._layout_engine = LayoutEngine(
            minimum_font_size=opts.minimum_font_size,
            maximum_font_size=opts.maximum_font_size,
            allow_multiline=opts.allow_multiline,
            line_spacing=opts.line_spacing,
            clip_tolerance_pixels=opts.clip_tolerance_pixels,
            font_path=opts.font_path,
            bold_font_path=opts.bold_font_path,
        )
        self._text_renderer = TextRenderer(
            font_path=opts.font_path,
            bold_font_path=opts.bold_font_path,
            line_spacing=opts.line_spacing,
        )

    def revise(
        self,
        cleaned_image: np.ndarray,
        regions: List[ProcessRegion],
    ) -> Tuple[np.ndarray, List[dict]]:
        h, w = cleaned_image.shape[:2]
        result = cleaned_image.copy()
        layouts: List[dict] = []

        for region in regions:
            if region.action != RegionAction.translate or not region.translated_text:
                continue

            style = _style_from_region(region)
            layout = self._layout_engine.compute_layout(
                region.polygon,
                region.translated_text,
                region_id=region.id,
                bold=style.bold,
            )
            text_layer = self._text_renderer.render_text_layer(w, h, layout, style)
            layout["render_layer"] = text_layer
            layout["render_bounds"] = measure_alpha_bounds(text_layer)
            shadow_extent = max(abs(style.shadow_offset[0]), abs(style.shadow_offset[1]))
            layout["stroke_margin"] = (
                style.stroke_width + max(style.shadow_blur, 0) + shadow_extent + 2
            )
            layouts.append(layout)
            result = _composite(result, text_layer)

        return result, layouts


def _style_from_region(region: ProcessRegion) -> TextStyle:
    style_data = region.style or {}
    style = TextStyle(
        color_rgb=tuple(style_data.get("color_rgb", (255, 255, 255))),
        opacity=float(style_data.get("opacity", 1.0)),
        bold=bool(style_data.get("bold", False)),
        italic_oblique=float(style_data.get("italic_oblique", 0.0)),
        stroke_width=int(style_data.get("stroke_width", 0)),
        stroke_color_rgb=tuple(style_data.get("stroke_color_rgb", (0, 0, 0))),
        shadow_offset=tuple(style_data.get("shadow_offset", (0, 0))),
        shadow_blur=int(style_data.get("shadow_blur", 0)),
        shadow_color_rgb=tuple(style_data.get("shadow_color_rgb", (0, 0, 0))),
        shadow_opacity=float(style_data.get("shadow_opacity", 0.0)),
    )
    if style_data.get("gradient_top_rgb") and style_data.get("gradient_bottom_rgb"):
        style.gradient_top_rgb = tuple(style_data["gradient_top_rgb"])
        style.gradient_bottom_rgb = tuple(style_data["gradient_bottom_rgb"])
    if style_data.get("alignment"):
        style.alignment = str(style_data["alignment"])
    return style


def _composite(image: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    if overlay_rgba.shape[2] != 4:
        return image
    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    overlay_rgb = overlay_rgba[:, :, :3].astype(np.float32)
    if image.shape[2] == 4:
        base_rgb = image[:, :, :3].astype(np.float32)
        out_rgb = overlay_rgb * alpha + base_rgb * (1.0 - alpha)
        out = image.copy()
        out[:, :, :3] = out_rgb.astype(np.uint8)
        return out
    overlay_bgr = overlay_rgb[:, :, ::-1]
    base = image.astype(np.float32)
    blended = overlay_bgr * alpha + base * (1.0 - alpha)
    return blended.astype(np.uint8)
