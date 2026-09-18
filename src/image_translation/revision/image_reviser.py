"""Image reviser – composites translated text onto cleaned images."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from ..config import RevisionConfig
from ..models.text_region import TextAction, TextRegion
from .layout import LayoutEngine
from .render_validation import measure_alpha_bounds
from .text_renderer import TextRenderer
from .text_style import TextStyle, estimate_style_from_crop

logger = logging.getLogger(__name__)


class ImageReviser:
    """Orchestrates layout + text rendering + compositing."""

    def __init__(self, config: RevisionConfig) -> None:
        self._config = config
        self._layout_engine = LayoutEngine(
            minimum_font_size=config.minimum_font_size,
            maximum_font_size=config.maximum_font_size,
            allow_multiline=config.allow_multiline,
            line_spacing=config.line_spacing,
            clip_tolerance_pixels=config.clip_tolerance_pixels,
            font_path=config.font_path,
            bold_font_path=config.bold_font_path,
        )
        self._text_renderer = TextRenderer(
            font_path=config.font_path,
            bold_font_path=config.bold_font_path,
            line_spacing=config.line_spacing,
        )

    def revise(
        self,
        cleaned_image: np.ndarray,
        source_image: np.ndarray,
        regions: List[TextRegion],
    ) -> Tuple[np.ndarray, List[dict]]:
        """Composite translated text onto a cleaned image."""
        h, w = cleaned_image.shape[:2]
        result = cleaned_image.copy()
        layouts: List[dict] = []

        for region in regions:
            if region.action != TextAction.translate:
                continue
            if not region.translation:
                continue

            text = self._pick_best_text(region)
            if not text:
                continue

            style = self._resolve_style(region, source_image)
            polygon = region.polygon if self._config.use_source_polygon else region.polygon
            layout = self._layout_engine.compute_layout(
                polygon,
                text,
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
            if style.review_warnings:
                layout["style_warnings"] = list(style.review_warnings)
            layouts.append(layout)
            result = self._composite(result, text_layer)

        return result, layouts

    def _resolve_style(self, region: TextRegion, source_image: np.ndarray) -> TextStyle:
        if region.style:
            style = TextStyle(
                color_rgb=tuple(region.style.get("color_rgb", (255, 255, 255))),
                opacity=float(region.style.get("opacity", 1.0)),
                bold=bool(region.style.get("bold", False)),
                italic_oblique=float(region.style.get("italic_oblique", 0.0)),
                stroke_width=int(region.style.get("stroke_width", 0)),
                stroke_color_rgb=tuple(region.style.get("stroke_color_rgb", (0, 0, 0))),
                shadow_offset=tuple(region.style.get("shadow_offset", (0, 0))),
                shadow_blur=int(region.style.get("shadow_blur", 0)),
                shadow_color_rgb=tuple(region.style.get("shadow_color_rgb", (0, 0, 0))),
                shadow_opacity=float(region.style.get("shadow_opacity", 0.0)),
                recovered_effects=list(region.style.get("recovered_effects", [])),
                review_warnings=list(region.style.get("review_warnings", [])),
            )
            if region.style.get("gradient_top_rgb") and region.style.get("gradient_bottom_rgb"):
                style.gradient_top_rgb = tuple(region.style["gradient_top_rgb"])
                style.gradient_bottom_rgb = tuple(region.style["gradient_bottom_rgb"])
            return style

        style = estimate_style_from_crop(source_image, region.source_crop_bbox)
        region.style = style.to_dict()
        return style

    @staticmethod
    def _pick_best_text(region: TextRegion) -> Optional[str]:
        t = region.translation
        for key in ("compact_text", "translated_text", "literal_text"):
            value = (t.get(key) or "").strip()
            if value:
                return value
        return None

    @staticmethod
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
