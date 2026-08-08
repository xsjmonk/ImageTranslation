"""Image reviser – composites translated text onto cleaned images."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ..config import RevisionConfig
from ..models.text_region import TextAction, TextRegion
from .layout import LayoutEngine
from .text_renderer import TextRenderer

logger = logging.getLogger(__name__)


class ImageReviser:
    """Orchestrates layout + text rendering + compositing.

    Delegates to LayoutEngine and TextRenderer for specific tasks.
    """

    def __init__(self, config: RevisionConfig) -> None:
        self._config = config
        self._layout_engine = LayoutEngine(
            minimum_font_size=config.minimum_font_size,
            allow_multiline=config.allow_multiline,
        )
        self._text_renderer = TextRenderer()

    def revise(
        self,
        cleaned_image: np.ndarray,
        regions: List[TextRegion],
    ) -> np.ndarray:
        """Composite translated text onto a cleaned image.

        Only 'translate' regions are rendered.

        Args:
            cleaned_image: Image with source text removed (BGR).
            regions: Text regions with translations populated.

        Returns:
            Final image with translated text composited (BGR).
        """
        h, w = cleaned_image.shape[:2]
        result = cleaned_image.copy()

        for region in regions:
            if region.action != TextAction.translate:
                continue
            if not region.translation:
                continue

            text = self._pick_best_text(region)
            if not text:
                continue

            layout = self._layout_engine.compute_layout(
                region.polygon,
                text,
            )

            # Render text on a transparent layer
            text_layer = self._text_renderer.render_text_layer(
                w, h, layout
            )

            # Composite onto result
            result = self._composite(result, text_layer)

        return result

    @staticmethod
    def _pick_best_text(region: TextRegion) -> Optional[str]:
        """Choose the best translation variant for rendering."""
        t = region.translation
        # Prefer compact > translated > literal
        compact = t.get("compact_text", "").strip()
        if compact:
            return compact
        translated = t.get("translated_text", "").strip()
        if translated:
            return translated
        return t.get("literal_text", "").strip() or None

    @staticmethod
    def _composite(image_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        """Alpha-composite an RGBA overlay onto a BGR image."""
        if overlay_rgba.shape[2] != 4:
            return image_bgr

        alpha = overlay_rgba[:, :, 3:4] / 255.0
        overlay_bgr = overlay_rgba[:, :, :3][:, :, ::-1]  # RGB -> BGR

        result = (overlay_bgr * alpha + image_bgr * (1 - alpha)).astype(np.uint8)
        return result
