"""Text renderer – draws translated text onto a transparent layer using Pillow."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TextRenderer:
    """Renders translated text onto a transparent RGBA layer.

    Uses Pillow for font rendering, composited back into the image later.
    """

    def __init__(self, default_font_size: int = 18) -> None:
        self.default_font_size = default_font_size

    def render_text_layer(
        self,
        image_width: int,
        image_height: int,
        layout: dict,
        text_color: tuple = (255, 255, 255),
    ) -> np.ndarray:
        """Render text on a transparent RGBA layer matching image dimensions.

        Args:
            image_width: Full image width.
            image_height: Full image height.
            layout: Layout dict from LayoutEngine.
            text_color: RGB color tuple.

        Returns:
            RGBA numpy array (height, width, 4).
        """
        from PIL import Image, ImageDraw, ImageFont

        layer = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        font_size = layout.get("font_size", self.default_font_size)
        font = self._load_font(font_size)
        lines = layout.get("lines", [])
        center = layout.get("center", (image_width / 2, image_height / 2))
        angle = layout.get("angle", 0.0)

        # Calculate total text block height
        line_height = font_size + 4
        total_height = len(lines) * line_height
        start_y = center[1] - total_height / 2

        for i, line in enumerate(lines):
            # Measure text for centering
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = center[0] - text_width / 2
            text_y = start_y + i * line_height

            draw.text((text_x, text_y), line, font=font, fill=(*text_color, 255))

        layer_np = np.array(layer)

        # Apply rotation if needed
        if abs(angle) > 0.5:
            layer_np = self._rotate_layer(layer_np, angle, center)

        return layer_np

    @staticmethod
    def _load_font(size: int) -> "ImageFont.FreeTypeFont":
        """Load a font, falling back to default."""
        from PIL import ImageFont
        try:
            # Try a common sans-serif font
            return ImageFont.truetype("arial.ttf", size)
        except (OSError, IOError):
            try:
                return ImageFont.truetype("C:\\Windows\\Fonts\\msyh.ttc", size)
            except (OSError, IOError):
                return ImageFont.load_default()

    @staticmethod
    def _rotate_layer(
        layer: np.ndarray, angle: float, center: tuple
    ) -> np.ndarray:
        """Rotate a transparent RGBA layer around a center point."""
        from PIL import Image
        pil_layer = Image.fromarray(layer)
        rotated = pil_layer.rotate(
            -angle,  # Pillow rotates counterclockwise; OCR angle is clockwise
            resample=Image.BICUBIC,
            center=center,
            expand=False,
        )
        return np.array(rotated)
