"""Text renderer – draws translated text onto a transparent layer using Pillow."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from .text_style import TextStyle

logger = logging.getLogger(__name__)


class TextRenderer:
    """Renders translated text onto a transparent RGBA layer."""

    def __init__(
        self,
        default_font_size: int = 18,
        font_path: Optional[str] = None,
        bold_font_path: Optional[str] = None,
        line_spacing: float = 1.2,
    ) -> None:
        self.default_font_size = default_font_size
        self.font_path = font_path
        self.bold_font_path = bold_font_path
        self.line_spacing = line_spacing

    def render_text_layer(
        self,
        image_width: int,
        image_height: int,
        layout: dict,
        style: Optional[TextStyle] = None,
    ) -> np.ndarray:
        from PIL import Image, ImageDraw, ImageFilter

        style = style or TextStyle()
        font_size = layout.get("font_size", self.default_font_size)
        bold = layout.get("bold", False)
        font = self._load_font(font_size, bold)
        lines = layout.get("lines", [])
        metrics = layout.get("line_metrics", [])
        center = layout.get("center", (image_width / 2, image_height / 2))
        angle = float(layout.get("angle", 0.0))
        alignment = layout.get("alignment", "center")
        region_width = layout.get("width", image_width)

        if not metrics:
            metrics = [{"text": line, "width": 0, "height": font_size} for line in lines]

        line_gap = font_size * (style.line_spacing or self.line_spacing)
        total_height = sum(m["height"] for m in metrics) + max(
            0, len(metrics) - 1
        ) * (line_gap - font_size)
        start_y = center[1] - total_height / 2

        shadow_layer = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
        text_layer = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        text_draw = ImageDraw.Draw(text_layer)

        alpha = int(max(0, min(255, round(style.opacity * 255))))
        stroke_fill = (*style.stroke_color_rgb, alpha) if style.stroke_width > 0 else None

        y = start_y
        for metric in metrics:
            line = metric["text"]
            bbox = text_draw.textbbox((0, 0), line, font=font, stroke_width=style.stroke_width)
            text_width = bbox[2] - bbox[0]
            if alignment == "left":
                text_x = center[0] - region_width / 2
            elif alignment == "right":
                text_x = center[0] + region_width / 2 - text_width
            else:
                text_x = center[0] - text_width / 2

            if style.shadow_opacity > 0 and style.shadow_offset != (0, 0):
                shadow_alpha = int(style.shadow_opacity * 255)
                shadow_fill = (*style.shadow_color_rgb, shadow_alpha)
                shadow_draw.text(
                    (text_x + style.shadow_offset[0], y + style.shadow_offset[1]),
                    line,
                    font=font,
                    fill=shadow_fill,
                )

            self._draw_line_with_gradient(
                text_draw,
                line,
                (text_x, y),
                font,
                style,
                alpha,
                stroke_fill,
            )
            y += metric["height"] + (line_gap - font_size)

        if style.shadow_blur > 0:
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=style.shadow_blur))

        layer = Image.alpha_composite(shadow_layer, text_layer)

        if style.italic_oblique:
            layer = layer.transform(
                layer.size,
                Image.AFFINE,
                (1, style.italic_oblique, 0, 0, 1, 0),
                resample=Image.BICUBIC,
            )

        layer_np = np.array(layer)
        if abs(angle) > 0.5:
            layer_np = self._rotate_layer(layer_np, angle, center)
        return layer_np

    def _draw_line_with_gradient(
        self,
        draw,
        line: str,
        position: Tuple[float, float],
        font,
        style: TextStyle,
        alpha: int,
        stroke_fill: Optional[Tuple[int, int, int, int]],
    ) -> None:
        if not (style.gradient_top_rgb and style.gradient_bottom_rgb):
            fill = (*style.color_rgb, alpha)
            draw.text(
                position,
                line,
                font=font,
                fill=fill,
                stroke_width=style.stroke_width,
                stroke_fill=stroke_fill,
            )
            return

        from PIL import Image, ImageDraw

        bbox = draw.textbbox(position, line, font=font, stroke_width=style.stroke_width)
        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        mask = Image.new("L", (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.text(
            (0, 0),
            line,
            font=font,
            fill=255,
            stroke_width=style.stroke_width,
        )
        gradient = Image.new("RGBA", (width, height))
        top = np.array(style.gradient_top_rgb, dtype=np.float32)
        bottom = np.array(style.gradient_bottom_rgb, dtype=np.float32)
        for row in range(height):
            t = row / max(height - 1, 1)
            color = (top * (1.0 - t) + bottom * t).astype(np.uint8)
            for col in range(width):
                gradient.putpixel((col, row), (*tuple(color), alpha))
        glyph = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        glyph.paste(gradient, (0, 0), mask)
        draw._image.paste(glyph, (int(x1), int(y1)), glyph)

    def _load_font(self, size: int, bold: bool):
        from PIL import ImageFont
        path = self.bold_font_path if bold and self.bold_font_path else self.font_path
        if path:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
        try:
            return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
        except OSError:
            return ImageFont.load_default()

    @staticmethod
    def _rotate_layer(layer: np.ndarray, angle: float, center: tuple) -> np.ndarray:
        from PIL import Image
        pil_layer = Image.fromarray(layer)
        rotated = pil_layer.rotate(
            -angle,
            resample=Image.BICUBIC,
            center=center,
            expand=False,
        )
        return np.array(rotated)
