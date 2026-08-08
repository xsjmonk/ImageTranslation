"""Layout engine – computes font size, wrapping, and placement for translated text."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LayoutEngine:
    """Computes text placement within a polygon region.

    Derives center, dimensions, and rotation from the OCR polygon,
    then fits the translated text within the region.
    """

    def __init__(self, minimum_font_size: int = 12, allow_multiline: bool = True) -> None:
        self.minimum_font_size = minimum_font_size
        self.allow_multiline = allow_multiline

    def compute_layout(
        self, polygon, text: str, max_font_size: int = 48
    ) -> dict:
        """Compute layout parameters for rendering text into a polygon.

        Args:
            polygon: [[x1,y1], [x2,y2], ...] from OCR.
            text: Translated text to fit.
            max_font_size: Upper bound for font size.

        Returns:
            Dict with center, width, height, angle, font_size, lines.
        """
        poly = np.array(polygon, dtype=np.float32)

        try:
            import cv2
            rect = cv2.minAreaRect(poly)
            center = (float(rect[0][0]), float(rect[0][1]))
            size = (float(rect[1][0]), float(rect[1][1]))
            angle = float(rect[2])
        except ImportError:
            center, size, angle = self._fallback_rect(poly)

        # Fit font size proportionally
        font_size = self._fit_font_size(text, size, max_font_size)

        lines = [text]
        if self.allow_multiline and len(text) > 10:
            lines = self._wrap_text(text, int(size[0] / (font_size * 0.5)))

        return {
            "center": center,
            "width": size[0],
            "height": size[1],
            "angle": angle,
            "font_size": max(font_size, self.minimum_font_size),
            "lines": lines,
        }

    def _fallback_rect(self, polygon: np.ndarray) -> Tuple[tuple, tuple, float]:
        """Compute bounding rect from polygon without OpenCV."""
        xs = polygon[:, 0]
        ys = polygon[:, 1]
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        w = float(np.max(xs) - np.min(xs))
        h = float(np.max(ys) - np.min(ys))
        return (cx, cy), (w, h), 0.0

    def _fit_font_size(
        self, text: str, size: Tuple[float, float], max_font: int
    ) -> int:
        """Estimate a reasonable font size given text and region dimensions."""
        char_count = len(text)
        region_width = size[0]
        if region_width <= 0 or char_count <= 0:
            return max_font

        # Rough: each char is ~0.6 * font_size wide
        estimated = int(region_width / (char_count * 0.6))
        return max(self.minimum_font_size, min(estimated, max_font))

    @staticmethod
    def _wrap_text(text: str, max_chars_per_line: int) -> list:
        """Simple word-aware line wrapping."""
        if max_chars_per_line <= 0:
            return [text]

        words = text.split()
        lines = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= max_chars_per_line:
                current = (current + " " + word).strip()
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [text]
