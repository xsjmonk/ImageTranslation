"""Layout engine – computes font size, wrapping, and placement for translated text."""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LayoutEngine:
    """Computes text placement within a polygon region using font metrics."""

    def __init__(
        self,
        minimum_font_size: int = 12,
        maximum_font_size: int = 48,
        allow_multiline: bool = True,
        line_spacing: float = 1.2,
        clip_tolerance_pixels: int = 2,
        font_path: Optional[str] = None,
        bold_font_path: Optional[str] = None,
    ) -> None:
        self.minimum_font_size = minimum_font_size
        self.maximum_font_size = maximum_font_size
        self.allow_multiline = allow_multiline
        self.line_spacing = line_spacing
        self.clip_tolerance_pixels = clip_tolerance_pixels
        self.font_path = font_path
        self.bold_font_path = bold_font_path

    def compute_layout(
        self,
        polygon,
        text: str,
        region_id: str = "",
        bold: bool = False,
    ) -> dict:
        """Compute layout parameters for rendering text into a polygon."""
        poly = np.array(polygon, dtype=np.float32)
        center, width, height, angle = self._normalize_oriented_rect(poly)

        font_size, lines, metrics = self._fit_text(
            text=text,
            max_width=width,
            max_height=height,
            bold=bold,
        )

        clipped = self._detect_clipping(metrics, width, height)
        fit_warning = None
        if font_size <= self.minimum_font_size and clipped:
            fit_warning = "Text may not fit legibly at minimum font size"

        return {
            "region_id": region_id,
            "center": center,
            "width": width,
            "height": height,
            "angle": angle,
            "font_size": font_size,
            "lines": lines,
            "line_metrics": metrics,
            "bold": bold,
            "clipped": clipped,
            "fit_warning": fit_warning,
            "alignment": self._infer_alignment(poly, center),
        }

    @staticmethod
    def _infer_alignment(poly: np.ndarray, center: Tuple[float, float]) -> str:
        xs = poly[:, 0]
        x1, x2 = float(xs.min()), float(xs.max())
        cx = float(np.mean(xs))
        third = (x2 - x1) / 3.0
        if cx < x1 + third:
            return "left"
        if cx > x2 - third:
            return "right"
        return "center"

    def _normalize_oriented_rect(
        self, polygon: np.ndarray
    ) -> Tuple[Tuple[float, float], float, float, float]:
        try:
            import cv2
            rect = cv2.minAreaRect(polygon)
            (cx, cy), (w, h), angle = rect
            if w < h:
                w, h = h, w
                angle += 90.0
            return (float(cx), float(cy)), float(w), float(h), float(angle)
        except ImportError:
            xs = polygon[:, 0]
            ys = polygon[:, 1]
            return (
                float(np.mean(xs)),
                float(np.mean(ys)),
            ), float(np.max(xs) - np.min(xs)), float(np.max(ys) - np.min(ys)), 0.0

    def _fit_text(
        self,
        text: str,
        max_width: float,
        max_height: float,
        bold: bool,
    ) -> Tuple[int, List[str], List[dict]]:
        from PIL import ImageFont

        if max_width <= 0 or max_height <= 0 or not text:
            return self.minimum_font_size, [text], []

        for size in range(self.maximum_font_size, self.minimum_font_size - 1, -1):
            font = self._load_font(size, bold)
            lines = self._wrap_with_metrics(text, font, max_width)
            metrics = [self._line_metric(line, font) for line in lines]
            total_height = sum(m["height"] for m in metrics) + (
                (len(lines) - 1) * size * (self.line_spacing - 1)
            )
            max_line_width = max(m["width"] for m in metrics) if metrics else 0
            if max_line_width <= max_width + self.clip_tolerance_pixels and total_height <= max_height + self.clip_tolerance_pixels:
                return size, lines, metrics

        font = self._load_font(self.minimum_font_size, bold)
        lines = self._wrap_with_metrics(text, font, max_width)
        metrics = [self._line_metric(line, font) for line in lines]
        return self.minimum_font_size, lines, metrics

    def _wrap_with_metrics(self, text: str, font, max_width: float) -> List[str]:
        if not self.allow_multiline:
            return [text]

        words = self._tokenize(text)
        lines: List[str] = []
        current = ""
        for word in words:
            candidate = (current + " " + word).strip() if current else word
            if self._line_metric(candidate, font)["width"] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                if self._line_metric(word, font)["width"] > max_width:
                    lines.extend(self._break_long_token(word, font, max_width))
                    current = ""
                else:
                    current = word
        if current:
            lines.append(current)
        return lines or [text]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if " " in text:
            return text.split()
        return [text]

    def _break_long_token(self, token: str, font, max_width: float) -> List[str]:
        parts: List[str] = []
        current = ""
        for ch in token:
            candidate = current + ch
            if self._line_metric(candidate, font)["width"] <= max_width:
                current = candidate
            else:
                if current:
                    parts.append(current)
                current = ch
        if current:
            parts.append(current)
        return parts

    @staticmethod
    def _line_metric(text: str, font) -> dict:
        bbox = font.getbbox(text)
        return {
            "text": text,
            "width": float(bbox[2] - bbox[0]),
            "height": float(bbox[3] - bbox[1]),
        }

    def _detect_clipping(self, metrics: List[dict], max_width: float, max_height: float) -> bool:
        if not metrics:
            return False
        total_h = sum(m["height"] for m in metrics)
        max_w = max(m["width"] for m in metrics)
        return (
            max_w > max_width + self.clip_tolerance_pixels
            or total_h > max_height + self.clip_tolerance_pixels
        )

    def _load_font(self, size: int, bold: bool):
        from PIL import ImageFont
        path = self.bold_font_path if bold and self.bold_font_path else self.font_path
        if path:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
        try:
            name = "arialbd.ttf" if bold else "arial.ttf"
            return ImageFont.truetype(name, size)
        except OSError:
            return ImageFont.load_default()
