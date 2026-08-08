"""PaddleOCR-backed OCR engine – lazy-loaded on first use."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ..config import OcrConfig
from ..models.text_region import TextRegion
from .base import OcrEngine

logger = logging.getLogger(__name__)

# Module-level cache for lazy initialization
_paddle_ocr_instance: Optional[object] = None


def _get_paddle_ocr(lang: str = "ch") -> object:
    """Lazy-load and cache the PaddleOCR instance."""
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr_instance = PaddleOCR(lang=lang, use_angle_cls=True)
            logger.info("PaddleOCR initialized (lang=%s)", lang)
        except ImportError:
            raise ImportError(
                "PaddleOCR is not installed. Install with: pip install paddleocr"
            )
    return _paddle_ocr_instance


class PaddleOcrEngine(OcrEngine):
    """PaddleOCR implementation of the OcrEngine interface.

    Heavy model initialization is deferred to first detect() call.
    """

    def __init__(self, config: OcrConfig) -> None:
        self._config = config
        self._lang = "ch"  # PaddleOCR uses "ch" for Chinese

    @property
    def name(self) -> str:
        return "paddleocr"

    def detect(self, image: np.ndarray) -> List[TextRegion]:
        """Run PaddleOCR detection + recognition on an image.

        Args:
            image: BGR image as numpy array.

        Returns:
            List of TextRegion objects.
        """
        ocr = _get_paddle_ocr(lang=self._lang)
        results = ocr.ocr(image, cls=True)

        regions: List[TextRegion] = []
        if not results or not results[0]:
            return regions

        for idx, line in enumerate(results[0]):
            polygon_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text_info = line[1]       # (text, confidence)
            text = text_info[0]
            confidence = text_info[1]

            # Filter low-confidence detections
            if confidence < self._config.min_confidence:
                continue

            region = TextRegion(
                id=f"text_{idx + 1:03d}",
                source_text=text,
                confidence=float(confidence),
                polygon=[[float(p[0]), float(p[1])] for p in polygon_points],
                language=self._config.source_language,
            )
            regions.append(region)

        return regions
