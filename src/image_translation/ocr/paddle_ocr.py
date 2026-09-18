"""PaddleOCR-backed OCR engine – lazy-loaded on first use."""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ..config import OcrConfig
from ..models.text_region import TextRegion
from .base import OcrEngine

logger = logging.getLogger(__name__)

_LANG_MAP = {
    "zh": "ch",
    "zh-cn": "ch",
    "ch": "ch",
    "en": "en",
}


class PaddleOcrEngine(OcrEngine):
    """PaddleOCR implementation of the OcrEngine interface."""

    def __init__(self, config: OcrConfig) -> None:
        self._config = config
        self._lang = _LANG_MAP.get(config.source_language.lower(), "ch")
        self._ocr: Optional[object] = None

    @property
    def name(self) -> str:
        return "paddleocr"

    def detect(self, image: np.ndarray, *, min_confidence: Optional[float] = None) -> List[TextRegion]:
        ocr = self._get_ocr()
        results = self._run_ocr(ocr, image)

        regions: List[TextRegion] = []
        if not results or not results[0]:
            return regions

        threshold = self._config.min_confidence if min_confidence is None else min_confidence

        for idx, line in enumerate(results[0]):
            polygon_points = line[0]
            text_info = line[1]
            text = text_info[0]
            confidence = float(text_info[1])

            if confidence < threshold:
                continue

            polygon = [[float(p[0]), float(p[1])] for p in polygon_points]
            orientation = self._polygon_angle(polygon)
            crop_bbox = self._crop_bbox(polygon, image.shape)

            regions.append(
                TextRegion(
                    id=f"text_{idx + 1:03d}",
                    source_text=text,
                    confidence=confidence,
                    polygon=polygon,
                    language=self._config.source_language,
                    orientation=orientation,
                    source_crop_bbox=crop_bbox,
                )
            )

        return regions

    def _get_ocr(self) -> object:
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ImportError(
                    "PaddleOCR is not installed. Run script/Initialize-Env.ps1."
                ) from exc
            self._ocr = PaddleOCR(
                lang=self._lang,
                use_angle_cls=self._config.detect_rotation,
            )
            logger.info(
                "PaddleOCR initialized (lang=%s, angle_cls=%s)",
                self._lang,
                self._config.detect_rotation,
            )
        return self._ocr

    def _run_ocr(self, ocr: object, image: np.ndarray):
        try:
            return ocr.ocr(image, cls=self._config.detect_rotation)
        except TypeError:
            # Older PaddleOCR versions use a different signature.
            try:
                return ocr.ocr(image)
            except Exception as exc:
                raise RuntimeError(
                    "PaddleOCR call failed. Verify paddleocr compatibility "
                    f"with the installed version: {exc}"
                ) from exc

    @staticmethod
    def _polygon_angle(polygon: list) -> float:
        poly = np.array(polygon, dtype=np.float32)
        try:
            import cv2
            rect = cv2.minAreaRect(poly)
            return float(rect[2])
        except Exception:
            return 0.0

    @staticmethod
    def _crop_bbox(polygon: list, shape: tuple) -> List[int]:
        poly = np.array(polygon, dtype=np.int32)
        h, w = shape[:2]
        xs, ys = poly[:, 0], poly[:, 1]
        return [
            int(max(0, np.min(xs))),
            int(max(0, np.min(ys))),
            int(min(w, np.max(xs))),
            int(min(h, np.max(ys))),
        ]
