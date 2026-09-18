"""Final-image OCR quality control tests."""

from __future__ import annotations

from typing import List

import numpy as np

from image_translation.config.models import QcConfig
from image_translation.models.text_region import TextAction, TextRegion
from image_translation.ocr.base import OcrEngine
from image_translation.quality_control import validate_final_image


class FakeFinalOcr(OcrEngine):
    def __init__(self, detections: List[TextRegion]):
        self._detections = detections

    @property
    def name(self) -> str:
        return "fake-final-ocr"

    def detect(self, image: np.ndarray, *, min_confidence=None) -> List[TextRegion]:
        threshold = 0.0 if min_confidence is None else min_confidence
        return [d for d in self._detections if d.confidence >= threshold]


def _translate_region(text: str = "枪色") -> TextRegion:
    return TextRegion(
        id="r1",
        source_text=text,
        confidence=0.95,
        polygon=[[10, 10], [100, 10], [100, 40], [10, 40]],
        action=TextAction.translate,
        translation={"translated_text": "GUNMETAL"},
    )


def test_final_ocr_rejects_residual_chinese():
    source = final = np.zeros((80, 120, 3), dtype=np.uint8)
    residual = TextRegion(
        id="d1",
        source_text="残留",
        confidence=0.9,
        polygon=[[12, 12], [90, 12], [90, 38], [12, 38]],
    )
    ocr = FakeFinalOcr([residual])
    result = validate_final_image(
        source,
        final,
        [_translate_region()],
        [{"region_id": "r1", "render_bounds": {"x1": 20, "y1": 15, "x2": 80, "y2": 35, "alpha_coverage": 0.05}}],
        QcConfig(enable_final_ocr=True, final_ocr_cjk_coverage_threshold=0.05),
        final_ocr_engine=ocr,
    )
    assert not result.passed
    assert any(i.code == "residual_chinese_in_output" for i in result.issues)
    assert result.evidence["final_ocr"]["backend"] == "fake-final-ocr"


def test_final_ocr_honors_lower_threshold_than_source_default():
    source = final = np.zeros((80, 120, 3), dtype=np.uint8)
    low_conf = TextRegion(
        id="d1",
        source_text="残留",
        confidence=0.35,
        polygon=[[12, 12], [90, 12], [90, 38], [12, 38]],
    )
    ocr = FakeFinalOcr([low_conf])
    result = validate_final_image(
        source,
        final,
        [_translate_region()],
        [{"region_id": "r1", "render_bounds": {"x1": 20, "y1": 15, "x2": 80, "y2": 35, "alpha_coverage": 0.05}}],
        QcConfig(
            enable_final_ocr=True,
            final_ocr_min_confidence=0.3,
            final_ocr_cjk_coverage_threshold=0.05,
        ),
        final_ocr_engine=ocr,
    )
    assert not result.passed
    assert result.evidence["final_ocr"]["effective_min_confidence"] == 0.3


def test_final_ocr_ignores_preserved_region_detections():
    source = final = np.zeros((80, 120, 3), dtype=np.uint8)
    logo = TextRegion(
        id="logo",
        source_text="华为",
        confidence=0.9,
        polygon=[[0, 0], [30, 0], [30, 20], [0, 20]],
        action=TextAction.preserve,
    )
    residual = TextRegion(
        id="d1",
        source_text="华为",
        confidence=0.9,
        polygon=[[0, 0], [30, 0], [30, 20], [0, 20]],
    )
    ocr = FakeFinalOcr([residual])
    result = validate_final_image(
        source,
        final,
        [logo],
        [],
        QcConfig(enable_final_ocr=True),
        final_ocr_engine=ocr,
    )
    assert result.passed
