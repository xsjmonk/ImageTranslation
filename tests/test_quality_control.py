"""Tests for image localization quality control."""

from image_translation.models.text_region import TextAction, TextRegion
from image_translation.quality_control import QcSeverity, validate_image_regions


def _region(
    text: str,
    action: TextAction = TextAction.translate,
    translation: dict | None = None,
    confidence: float = 0.9,
) -> TextRegion:
    return TextRegion(
        id="r1",
        source_text=text,
        confidence=confidence,
        polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
        action=action,
        translation=translation or {},
    )


def test_validate_passes_for_good_translation():
    region = _region("枪色", translation={"translated_text": "GUNMETAL"})
    result = validate_image_regions([region])
    assert result.passed
    assert not result.has_errors


def test_validate_fails_for_noop_translation():
    region = _region("枪色", translation={"translated_text": "[枪色]"})
    result = validate_image_regions([region])
    assert not result.passed
    assert result.has_errors


def test_validate_warns_for_review_regions():
    region = _region("品牌LOGO", action=TextAction.review)
    result = validate_image_regions([region])
    assert result.passed
    assert result.has_warnings
    assert result.issues[0].severity == QcSeverity.warning
