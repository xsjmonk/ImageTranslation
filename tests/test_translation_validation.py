"""Translation validation tests."""

from image_translation.config.models import TranslationConfig
from image_translation.models.text_region import TextAction, TextRegion
from image_translation.translation.validation import validate_translations


def _region(text: str, translation: dict) -> TextRegion:
    return TextRegion(
        id="r1",
        source_text=text,
        confidence=0.9,
        polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
        action=TextAction.translate,
        translation=translation,
    )


def test_protected_term_preserved():
    config = TranslationConfig(preserve_terms=["Ray-Ban"])
    regions = [
        _region("Ray-Ban 眼镜", {"translated_text": "Eyeglasses"})
    ]
    result = validate_translations(regions, config)
    assert not result.passed
    assert any(i.code == "protected_term_lost" for i in result.issues)


def test_cardinality_not_applicable_here():
    config = TranslationConfig()
    regions = [_region("枪色", {"translated_text": "GUNMETAL"})]
    result = validate_translations(regions, config)
    assert result.passed
