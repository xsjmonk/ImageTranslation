"""Tests for text action classifier."""

from __future__ import annotations

import pytest

from image_translation.config import TranslationConfig
from image_translation.models.text_region import TextAction, TextRegion
from image_translation.translation.classifier import classify_regions


def _make_region(text: str, id: str = "t1") -> TextRegion:
    return TextRegion(
        id=id,
        source_text=text,
        confidence=0.9,
        polygon=[[0, 0], [100, 0], [100, 30], [0, 30]],
    )


class TestClassifier:
    def test_translate_chinese(self):
        config = TranslationConfig(default_action="translate")
        regions = [_make_region("加厚升级")]
        result = classify_regions(regions, config)
        assert result[0].action == TextAction.translate

    def test_preserve_english(self):
        config = TranslationConfig(preserve_already_target_language=True)
        regions = [_make_region("UL Certified Premium Quality")]
        result = classify_regions(regions, config)
        assert result[0].action == TextAction.preserve
        assert "already_target_language" in result[0].action_reason

    def test_preserve_term_match(self):
        config = TranslationConfig(
            preserve_terms=["HUAWEI", "USB-C"],
            preserve_already_target_language=False,
        )
        regions = [_make_region("HUAWEI Mate 60 Pro")]
        result = classify_regions(regions, config)
        assert result[0].action == TextAction.preserve
        assert "preserve_term" in result[0].action_reason

    def test_preserve_pattern_match(self):
        config = TranslationConfig(preserve_patterns=["^[A-Z0-9\\-]+$"])
        regions = [_make_region("USB-C")]
        result = classify_regions(regions, config)
        assert result[0].action == TextAction.preserve
        assert "preserve_pattern" in result[0].action_reason

    def test_empty_text_removed(self):
        config = TranslationConfig()
        regions = [_make_region("  ")]
        result = classify_regions(regions, config)
        assert result[0].action == TextAction.remove

    def test_default_review_on_uncertain(self):
        config = TranslationConfig(default_action="review")
        regions = [_make_region("Some混合text")]
        result = classify_regions(regions, config)
        assert result[0].action == TextAction.review
