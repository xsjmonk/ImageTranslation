"""Tests for text action classifier."""

from __future__ import annotations

import pytest

from image_translation.config.models import ClassificationConfig, TranslationConfig
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
        result = classify_regions(regions, config, ClassificationConfig())
        assert result[0].action == TextAction.translate

    def test_preserve_english(self):
        config = TranslationConfig(preserve_already_target_language=True)
        regions = [_make_region("UL Certified Premium Quality")]
        result = classify_regions(regions, config, ClassificationConfig())
        assert result[0].action == TextAction.preserve
        assert "already_target_language" in result[0].action_reason

    def test_preserve_term_match(self):
        config = TranslationConfig(
            preserve_terms=["HUAWEI", "USB-C"],
            preserve_already_target_language=False,
        )
        regions = [_make_region("HUAWEI Mate 60 Pro")]
        result = classify_regions(regions, config, ClassificationConfig())
        assert result[0].action == TextAction.preserve
        assert "preserve_term" in result[0].action_reason

    def test_preserve_pattern_match(self):
        config = TranslationConfig(
            preserve_patterns=["^MODEL-\\d+$"],
            preserve_already_target_language=False,
        )
        regions = [_make_region("MODEL-1234")]
        result = classify_regions(regions, config, ClassificationConfig())
        assert result[0].action == TextAction.preserve
        assert "preserve_pattern" in result[0].action_reason

    def test_empty_text_removed(self):
        config = TranslationConfig()
        regions = [_make_region("  ")]
        result = classify_regions(regions, config, ClassificationConfig())
        assert result[0].action == TextAction.remove

    def test_short_ascii_logo_review(self):
        class_cfg = ClassificationConfig(
            logo_max_chars=4,
            logo_aspect_ratio_min=2.0,
            review_patterns=["^LOGO$"],
        )
        trans_cfg = TranslationConfig(preserve_already_target_language=False)
        region = _make_region("LOGO", id="t1")
        region.polygon = [[0, 0], [80, 0], [80, 10], [0, 10]]
        result = classify_regions([region], trans_cfg, class_cfg)
        assert result[0].action == TextAction.review

    def test_default_review_on_uncertain(self):
        config = TranslationConfig(default_action="review")
        regions = [_make_region("Some混合text")]
        result = classify_regions(regions, config, ClassificationConfig())
        assert result[0].action == TextAction.review

    def test_chinese_brand_mark_review(self):
        config = TranslationConfig(preserve_already_target_language=False)
        regions = [_make_region("华为")]
        result = classify_regions(regions, config, ClassificationConfig())
        assert result[0].action == TextAction.review
        assert "brand" in result[0].action_reason or "logo" in result[0].action_reason

    def test_chinese_descriptive_translates(self):
        config = TranslationConfig(default_action="translate")
        regions = [_make_region("加厚升级防锈")]
        result = classify_regions(regions, config, ClassificationConfig())
        assert result[0].action == TextAction.translate
        assert result[0].action_reason == "descriptive_callout"

    def test_force_translate_overrides_brand_review(self):
        class_cfg = ClassificationConfig(force_translate_patterns=[r"^华为$"])
        config = TranslationConfig(preserve_already_target_language=False)
        regions = [_make_region("华为")]
        result = classify_regions(regions, config, class_cfg)
        assert result[0].action == TextAction.translate
        assert result[0].action_reason == "force_translate_pattern"

    def test_model_serial_preserved(self):
        config = TranslationConfig(preserve_already_target_language=False)
        regions = [_make_region("RB2140")]
        result = classify_regions(regions, config, ClassificationConfig())
        assert result[0].action == TextAction.preserve

    def test_long_descriptive_label_translates(self):
        config = TranslationConfig(default_action="translate")
        regions = [_make_region("加厚升级防锈耐用说明")]
        result = classify_regions(regions, config, ClassificationConfig())
        assert result[0].action == TextAction.translate
