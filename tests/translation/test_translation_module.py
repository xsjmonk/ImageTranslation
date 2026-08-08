"""Tests for shared translation module — no GPU/model loading required."""

from __future__ import annotations

import pytest

from image_translation.translation.config import TranslationConfig, GenerationConfig
from image_translation.translation.models import TranslationRequest, TranslationResult, TranslationRuntimeInfo
from image_translation.translation.text_utils import preprocess
from image_translation.translation.factory import create_translator


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_normal_text(self):
        assert preprocess("你好") == "你好"

    def test_whitespace_trim(self):
        assert preprocess("  你好  ") == "你好"

    def test_reject_none(self):
        with pytest.raises(ValueError, match="None"):
            preprocess(None)

    def test_reject_empty(self):
        with pytest.raises(ValueError, match="empty"):
            preprocess("")

    def test_reject_whitespace_only(self):
        with pytest.raises(ValueError, match="empty"):
            preprocess("   \n  ")

    def test_reject_oversized(self):
        with pytest.raises(ValueError, match="maximum length"):
            preprocess("x" * 5000, max_characters=4000)

    def test_unicode_normalization(self):
        # Non-breaking space → normal space
        result = preprocess("hello\u00a0world")
        assert "\u00a0" not in result

    def test_line_break_normalization(self):
        result = preprocess("line1\r\nline2\rline3")
        assert "\r" not in result
        assert "\r\n" not in result


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestTranslationModels:
    def test_translation_request(self):
        req = TranslationRequest(text="你好")
        assert req.text == "你好"
        assert req.source_language == "zh"
        assert req.target_language == "en"

    def test_translation_result(self):
        r = TranslationResult(
            source_text="你好",
            translated_text="Hello",
            model_name="m2m100_418M",
            device="cuda:0",
        )
        assert r.translated_text == "Hello"
        assert r.device == "cuda:0"

    def test_runtime_info(self):
        info = TranslationRuntimeInfo(
            model_name="m2m100_418M",
            device="cuda:0",
            ready=True,
            cuda_available=True,
            gpu_name="NVIDIA Test",
        )
        assert info.ready
        assert info.cuda_available


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestTranslationConfig:
    def test_defaults(self):
        cfg = TranslationConfig()
        assert cfg.model_name == "facebook/m2m100_418M"
        assert cfg.source_language == "zh"
        assert cfg.target_language == "en"
        assert cfg.device == "cuda"
        assert cfg.allow_cpu_fallback is False

    def test_effective_device(self):
        cfg = TranslationConfig(device="cuda", cuda_device=0)
        assert cfg.effective_device() == "cuda:0"

    def test_effective_device_cpu(self):
        cfg = TranslationConfig(device="cpu")
        assert cfg.effective_device() == "cpu"

    def test_invalid_precision(self):
        with pytest.raises(ValueError, match="precision"):
            TranslationConfig(precision="int8")

    def test_invalid_device(self):
        with pytest.raises(ValueError, match="device"):
            TranslationConfig(device="tpu")

    def test_generation_config_defaults(self):
        gen = GenerationConfig()
        assert gen.max_new_tokens == 256
        assert gen.num_beams == 1

    def test_batch_size_positive(self):
        with pytest.raises(ValueError, match="batch_size"):
            TranslationConfig(batch_size=0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_creates_m2m100(self):
        cfg = TranslationConfig(model_name="facebook/m2m100_418M")
        translator = create_translator(cfg)
        assert translator.name == "m2m100@facebook/m2m100_418M"

    def test_rejects_unknown(self):
        cfg = TranslationConfig(model_name="unknown/model")
        with pytest.raises(ValueError, match="Unknown translation engine"):
            create_translator(cfg)
