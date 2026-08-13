"""Tests for shared translation module — no GPU/model loading required."""

from __future__ import annotations

import pytest

from image_translation.translation.config import TranslationConfig, GenerationConfig
from image_translation.translation.exceptions import (
    TranslationDeviceError,
    TranslationInputError,
)
from image_translation.translation.models import (
    TranslationRequest,
    TranslationResult,
    TranslationRuntimeInfo,
)
from image_translation.translation.text_utils import preprocess
from image_translation.translation.factory import create_translator


# ---------------------------------------------------------------------------
# Text preprocessing (minimal, conservative)
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_normal_text(self):
        assert preprocess("你好") == "你好"

    def test_whitespace_trim(self):
        assert preprocess("  你好  ") == "你好"

    def test_reject_none(self):
        with pytest.raises(TranslationInputError, match="None"):
            preprocess(None)

    def test_reject_empty(self):
        with pytest.raises(TranslationInputError, match="empty"):
            preprocess("")

    def test_reject_whitespace_only(self):
        with pytest.raises(TranslationInputError, match="empty"):
            preprocess("   \n  ")

    def test_reject_oversized(self):
        with pytest.raises(TranslationInputError, match="maximum length"):
            preprocess("x" * 5000, max_characters=4000)

    def test_content_preserved(self):
        """NFKC is NOT applied — compatibility characters stay untouched."""
        assert preprocess("① ② ③") == "① ② ③"
        assert preprocess("hello\u00a0world") == "hello\u00a0world"

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

    def test_invalid_cuda_device_negative(self):
        with pytest.raises(ValueError, match="cuda_device"):
            TranslationConfig(cuda_device=-1)

    def test_generation_config_defaults(self):
        gen = GenerationConfig()
        assert gen.max_new_tokens == 256
        assert gen.num_beams == 4
        assert gen.length_penalty == 1.0

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


# ---------------------------------------------------------------------------
# Device resolution (CUDA required, CPU fallback logic)
# ---------------------------------------------------------------------------

class TestDeviceResolution:
    def _make_translator(self, **kwargs):
        from image_translation.translation.m2m100_translator import M2M100Translator
        return M2M100Translator(TranslationConfig(**kwargs))

    def test_cuda_required_raises_when_unavailable(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        t = self._make_translator(device="cuda", allow_cpu_fallback=False)
        with pytest.raises(TranslationDeviceError, match="CUDA is required"):
            t._resolve_device()

    def test_no_cpu_fallback_when_disabled(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        t = self._make_translator(device="cuda", allow_cpu_fallback=False)
        with pytest.raises(TranslationDeviceError):
            t._resolve_device()

    def test_cpu_fallback_uses_cpu(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        t = self._make_translator(device="cuda", allow_cpu_fallback=True)
        assert t._resolve_device() == "cpu"

    def test_requested_cpu_uses_cpu(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        t = self._make_translator(device="cpu")
        assert t._resolve_device() == "cpu"

    def test_cuda_device_index_validation(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        t = self._make_translator(device="cuda", cuda_device=5)
        with pytest.raises(TranslationDeviceError, match="Invalid CUDA device index"):
            t._resolve_device()

    def test_cuda_device_index_ok(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        t = self._make_translator(device="cuda", cuda_device=1)
        assert t._resolve_device() == "cuda:1"

    def test_float16_on_cpu_rejected(self):
        from image_translation.translation.exceptions import TranslationConfigurationError
        from image_translation.translation.m2m100_translator import M2M100Translator
        t = M2M100Translator(TranslationConfig(device="cpu", precision="float16"))
        with pytest.raises(TranslationConfigurationError, match="float16"):
            t._load_model()


# ---------------------------------------------------------------------------
# Batch translation: real chunked GPU batching, order preserved
# ---------------------------------------------------------------------------

class TestBatchTranslation:
    def test_batch_chunks_and_preserves_order(self, monkeypatch):
        from image_translation.translation.m2m100_translator import M2M100Translator

        t = M2M100Translator(TranslationConfig(batch_size=2))
        # Fake a loaded model so _translate_impl is reached
        t._model = object()
        t._tokenizer = object()
        t._device_str = "cpu"

        calls: list[list[str]] = []

        def fake_impl(texts, source_lang, target_lang, max_new_tokens=None):
            calls.append(list(texts))
            return [
                TranslationResult(
                    source_text=x,
                    translated_text=f"EN:{x}",
                    source_language=source_lang,
                    target_language=target_lang,
                    model_name="m2m100",
                    device="cpu",
                )
                for x in texts
            ]

        monkeypatch.setattr(t, "_translate_impl", fake_impl)

        texts = ["a", "b", "c", "d", "e"]
        results = t.translate_batch_texts(texts)

        # Chunked: [a,b], [c,d], [e]
        assert calls == [["a", "b"], ["c", "d"], ["e"]]
        # Order preserved
        assert [r.source_text for r in results] == texts
        assert [r.translated_text for r in results] == [
            "EN:a", "EN:b", "EN:c", "EN:d", "EN:e",
        ]

    def test_batch_empty(self):
        from image_translation.translation.m2m100_translator import M2M100Translator
        t = M2M100Translator(TranslationConfig())
        assert t.translate_batch_texts([]) == []


class TestActiveGenerationPath:
    """Prove the ACTIVE M2M100 generation path (not just config defaults):
    the exact kwargs passed to tokenizer() and model.generate()."""

    def _stub_translator(self):
        import torch
        from image_translation.translation.m2m100_translator import (
            M2M100Translator,
        )

        cfg = TranslationConfig()  # precision="auto", num_beams=4
        translator = M2M100Translator(cfg)
        captured = {}

        class FakeTokenizer:
            def __init__(self):
                self.src_lang = None

            def get_lang_id(self, lang):
                return {"zh": 128102, "en": 128022, "fr": 128014}.get(lang, -1)

            def __call__(self, texts, return_tensors=None, padding=None,
                         truncation=None):
                captured["tokenize_kwargs"] = {
                    "texts": list(texts),
                    "return_tensors": return_tensors,
                    "padding": padding,
                    "truncation": truncation,
                }
                ids = torch.tensor([[128102, 1, 2]])
                return {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                }

            def batch_decode(self, sequences, skip_special_tokens=None):
                captured["decode_kwargs"] = {
                    "skip_special_tokens": skip_special_tokens,
                }
                return ["Hello", "Bonjour"][: len(sequences)]

        class FakeParam:
            dtype = torch.float32

        class FakeModel:
            class Config:
                max_position_embeddings = 1024

            config = Config()

            def parameters(self):
                return iter([FakeParam()])

            def generate(self, **kwargs):
                captured["generate_kwargs"] = kwargs
                return torch.tensor([[2, 128022, 3]])

        translator._tokenizer = FakeTokenizer()
        translator._model = FakeModel()
        translator._device_str = "cpu"
        translator._precision_str = "float32"
        return translator, captured

    def test_tokenizer_receives_truncation_false(self):
        translator, captured = self._stub_translator()
        translator.translate_text("你好")
        assert captured["tokenize_kwargs"]["truncation"] is False
        assert captured["tokenize_kwargs"]["padding"] is True

    def test_generate_receives_forced_bos_and_beams(self):
        translator, captured = self._stub_translator()
        translator.translate_text("你好", source_lang="zh", target_lang="fr")
        kw = captured["generate_kwargs"]
        # forced_bos_token_id resolved from the REQUEST target language
        assert kw["forced_bos_token_id"] == 128014  # fr
        assert kw["num_beams"] == 4
        assert kw["max_new_tokens"] == 256  # GenerationConfig default
        assert "input_ids" in kw and "attention_mask" in kw

    def test_generate_has_no_no_repeat_ngram_size(self):
        translator, captured = self._stub_translator()
        translator.translate_text("你好")
        assert "no_repeat_ngram_size" not in captured["generate_kwargs"]

    def test_auto_precision_never_calls_half(self):
        from image_translation.translation.m2m100_translator import (
            M2M100Translator,
        )
        translator = M2M100Translator(TranslationConfig(precision="auto"))

        class Model:
            def half(self):
                raise AssertionError("model.half() called under precision=auto")

        assert translator._resolve_precision(Model(), "cpu") == "float32"

    def test_explicit_float16_calls_half(self):
        from image_translation.translation.m2m100_translator import (
            M2M100Translator,
        )
        translator = M2M100Translator(TranslationConfig(precision="float16"))
        calls = []

        class Model:
            def half(self):
                calls.append("half")

        assert translator._resolve_precision(Model(), "cpu") == "float16"
        assert calls == ["half"]
