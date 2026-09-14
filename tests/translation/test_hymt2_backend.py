"""Hy-MT2 additive backend tests — deterministic, no weight downloads."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from image_translation.translation.config import GenerationConfig, StructuredConfig, TranslationConfig
from image_translation.translation.factory import create_translator
from image_translation.translation.hymt2_translator import HyMt2Translator
from image_translation.translation.language_names import resolve_language_name
from image_translation.translation.models import ResolvedModel
from image_translation.translation.structured_translation import StructuredTranslator
from translation_server.config import load_server_config


class FakeBatchEncoding(Mapping):
    """Mapping-like tokenizer output that is not a built-in dict."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


class FakeTokenizer:
    chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.plain_tokenizer_used = False
        self.chat_template_calls: list[dict] = []

    def apply_chat_template(
        self,
        messages_batch,
        return_tensors="pt",
        return_dict=True,
        padding=True,
        add_generation_prompt=True,
        truncation=False,
    ):
        self.chat_template_calls.append(
            {
                "return_dict": return_dict,
                "add_generation_prompt": add_generation_prompt,
                "padding": padding,
                "truncation": truncation,
            }
        )
        rows = []
        for messages in messages_batch:
            text = messages[0]["content"]
            rows.append([10 + len(text) % 5, 20 + len(text) % 3, 30])
        max_len = max(len(row) for row in rows)
        padded = []
        masks = []
        for row in rows:
            pad_len = max_len - len(row)
            padded.append(row + [0] * pad_len)
            masks.append([1] * len(row) + [0] * pad_len)
        return FakeBatchEncoding(
            {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        )

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=False):
        self.plain_tokenizer_used = True
        raise AssertionError(
            "plain tokenizer must not be used when a chat template is available"
        )

    def decode(self, token_ids, skip_special_tokens=True):
        values = tuple(
            token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        )
        decode_map = {
            (100, 110): "Hello",
            (101, 111): "World",
            (102, 112): "Alpha",
            (103, 113): "Beta",
        }
        return decode_map.get(values, f"GEN-{'-'.join(map(str, values))}")


class FakePlainTokenizer(FakeTokenizer):
    """Test double without a chat template (compatibility-only path)."""

    chat_template = None

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("apply_chat_template must not be called")

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=False):
        batch = []
        for text in texts:
            batch.append([len(text) % 7 + 1, len(text) % 5 + 2, 3])
        input_ids = torch.tensor(batch)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class FakeCausalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, attention_mask=None, **kwargs):
        batch, _width = input_ids.shape
        suffix_rows = [[100 + index, 110 + index] for index in range(batch)]
        suffix = torch.tensor(suffix_rows, dtype=input_ids.dtype)
        return torch.cat([input_ids, suffix], dim=1)

    def eval(self):
        return self


@pytest.fixture
def hymt2_config(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return TranslationConfig(
        backend="hymt2",
        model_name="tencent/Hy-MT2-Test",
        model_family="hymt2",
        model_cache_dir=str(cache_dir),
        device="cpu",
        allow_cpu_fallback=True,
        max_input_tokens=8192,
        generation=GenerationConfig(
            max_new_tokens=32,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
        ),
    )


def _patch_hymt2_load(monkeypatch, tmp_path, tokenizer_factory=FakeTokenizer):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_text("x", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")

    resolved = ResolvedModel(
        snapshot_path=str(snapshot),
        model_name="tencent/Hy-MT2-Test",
        revision="main",
        cache_dir=str(tmp_path / "cache"),
        cache_status="cache_hit",
        offline=False,
        model_family="hymt2",
    )
    monkeypatch.setattr(
        HyMt2Translator,
        "_resolve_model_snapshot",
        lambda self: resolved,
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer_factory(),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: FakeCausalModel(),
    )


class TestBackendConfiguration:
    def test_default_backend_is_current(self):
        cfg = TranslationConfig()
        assert cfg.backend == "current"
        assert create_translator(cfg).__class__.__name__ == "Seq2SeqTranslator"

    def test_explicit_current_backend(self):
        cfg = TranslationConfig(backend="current")
        assert create_translator(cfg).__class__.__name__ == "Seq2SeqTranslator"

    def test_hymt2_backend_factory(self, hymt2_config):
        translator = create_translator(hymt2_config)
        assert isinstance(translator, HyMt2Translator)
        assert translator.name.startswith("hymt2@")

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValueError, match="backend"):
            TranslationConfig(backend="unknown")

    def test_hymt2_rejects_seq2seq_model_name(self):
        with pytest.raises(ValueError, match="Hy-MT2 model_name"):
            TranslationConfig(
                backend="hymt2",
                model_name="facebook/nllb-200-distilled-600M",
            )

    def test_invalid_generation_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            GenerationConfig(temperature=0)

    def test_server_config_loads_backend(self, tmp_path):
        config_path = tmp_path / "server.json"
        config_path.write_text(
            json.dumps(
                {
                    "server": {"model_cache_dir": str(tmp_path / "cache")},
                    "translation": {
                        "backend": "hymt2",
                        "model_name": "tencent/Hy-MT2-1.8B-FP8",
                    },
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "cache").mkdir()
        cfg = load_server_config(config_path)
        assert cfg.translation.backend == "hymt2"
        assert cfg.translation.model_name == "tencent/Hy-MT2-1.8B-FP8"
        assert cfg.translation.model_cache_dir == str((tmp_path / "cache").resolve())

    def test_backend_switch_requires_only_json_change(self, tmp_path):
        cache = str(tmp_path / "cache")
        (tmp_path / "cache").mkdir()
        base = {
            "server": {"model_cache_dir": cache},
            "translation": {
                "source_language": "zho_Hans",
                "generation": {"num_beams": 1, "do_sample": False},
                "active_model": "nllb",
                "models": {
                    "nllb": {
                        "backend": "current",
                        "model_name": "facebook/nllb-200-distilled-600M",
                        "model_family": "nllb",
                        "model_revision": "main",
                    },
                    "hymt2": {
                        "backend": "hymt2",
                        "model_name": "tencent/Hy-MT2-1.8B-FP8",
                        "model_family": "hymt2",
                        "model_revision": "main",
                    },
                },
            },
        }
        current_path = tmp_path / "current.json"
        hymt2_path = tmp_path / "hymt2.json"
        current_path.write_text(json.dumps(base), encoding="utf-8")
        hymt2_path.write_text(
            json.dumps(
                {
                    **base,
                    "translation": {**base["translation"], "active_model": "hymt2"},
                }
            ),
            encoding="utf-8",
        )
        current_cfg = load_server_config(current_path)
        hymt2_cfg = load_server_config(hymt2_path)
        assert current_cfg.translation.backend == "current"
        assert hymt2_cfg.translation.backend == "hymt2"
        assert (
            current_cfg.translation.generation.num_beams
            == hymt2_cfg.translation.generation.num_beams
        )
        assert current_cfg.server.model_cache_dir == hymt2_cfg.server.model_cache_dir


class TestHyMt2AdapterContract:
    def test_prompt_uses_configured_target_language(self):
        prompt = HyMt2Translator(
            TranslationConfig(
                backend="hymt2",
                model_name="tencent/Hy-MT2-Test",
                target_language="eng_Latn",
            )
        )._build_prompt("你好", "zh", "en")
        assert "English" in prompt
        assert "你好" in prompt
        assert resolve_language_name("zh") == "Chinese"

    def test_only_generated_tokens_are_decoded(self, monkeypatch, hymt2_config, tmp_path):
        _patch_hymt2_load(monkeypatch, tmp_path)
        translator = HyMt2Translator(hymt2_config)
        result = translator.translate_text("德国蔡司纯钛眼镜")
        assert result.translated_text == "Hello"
        assert "Translate the following text into English" not in result.translated_text
        assert result.translated_text != "德国蔡司纯钛眼镜"
        assert not result.translated_text.startswith("translated-")
        assert translator._encode_used_chat_template is True

    def test_model_load_failure_is_explicit(self, monkeypatch, hymt2_config, tmp_path):
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        monkeypatch.setattr(
            HyMt2Translator,
            "_resolve_model_snapshot",
            lambda self: ResolvedModel(
                snapshot_path=str(snapshot),
                model_name="tencent/Hy-MT2-Test",
                revision="main",
                cache_dir=str(tmp_path),
                cache_status="cache_hit",
                offline=False,
                model_family="hymt2",
            ),
        )

        def _fail(*args, **kwargs):
            raise RuntimeError("tokenizer load failed")

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            _fail,
        )
        translator = HyMt2Translator(hymt2_config)
        with pytest.raises(Exception, match="Failed to load Hy-MT2 tokenizer"):
            translator.translate_text("你好")

    def test_no_fallback_to_current_backend(self, monkeypatch, hymt2_config, tmp_path):
        _patch_hymt2_load(monkeypatch, tmp_path)
        translator = create_translator(hymt2_config)
        assert translator.__class__.__name__ == "HyMt2Translator"
        monkeypatch.setattr(
            translator,
            "_translate_impl",
            MagicMock(side_effect=RuntimeError("inference failed")),
        )
        with pytest.raises(RuntimeError, match="inference failed"):
            translator.translate_text("你好")


class TestPlainTextRegression:
    PRODUCT_TITLE = "德国蔡司纯钛眼镜近视男可配度数防蓝光商务超轻镜框专业"

    @pytest.fixture(autouse=True)
    def _load(self, monkeypatch, hymt2_config, tmp_path):
        _patch_hymt2_load(monkeypatch, tmp_path)
        self.translator = HyMt2Translator(hymt2_config)

    def test_product_title_translates(self):
        result = self.translator.translate_text(self.PRODUCT_TITLE)
        assert result.translated_text
        assert result.translated_text != self.PRODUCT_TITLE

    def test_short_phrase_not_invented_sentence(self):
        result = self.translator.translate_text("加厚防水面料")
        assert result.translated_text == "Hello"
        assert not result.translated_text.startswith("translated-")

    def test_mixed_model_numbers_preserved_in_source(self):
        source = "适合 iPhone 15 Pro Max 与 UV400 防护"
        result = self.translator.translate_text(source)
        assert "iPhone 15 Pro Max" in source
        assert result.translated_text

    def test_batch_preserves_order(self):
        texts = ["第一段", "第二段", "第三段"]
        results = self.translator.translate_batch_texts(texts)
        assert [item.source_text for item in results] == texts
        assert [item.translated_text for item in results] == [
            "Hello",
            "World",
            "Alpha",
        ]


class TestHyMt2ChatTemplateRegression:
    def test_batchencoding_is_accepted_and_chat_template_is_used(
        self, monkeypatch, hymt2_config, tmp_path
    ):
        _patch_hymt2_load(monkeypatch, tmp_path)
        translator = HyMt2Translator(hymt2_config)
        result = translator.translate_text("你好")
        tokenizer = translator._tokenizer
        assert tokenizer is not None
        assert result.translated_text == "Hello"
        assert tokenizer.chat_template_calls
        assert tokenizer.chat_template_calls[-1]["return_dict"] is True
        assert tokenizer.chat_template_calls[-1]["add_generation_prompt"] is True
        assert tokenizer.plain_tokenizer_used is False
        assert translator._encode_used_chat_template is True

    def test_chat_template_failure_is_explicit(self, monkeypatch, hymt2_config, tmp_path):
        _patch_hymt2_load(monkeypatch, tmp_path)
        translator = HyMt2Translator(hymt2_config)
        translator._ensure_loaded()
        tokenizer = translator._tokenizer
        assert tokenizer is not None

        def _fail(*args, **kwargs):
            raise RuntimeError("template exploded")

        tokenizer.apply_chat_template = _fail
        with pytest.raises(Exception, match="Failed to apply Hy-MT2 chat template"):
            translator.translate_text("你好")

    def test_plain_tokenizer_only_without_chat_template(
        self, monkeypatch, hymt2_config, tmp_path
    ):
        _patch_hymt2_load(
            monkeypatch,
            tmp_path,
            tokenizer_factory=lambda: FakePlainTokenizer(),
        )
        translator = HyMt2Translator(hymt2_config)
        result = translator.translate_text("你好")
        assert result.translated_text == "Hello"
        assert translator._encode_used_chat_template is False

    def test_padded_batch_decoding_has_no_prompt_leakage(
        self, monkeypatch, hymt2_config, tmp_path
    ):
        _patch_hymt2_load(monkeypatch, tmp_path)
        translator = HyMt2Translator(hymt2_config)
        short = "短"
        long = "这是一个明显更长的中文句子用于测试填充批次"
        results = translator.translate_batch_texts([short, long])
        assert [item.translated_text for item in results] == ["Hello", "World"]
        assert all("Translate the following text" not in item.translated_text for item in results)


class TestHyMt2StructuredMixedHtmlRegression:
    HTML = (
        '<p>德国蔡司<span class="brand">ZEISS</span>纯钛眼镜</p>'
        '<p>适合 iPhone 16 Pro 与 USB-C1 使用</p>&nbsp;&amp;'
    )

    def test_mixed_document_preserves_structure_and_order(
        self, monkeypatch, hymt2_config, tmp_path
    ):
        _patch_hymt2_load(monkeypatch, tmp_path)
        translator = HyMt2Translator(hymt2_config)
        structured = StructuredTranslator(
            translator,
            StructuredConfig(
                max_segment_tokens=450,
                max_target_tokens=400,
                batch_size=2,
            ),
            translation_config=hymt2_config,
        )
        result = structured.translate(self.HTML)
        assert "<span" in result.translated_html
        assert "ZEISS" in result.translated_html
        assert "&nbsp;" in result.translated_html
        assert "&amp;" in result.translated_html
        assert "iPhone 16 Pro" in result.translated_html
        assert result.translated_html.index("<p>") < result.translated_html.index(
            "iPhone 16 Pro"
        )
        assert "Hello" in result.translated_html or "World" in result.translated_html
        assert "UCLA" not in result.translated_html


class TestHtmlRegressionUsesStructuredPipeline:
    HTML = (
        '德国蔡司<span class="brand">ZEISS</span>纯钛眼镜<br>&nbsp;'
        "适合 iPhone 15 Pro Max 使用"
    )

    def test_structured_pipeline_delegates_to_translator(self, monkeypatch, hymt2_config, tmp_path):
        _patch_hymt2_load(monkeypatch, tmp_path)
        translator = HyMt2Translator(hymt2_config)
        calls = []

        def _batch(texts, source_lang="zh", target_lang="en", max_new_tokens=None, style=None):
            calls.extend(texts)
            return [
                translator._translate_impl([text], source_lang, target_lang)[0]
                for text in texts
            ]

        monkeypatch.setattr(translator, "translate_batch_texts", _batch)
        structured = StructuredTranslator(
            translator,
            StructuredConfig(max_segment_tokens=450, max_target_tokens=400),
            translation_config=hymt2_config,
        )
        result = structured.translate(self.HTML)
        assert "<span" in result.translated_html
        assert "ZEISS" in result.translated_html
        assert calls


class TestRealModelFactorySmoke:
    def test_hymt2_translator_can_be_constructed_without_download(self, hymt2_config):
        translator = create_translator(hymt2_config)
        assert isinstance(translator, HyMt2Translator)
        assert translator.runtime_info.backend == "hymt2"


class TestBoundaryChecks:
    def test_translation_module_has_no_fastapi_imports(self):
        package = Path(__file__).parents[2] / "src" / "image_translation" / "translation"
        forbidden = ("fastapi", "translation_server")
        for path in package.rglob("*.py"):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    lowered = stripped.lower()
                    for name in forbidden:
                        assert name not in lowered, f"{name} import in {path}"
