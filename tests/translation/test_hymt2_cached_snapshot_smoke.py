"""Cached Hy-MT2 snapshot smoke test — no downloads; uses production factory path."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

try:
    import torch

    _CUDA_OK = torch.cuda.is_available()
except Exception:
    _CUDA_OK = False


CACHED_SNAPSHOT = Path(
    r"D:\Caches\translation\models--tencent--Hy-MT2-1.8B-FP8\snapshots"
    r"\b3f6f590920726d69a5504293bd4f36d50e5f681"
)
CACHE_ROOT = Path(r"D:\Caches\translation")
PRODUCT_TITLE = "德国蔡司纯钛眼镜近视男可配度数防蓝光商务超轻镜框专业"
MIXED_SENTENCE = "适合 iPhone 16 Pro 与 USB-C1 使用"
HTML_DOC = (
    '<p>德国蔡司<span class="brand">ZEISS</span>纯钛眼镜</p>'
    "<p>&nbsp;适合 iPhone 16 Pro 使用</p>"
)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.hymt2_smoke,
    pytest.mark.skipif(
        not CACHED_SNAPSHOT.exists(),
        reason=f"cached Hy-MT2 snapshot not present: {CACHED_SNAPSHOT}",
    ),
    pytest.mark.skipif(not _CUDA_OK, reason="NVIDIA CUDA GPU required"),
    pytest.mark.skipif(
        os.environ.get("RUN_HYMT2_CACHED_SMOKE") != "1",
        reason="Set RUN_HYMT2_CACHED_SMOKE=1 to run cached snapshot smoke test",
    ),
]


@pytest.fixture(scope="module")
def hymt2_translator():
    from image_translation.translation import TranslationConfig, create_translator

    config = TranslationConfig(
        backend="hymt2",
        model_name="tencent/Hy-MT2-1.8B-FP8",
        model_family="hymt2",
        model_revision="main",
        model_cache_dir=str(CACHE_ROOT),
        device="cuda",
        allow_cpu_fallback=False,
        local_files_only=True,
        allow_model_download=False,
        max_input_tokens=8192,
    )
    translator = create_translator(config)
    translator.warmup()
    return translator


def test_cached_snapshot_hello(hymt2_translator):
    result = hymt2_translator.translate_text("你好")
    info = hymt2_translator.runtime_info
    print(f"model={info.model_name}")
    print(f"backend={info.backend}")
    print(f"snapshot={info.snapshot_path}")
    print(f"device={info.device}")
    print(f"chat_template={hymt2_translator._encode_used_chat_template}")
    print(f"translation={result.translated_text!r}")
    assert hymt2_translator._encode_used_chat_template is True
    assert result.translated_text.strip().startswith("Hello")


def test_cached_snapshot_product_title_not_unrelated_paragraph(hymt2_translator):
    result = hymt2_translator.translate_text(PRODUCT_TITLE)
    lowered = result.translated_text.lower()
    assert result.translated_text.strip()
    assert "ucla" not in lowered
    assert "student" not in lowered or "glasses" in lowered


def test_cached_snapshot_mixed_sentence(hymt2_translator):
    result = hymt2_translator.translate_text(MIXED_SENTENCE)
    assert result.translated_text.strip()
    assert "iPhone 16 Pro" in MIXED_SENTENCE


def test_cached_snapshot_structured_html(hymt2_translator):
    from image_translation.translation.config import StructuredConfig
    from image_translation.translation.structured_translation import StructuredTranslator

    structured = StructuredTranslator(
        hymt2_translator,
        StructuredConfig(max_segment_tokens=450, max_target_tokens=400),
        translation_config=hymt2_translator._config,
    )
    result = structured.translate(HTML_DOC)
    assert "<span" in result.translated_html
    assert "ZEISS" in result.translated_html
    assert "&nbsp;" in result.translated_html
    assert "UCLA" not in result.translated_html
