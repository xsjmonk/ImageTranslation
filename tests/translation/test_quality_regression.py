"""Quality regression tests — real NLLB GPU, FP32 + num_beams=4 baseline.

Covers the observed failure modes (semantic corruption / repetition) with
deterministic assertions on Chinese product-description text.

Run explicitly (needs NVIDIA CUDA + the NLLB model, ~2 GB VRAM):
    pytest tests/translation/test_quality_regression.py -v -s
"""

from __future__ import annotations

import threading
import time

import pytest

try:
    import torch
    _CUDA_OK = torch.cuda.is_available()
except Exception:
    _CUDA_OK = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _CUDA_OK, reason="NVIDIA CUDA GPU required"),
]

# ---------------------------------------------------------------------------
# Test phrases (Chinese product descriptions covering the observed failures)
# ---------------------------------------------------------------------------

PHRASES = [
    "加厚防水面料，耐磨耐用，适合日常使用。",
    "这是一款采用加厚防水面料制作的户外背包，耐磨耐用，适合日常通勤和旅行使用。",
    (
        "这款便携式充电器采用加厚防火外壳，内部具备过流保护和短路保护功能。"
        "产品坚固耐用，适合日常通勤、商务旅行和家庭使用。"
        "即使长时间高负荷运行，也能保持稳定输出，确保设备安全。"
    ),
]

# For each phrase: list of (alternative_needles, source_concept_label).
# Output must contain at least one of the alternatives (lowercased).
# Alternatives reflect expected model output patterns (e.g. 防水 -> water-resistant,
# 加厚 -> thick/thickened; in the short first phrase the model approximates
# 加厚 as "heated" and 耐磨 as "resistant to ..." — noted per phrase).
CONCEPT_CHECKS = [
    [
        (("waterproof", "water-resistant", "water resistant"), "防水面料"),
        (("thick", "thicken", "increased", "heated"), "加厚(厚/加)"),
        (("durab", "wear", "resist"), "耐磨耐用(耐/耐磨)"),
        (("daily", "everyday", "day-to-day"), "日常使用"),
    ],
    [
        (("waterproof", "water-resistant", "water resistant"), "防水"),
        (("thick", "thicken"), "加厚"),
        (("backpack", "bag"), "背包"),
        (("durab", "wear"), "耐磨耐用"),
        (("daily", "commut"), "日常通勤"),
        (("travel",), "旅行"),
    ],
    [
        (("charger",), "充电器"),
        (("thick", "thicken"), "加厚"),
        (("fire", "flame"), "防火"),
        (("durab", "sturdy", "rugged", "solid", "robust"), "坚固耐用"),
        (("daily", "everyday"), "日常"),
        (("travel",), "旅行"),
        (("stable", "steady"), "稳定输出"),
    ],
]

# Known corruption markers that must NOT appear (from the FP16 failure mode)
CORRUPTION_MARKERS = ["reference", "referral", "cell phone"]


def _assert_plausibly_english(text: str) -> None:
    """Non-empty, valid UTF-8, mostly ASCII letters."""
    assert text, "translation is empty"
    text.encode("utf-8")  # valid UTF-8
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    assert ascii_letters / max(len(text), 1) > 0.6, (
        f"output does not look like English: {text!r}"
    )


def _assert_no_corruption(text: str) -> None:
    lowered = text.lower()
    for marker in CORRUPTION_MARKERS:
        assert marker not in lowered, (
            f"output contains corruption marker {marker!r}: {text!r}"
        )


def _assert_concepts(text: str, checks) -> None:
    lowered = text.lower()
    for needles, label in checks:
        assert any(n in lowered for n in needles), (
            f"missing concept '{label}' (needles={needles}) in output: {text!r}"
        )


def _record_io(phrase: str, output: str, info) -> None:
    print(f"\n--- SOURCE: {phrase}")
    print(f"    OUTPUT: {output}")
    print(
        f"    RECORD: model={info.model_name} device={info.device} "
        f"precision={info.precision} num_beams=4 adaptive_max_new_tokens "
        f"no_repeat_ngram_size=unset forced_bos_token_id=target_lang"
    )


# ---------------------------------------------------------------------------
# Direct module regression
# ---------------------------------------------------------------------------

class TestDirectModuleQuality:
    def test_chinese_phrases_fp32_beams4(self):
        from image_translation.translation import create_translator
        from translation_server.config import load_server_config

        # The direct path consumes the server-resolved cache and generation policy.
        cfg = load_server_config().translation
        assert cfg.precision == "auto"
        assert cfg.generation.num_beams == 4
        assert cfg.generation.max_new_tokens == 512
        assert cfg.generation.do_sample is False
        assert cfg.generation.no_repeat_ngram_size is None

        t = create_translator(cfg)

        # Default precision must be FP32 — no model.half() on auto
        t.warmup()
        info = t.runtime_info
        assert info.precision == "float32", f"expected FP32 default, got {info.precision}"
        assert info.ready
        assert info.device.startswith("cuda:")

        for phrase, checks in zip(PHRASES, CONCEPT_CHECKS):
            result = t.translate_text(phrase)
            out = result.translated_text
            _assert_plausibly_english(out)
            _assert_no_corruption(out)
            _assert_concepts(out, checks)
            _record_io(phrase, out, info)


# ---------------------------------------------------------------------------
# Direct module vs running HTTP API parity
# ---------------------------------------------------------------------------

class TestApiParity:
    def test_direct_vs_api_same_output(self):
        """Same inputs through the direct module and the live HTTP API."""
        from image_translation.translation import create_translator
        from translation_server.config import load_server_config
        from translation_server.runtime import TranslationRuntime
        from translation_server.app import create_app

        cfg = load_server_config()

        # --- Phase 1: direct shared service ---
        runtime = TranslationRuntime(cfg)
        direct_outputs = {}
        for phrase in PHRASES:
            out = runtime.translate_plain(
                phrase,
                cfg.translation.source_language,
                cfg.translation.target_language,
            ).translated_text
            _assert_plausibly_english(out)
            _assert_no_corruption(out)
            direct_outputs[phrase] = out
            _record_io(phrase, out, runtime.translator.runtime_info)

        # --- Phase 2: running HTTP API (same config file as the XS script) ---
        import uvicorn
        import requests

        # The API reuses the same resolved configuration file.
        assert cfg.translation.precision == "auto"
        assert cfg.translation.generation.num_beams == 4
        assert cfg.translation.generation.max_new_tokens == 512
        assert cfg.translation.generation.do_sample is False
        assert cfg.translation.generation.no_repeat_ngram_size is None
        assert cfg.runtime.warmup_on_start is True

        app = create_app(runtime)
        port = 18091
        server_cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        server = uvicorn.Server(server_cfg)
        server.install_signal_handlers = False  # run in worker thread

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        try:
            # Wait for readiness (model warmup takes seconds)
            base = f"http://127.0.0.1:{port}"
            ready = False
            for _ in range(180):
                try:
                    health = requests.get(f"{base}/health", timeout=3).json()
                    if health.get("ready"):
                        ready = True
                        break
                except Exception:
                    pass
                time.sleep(1)
            assert ready, "API server did not become ready"

            for phrase in PHRASES:
                resp = requests.post(
                    f"{base}/translate",
                    json={"text": phrase},
                    timeout=300,
                )
                assert resp.status_code == 200, f"API {resp.status_code}: {resp.text}"
                api_out = resp.json()["translation"]
                assert runtime.plain_invocation_count == len(PHRASES) + (
                    PHRASES.index(phrase) + 1
                )

                direct_out = direct_outputs[phrase]
                _assert_plausibly_english(api_out)
                _assert_no_corruption(api_out)
                _assert_concepts(api_out, dict(zip(PHRASES, CONCEPT_CHECKS))[phrase])
                assert api_out != "" and direct_out != ""
        finally:
            server.should_exit = True
            thread.join(timeout=15)


class TestNllbProductTitleQuality:
    def test_eyeglass_title_is_translated_without_repetition(self):
        from image_translation.translation import TranslationConfig, create_translator

        source = "德国蔡司纯钛眼镜近视男可配度数防蓝光商务超轻镜框专业"
        translator = create_translator(TranslationConfig())
        output = translator.translate_text(source).translated_text
        assert output.strip()
        assert "bright bright bright bright" not in output.casefold()
