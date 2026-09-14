"""Optional Hy-MT2 GPU smoke test - downloads weights and uses GPU memory.

Run explicitly:
    .\\script\\Smoke-HyMt2Translation.ps1
Or:
    pytest tests/translation/test_hymt2_smoke.py -v -s
"""

from __future__ import annotations

import os
import time

import pytest

try:
    import torch

    _CUDA_OK = torch.cuda.is_available()
except Exception:
    _CUDA_OK = False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.hymt2_smoke,
    pytest.mark.skipif(
        os.environ.get("RUN_HYMT2_SMOKE") != "1",
        reason="Set RUN_HYMT2_SMOKE=1 or run .\\script\\Smoke-HyMt2Translation.ps1",
    ),
    pytest.mark.skipif(not _CUDA_OK, reason="NVIDIA CUDA GPU required"),
]


def test_hymt2_real_gpu_translation(tmp_path):
    """Load configured Hy-MT2 model and translate one Chinese sentence."""
    from image_translation.translation import TranslationConfig, create_translator

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    config = TranslationConfig(
        backend="hymt2",
        model_name="tencent/Hy-MT2-1.8B-FP8",
        model_family="hymt2",
        model_cache_dir=str(cache_dir),
        device="cuda",
        allow_cpu_fallback=False,
        max_input_tokens=8192,
    )
    started = time.perf_counter()
    translator = create_translator(config)
    result = translator.translate_text("你好，世界。")
    elapsed = time.perf_counter() - started
    info = translator.runtime_info
    assert result.translated_text
    assert info.ready
    assert info.backend == "hymt2"
    assert info.device.startswith("cuda:")
    print(f"translation={result.translated_text!r}")
    print(f"latency_seconds={elapsed:.3f}")
    print(f"model={info.model_name}")
    print(f"revision={info.model_revision}")
    print(f"device={info.device}")
    print(f"dtype={info.dtype}")
    print(f"cache_dir={info.cache_dir}")


if __name__ == "__main__":
    if not _CUDA_OK:
        print("SKIP: NVIDIA CUDA GPU required")
        raise SystemExit(2)
    test_hymt2_real_gpu_translation(None)
    print("Hy-MT2 smoke test passed.")
