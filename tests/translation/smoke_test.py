"""GPU integration test — requires CUDA, downloads NLLB model (~1.7 GB).

Excluded from default unit suite. Run explicitly:
    python tests/translation/smoke_test.py
Or:
    pytest tests/translation/smoke_test.py -v -s
"""

import pytest

# Auto-skip if CUDA not available
try:
    import torch
    _CUDA_OK = torch.cuda.is_available()
except Exception:
    _CUDA_OK = False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _CUDA_OK, reason="NVIDIA CUDA GPU required"),
]


def test_nllb_real_gpu_translation():
    """Real GPU translation: loads facebook/nllb-200-distilled-600M, translates zh→en."""
    from image_translation.translation import TranslationConfig, create_translator

    t = create_translator(TranslationConfig())

    r1 = t.translate_text("你好")
    assert r1.translated_text, "Empty translation for 你好"
    print(f"你好 -> {r1.translated_text}")

    r2 = t.translate_text("加厚防水面料")
    assert r2.translated_text, "Empty translation for 加厚防水面料"
    assert r2.translated_text != "加厚防水面料"
    print(f"加厚防水面料 -> {r2.translated_text}")

    info = t.runtime_info
    assert info.ready
    assert info.cuda_available
    assert info.device.startswith("cuda:")
    print(f"Model: {info.model_name}")
    print(f"Device: {info.device}")
    print(f"GPU: {info.gpu_name}")


if __name__ == "__main__":
    if not _CUDA_OK:
        print("SKIP: NVIDIA CUDA GPU required")
        raise SystemExit(2)
    test_nllb_real_gpu_translation()
    print("GPU smoke test passed.")
