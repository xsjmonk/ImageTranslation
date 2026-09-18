"""Make the src directory importable for tests without pip install."""

import os
import sys
from pathlib import Path

# Use a workspace-local temp base when the system temp dir is inaccessible.
_workspace_tmp = Path(__file__).resolve().parent.parent / ".pytest_tmp"
_workspace_tmp.mkdir(exist_ok=True)
os.environ.setdefault("TMPDIR", str(_workspace_tmp))
os.environ.setdefault("TEMP", str(_workspace_tmp))
os.environ.setdefault("TMP", str(_workspace_tmp))

# Add src/ and repo root so `image_translation` and `LocalImageProcessing` import
_repo = Path(__file__).resolve().parent.parent
_src = _repo / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: test requires NVIDIA CUDA GPU and model download")
    config.addinivalue_line(
        "markers",
        "quality_regression: explicit GPU quality suite; set RUN_QUALITY_REGRESSION=1",
    )
    config.addinivalue_line(
        "markers",
        "html_gpu_quality: explicit HTML GPU suite; set RUN_HTML_GPU_QUALITY=1",
    )
    config.addinivalue_line(
        "markers",
        "nllb_smoke: explicit NLLB GPU smoke test; set RUN_NLLB_SMOKE=1",
    )
    config.addinivalue_line(
        "markers",
        "hymt2_smoke: explicit Hy-MT2 GPU smoke test; set RUN_HYMT2_SMOKE=1",
    )
    config.addinivalue_line(
        "markers",
        "hymt2_cached_smoke: cached Hy-MT2 snapshot smoke; set RUN_HYMT2_CACHED_SMOKE=1",
    )
