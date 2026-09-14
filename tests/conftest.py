"""Make the src directory importable for tests without pip install."""

import sys
from pathlib import Path

# Add src/ to sys.path so `import image_translation` works
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: test requires NVIDIA CUDA GPU and model download")
    config.addinivalue_line(
        "markers",
        "hymt2_smoke: explicit Hy-MT2 GPU smoke test; set RUN_HYMT2_SMOKE=1",
    )
