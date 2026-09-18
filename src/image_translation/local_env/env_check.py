"""Environment validation for local deterministic image operations."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Iterable, List, Sequence, Tuple

# (import name, human-readable package label from environment.yml)
REQUIRED_IMPORTS: Sequence[Tuple[str, str]] = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("PIL", "pillow"),
    ("cv2", "opencv"),
    ("skimage", "scikit-image"),
    ("imageio", "imageio"),
    ("shapely", "shapely"),
    ("pyclipper", "pyclipper"),
    ("pydantic", "pydantic"),
    ("jsonschema", "jsonschema"),
)


@dataclass(frozen=True)
class EnvironmentIssue:
    package: str
    message: str


def check_imports(
    imports: Sequence[Tuple[str, str]] = REQUIRED_IMPORTS,
) -> List[EnvironmentIssue]:
    """Return import issues for packages declared in environment.yml."""
    issues: List[EnvironmentIssue] = []
    for module_name, package_label in imports:
        try:
            import_module(module_name)
        except ImportError as exc:
            issues.append(EnvironmentIssue(package_label, str(exc)))
    return issues


def check_basic_operations() -> List[EnvironmentIssue]:
    """Run read-only in-memory checks for core image/geometry libraries."""
    issues: List[EnvironmentIssue] = []
    for checker in (
        _check_pillow_in_memory,
        _check_opencv_basic,
        _check_numpy_scipy_skimage,
        _check_shapely_basic,
        _check_pyclipper_basic,
    ):
        issue = checker()
        if issue is not None:
            issues.append(issue)
    return issues


def run_environment_check() -> List[EnvironmentIssue]:
    """Validate imports and basic in-memory operations (read-only)."""
    issues = check_imports()
    if issues:
        return issues
    return check_basic_operations()


def format_environment_errors(issues: Iterable[EnvironmentIssue]) -> str:
    lines = [
        "Local image-processing environment is incomplete.",
        "Run .\\script\\Initialize-Env.ps1 from the ImageTranslation repository,",
        "then retry the environment check.",
        "",
        "Failures:",
    ]
    for issue in issues:
        lines.append(f"  - {issue.package}: {issue.message}")
    return "\n".join(lines)


def _check_pillow_in_memory() -> EnvironmentIssue | None:
    import io

    from PIL import Image

    image = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    loaded = Image.open(buffer)
    if loaded.size != (8, 8):
        return EnvironmentIssue("pillow", "in-memory PNG roundtrip size mismatch")
    return None


def _check_opencv_basic() -> EnvironmentIssue | None:
    import cv2
    import numpy as np

    array = np.zeros((16, 16, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(array, (3, 3), 0)
    if blurred.shape != array.shape:
        return EnvironmentIssue("opencv", "GaussianBlur changed array shape")
    return None


def _check_numpy_scipy_skimage() -> EnvironmentIssue | None:
    import numpy as np
    import scipy.ndimage
    from skimage import filters

    array = np.ones((12, 12), dtype=np.float64)
    smoothed = scipy.ndimage.gaussian_filter(array, sigma=1.0)
    edges = filters.sobel(smoothed)
    if edges.shape != array.shape:
        return EnvironmentIssue("scikit-image", "sobel output shape mismatch")
    return None


def _check_shapely_basic() -> EnvironmentIssue | None:
    from shapely.geometry import Polygon

    polygon = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    if polygon.area != 2:
        return EnvironmentIssue("shapely", "basic polygon area check failed")
    return None


def _check_pyclipper_basic() -> EnvironmentIssue | None:
    import pyclipper

    if not hasattr(pyclipper, "Pyclipper"):
        return EnvironmentIssue("pyclipper", "Pyclipper class is unavailable")
    return None
