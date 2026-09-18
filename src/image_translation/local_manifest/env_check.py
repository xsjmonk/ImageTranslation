"""Validate conda packages required by the local manifest image processor."""

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
    ("typer", "typer"),
    ("jsonschema", "jsonschema"),
)


@dataclass(frozen=True)
class EnvironmentIssue:
    package: str
    message: str


def check_local_image_environment(
    imports: Sequence[Tuple[str, str]] = REQUIRED_IMPORTS,
) -> List[EnvironmentIssue]:
    """Return import issues for packages needed by local image tools."""
    issues: List[EnvironmentIssue] = []
    for module_name, package_label in imports:
        try:
            import_module(module_name)
        except ImportError as exc:
            issues.append(EnvironmentIssue(package_label, str(exc)))
    return issues


def format_environment_errors(issues: Iterable[EnvironmentIssue]) -> str:
    lines = [
        "Local image-processing environment is incomplete.",
        "Run .\\script\\Initialize-Env.ps1 from the ImageTranslation repository,",
        "then retry process-image-manifest.",
        "",
        "Missing or broken packages:",
    ]
    for issue in issues:
        lines.append(f"  - {issue.package}: {issue.message}")
    return "\n".join(lines)
