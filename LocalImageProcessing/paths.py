"""Repository and module path resolution — cwd-independent."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_REPO = Path(r"D:\Drop\outlook.com\LocalBox\ImageTranslation")
_MODULE_ROOT = Path(__file__).resolve().parent


def module_root() -> Path:
    return _MODULE_ROOT


def resolve_repo_root(explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_root = os.environ.get("IMAGE_TRANSLATION_REPO")
    if env_root:
        return Path(env_root).expanduser().resolve()
    candidate = _MODULE_ROOT.parent
    if (candidate / "environment.yml").is_file():
        return candidate
    if DEFAULT_REPO.is_dir() and (DEFAULT_REPO / "environment.yml").is_file():
        return DEFAULT_REPO
    return candidate


def ensure_import_paths(repo_root: Path | None = None) -> Path:
    """Ensure the repository root is importable for `python -m LocalImageProcessing`."""
    root = resolve_repo_root(repo_root)
    repo_str = str(root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return root
