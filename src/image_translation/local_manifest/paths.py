"""Repository and environment path resolution for launchers and tests."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_NAME = "dp"
DEFAULT_REPO = Path(r"D:\Drop\outlook.com\LocalBox\ImageTranslation")


def resolve_repo_root(explicit: str | Path | None = None) -> Path:
    """Resolve the ImageTranslation repository root."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_root = os.environ.get("IMAGE_TRANSLATION_REPO")
    if env_root:
        return Path(env_root).expanduser().resolve()
    here = Path(__file__).resolve()
    # src/image_translation/local_manifest/paths.py -> repo root
    candidate = here.parents[3]
    if (candidate / "environment.yml").is_file():
        return candidate
    if DEFAULT_REPO.is_dir() and (DEFAULT_REPO / "environment.yml").is_file():
        return DEFAULT_REPO
    return candidate


def resolve_src_root(repo_root: Path | None = None) -> Path:
    root = repo_root or resolve_repo_root()
    return root / "src"


def conda_environment_name() -> str:
    return os.environ.get("IMAGE_TRANSLATION_CONDA_ENV", DEFAULT_ENV_NAME)
