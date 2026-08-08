"""Configuration loader – resolves config.json, validates, returns AppConfig."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .defaults import build_default_config
from .models import AppConfig


class ConfigLoadError(Exception):
    """Raised when configuration cannot be loaded or is invalid."""


def _find_repo_root() -> Path:
    """Return the repository root (parent of src/image_translation)."""
    this_file = Path(__file__).resolve()
    # config/loader.py -> config -> image_translation -> src -> repo_root
    return this_file.parent.parent.parent.parent


def _resolve_config_path(explicit_path: Optional[Path]) -> Optional[Path]:
    """Determine which config file to use."""
    if explicit_path is not None:
        resolved = explicit_path.resolve()
        if not resolved.exists():
            raise ConfigLoadError(f"Config file not found: {resolved}")
        return resolved
    # Try repo-root config.json
    default_path = _find_repo_root() / "config.json"
    if default_path.exists():
        return default_path
    return None


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load and validate configuration.

    Resolution:
    1. If config_path is provided, it must exist and be valid.
    2. Otherwise look for <repo-root>/config.json.
    3. If neither exists, return built-in defaults.

    Raises ConfigLoadError on invalid/missing explicit config.
    """
    resolved = _resolve_config_path(config_path)

    if resolved is None:
        return build_default_config()

    try:
        raw_text = resolved.read_text(encoding="utf-8")
        raw = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ConfigLoadError(f"Invalid JSON in {resolved}: {e}") from e

    try:
        config = AppConfig.model_validate(raw)
    except Exception as e:
        raise ConfigLoadError(f"Validation failed for {resolved}: {e}") from e

    return config
