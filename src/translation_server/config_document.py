"""Comment-capable translation-server configuration document parsing.

Uses the ``json5`` library (JSON5 syntax) so configuration files may contain
``// line`` and ``/* block */`` comments without corrupting string literals.
Comments are documentation only; they are not preserved or round-tripped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json5

_SUPPORTED_BACKENDS = frozenset({"current", "hymt2"})
_ALLOWED_TOP_LEVEL = frozenset(
    {"server", "runtime", "translation", "structured", "quality"}
)
_MODEL_PROFILE_KEYS = frozenset(
    {"backend", "model_name", "model_family", "model_revision"}
)
_LEGACY_MODEL_KEYS = frozenset(
    {"backend", "model_name", "model_family", "model_revision"}
)
_TRANSLATION_SHARED_KEYS = frozenset(
    {
        "active_model",
        "models",
        "source_language",
        "target_language",
        "default_style",
        "device",
        "cuda_device",
        "allow_cpu_fallback",
        "precision",
        "batch_size",
        "max_input_characters",
        "max_input_tokens",
        "allow_model_download",
        "local_files_only",
        "commercial_use",
        "generation",
        "model_cache_dir",
        "cache_dir",
        "offline",
    }
)


class ConfigDocumentError(ValueError):
    """Raised when a configuration document cannot be parsed or normalized."""


def parse_config_text(text: str, path: Path | str | None = None) -> dict[str, Any]:
    """Parse JSON/JSONC configuration text using JSON5."""
    location = str(path) if path is not None else "<config>"
    try:
        value = json5.loads(text)
    except ValueError as exc:
        raise ConfigDocumentError(f"{location}: {exc}") from exc
    except Exception as exc:
        raise ConfigDocumentError(f"{location}: invalid configuration: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigDocumentError(f"{location}: root value must be an object")
    return value


def parse_config_file(path: Path) -> dict[str, Any]:
    """Parse a configuration file from disk."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Server config not found: {resolved}")
    return parse_config_text(resolved.read_text(encoding="utf-8"), resolved)


def validate_top_level(raw: dict[str, Any], path: Path | str | None = None) -> None:
    """Reject unknown top-level keys used by the server loader."""
    location = str(path) if path is not None else "<config>"
    unknown = sorted(set(raw) - _ALLOWED_TOP_LEVEL)
    if unknown:
        raise ConfigDocumentError(
            f"{location}: unknown top-level key(s): {', '.join(unknown)}"
        )


def select_model_profile(
    trans_raw: dict[str, Any],
    path: Path | str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve the active model profile from the translation section."""
    location = str(path) if path is not None else "<config>"
    if not isinstance(trans_raw, dict):
        raise ConfigDocumentError(f"{location}: translation must be an object")

    models = trans_raw.get("models")
    if models is not None:
        if not isinstance(models, dict) or not models:
            raise ConfigDocumentError(
                f"{location}: translation.models must be a non-empty object"
            )
        active = trans_raw.get("active_model", "nllb")
        if not isinstance(active, str) or not active.strip():
            raise ConfigDocumentError(
                f"{location}: translation.active_model must be a non-empty string"
            )
        if active not in models:
            known = ", ".join(sorted(models))
            raise ConfigDocumentError(
                f"{location}: translation.active_model {active!r} is not defined "
                f"in translation.models (available: {known})"
            )
        profile = models[active]
        if not isinstance(profile, dict):
            raise ConfigDocumentError(
                f"{location}: translation.models[{active!r}] must be an object"
            )
        unknown_profile = sorted(set(profile) - _MODEL_PROFILE_KEYS)
        if unknown_profile:
            raise ConfigDocumentError(
                f"{location}: translation.models[{active!r}] has unknown key(s): "
                f"{', '.join(unknown_profile)}"
            )
        for key in _LEGACY_MODEL_KEYS:
            if key in trans_raw:
                raise ConfigDocumentError(
                    f"{location}: translation.{key} must not appear alongside "
                    f"translation.models; configure model identity only in the "
                    f"selected profile"
                )
        backend = profile.get("backend")
        if backend not in _SUPPORTED_BACKENDS:
            raise ConfigDocumentError(
                f"{location}: translation.models[{active!r}].backend must be "
                f"one of {sorted(_SUPPORTED_BACKENDS)}"
            )
        if backend != active and active in _SUPPORTED_BACKENDS:
            raise ConfigDocumentError(
                f"{location}: translation.models[{active!r}].backend must equal "
                f"{active!r}"
            )
        for required in ("model_name", "model_family", "model_revision"):
            if not profile.get(required):
                raise ConfigDocumentError(
                    f"{location}: translation.models[{active!r}] requires "
                    f"{required}"
                )
        if "model_cache_dir" in profile or "cache_dir" in profile:
            raise ConfigDocumentError(
                f"{location}: model profiles must not configure cache roots; "
                f"use server.model_cache_dir"
            )
        return active, profile

    # Legacy flat schema (compatibility adapter).
    backend = trans_raw.get("backend", "current")
    if backend not in _SUPPORTED_BACKENDS:
        raise ConfigDocumentError(
            f"{location}: translation.backend must be one of "
            f"{sorted(_SUPPORTED_BACKENDS)}"
        )
    profile = {
        "backend": backend,
        "model_name": trans_raw.get(
            "model_name", "facebook/nllb-200-distilled-600M"
        ),
        "model_family": trans_raw.get("model_family", "nllb"),
        "model_revision": trans_raw.get("model_revision", "main"),
    }
    return backend, profile


def validate_translation_section_keys(
    trans_raw: dict[str, Any],
    path: Path | str | None = None,
) -> None:
    """Reject unknown translation keys beyond shared + profile registry."""
    location = str(path) if path is not None else "<config>"
    allowed = set(_TRANSLATION_SHARED_KEYS)
    if trans_raw.get("models") is not None:
        allowed.update({"models", "active_model"})
    else:
        allowed.update(_LEGACY_MODEL_KEYS)
    unknown = sorted(set(trans_raw) - allowed)
    if unknown:
        raise ConfigDocumentError(
            f"{location}: unknown translation key(s): {', '.join(unknown)}"
        )
