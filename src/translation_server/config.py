"""Translation server configuration — FastAPI host settings + shared translation config.

Validation happens at load time:
- 1 <= port <= 65535
- workers >= 1; workers == 1 required when translation.device == "cuda"
- cuda_device >= 0
- batch_size >= 1
- max_input_characters >= 1
- log_level in supported Uvicorn levels
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from image_translation.translation.config import GenerationConfig, TranslationConfig

_VALID_LOG_LEVELS = {"critical", "error", "warning", "info", "debug", "trace"}


@dataclass
class ServerConfig:
    """Translation server host configuration."""
    host: str = "127.0.0.1"
    port: int = 8091
    workers: int = 1
    log_level: str = "info"


@dataclass
class RuntimeConfig:
    """Runtime behavior configuration."""
    warmup_on_start: bool = True


@dataclass
class TranslationServerConfig:
    """Root config for the standalone translation server."""
    server: ServerConfig = field(default_factory=ServerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)


def _validate(config: TranslationServerConfig) -> None:
    """Validate the assembled configuration; raise ValueError on problems."""
    server = config.server
    if not (1 <= server.port <= 65535):
        raise ValueError(f"server.port must be between 1 and 65535, got {server.port}")
    if server.workers < 1:
        raise ValueError(f"server.workers must be >= 1, got {server.workers}")
    if server.log_level.lower() not in _VALID_LOG_LEVELS:
        raise ValueError(
            f"server.log_level must be one of {sorted(_VALID_LOG_LEVELS)}, "
            f"got '{server.log_level}'"
        )

    trans = config.translation
    if trans.cuda_device < 0:
        raise ValueError(f"translation.cuda_device must be >= 0, got {trans.cuda_device}")
    if trans.batch_size < 1:
        raise ValueError(f"translation.batch_size must be >= 1, got {trans.batch_size}")
    if trans.max_input_characters < 1:
        raise ValueError(
            f"translation.max_input_characters must be >= 1, "
            f"got {trans.max_input_characters}"
        )

    # CUDA mode must use a single worker so only one 418M model is loaded into VRAM
    if trans.device == "cuda" and server.workers != 1:
        raise ValueError(
            f"server.workers must be 1 when translation.device == 'cuda' "
            f"(got workers={server.workers}); multiple workers would each load "
            f"their own GPU model."
        )


def _find_repo_root() -> Path:
    """Return the repository root: src/translation_server -> src -> repo root."""
    return Path(__file__).resolve().parent.parent.parent


def _resolve_config_path(explicit_path: Optional[Path]) -> Optional[Path]:
    """Determine which config file to use.

    Resolution:
    1. Explicit -c/--config path: must exist, else raise.
    2. Otherwise: <repo-root>/translation-server.config.json if it exists.
    3. Otherwise: None (built-in defaults).
    """
    if explicit_path is not None:
        resolved = Path(explicit_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Server config not found: {resolved}")
        return resolved

    default_path = _find_repo_root() / "translation-server.config.json"
    if default_path.exists():
        return default_path
    return None


def load_server_config(path: Optional[Path] = None) -> TranslationServerConfig:
    """Load server configuration.

    Args:
        path: Explicit config path. If None, tries <repo-root>/
            translation-server.config.json, then built-in defaults.

    Returns:
        Validated TranslationServerConfig.

    Raises:
        FileNotFoundError: If an explicit path is given but does not exist.
        ValueError: If JSON is malformed or validation fails.
    """
    resolved = _resolve_config_path(path)

    if resolved is None:
        config = TranslationServerConfig()
        _validate(config)
        return config

    raw = json.loads(resolved.read_text(encoding="utf-8"))

    server_raw = raw.get("server", {})
    server = ServerConfig(
        host=server_raw.get("host", "127.0.0.1"),
        port=server_raw.get("port", 8091),
        workers=server_raw.get("workers", 1),
        log_level=server_raw.get("log_level", "info"),
    )

    runtime_raw = raw.get("runtime", {})
    runtime = RuntimeConfig(
        warmup_on_start=runtime_raw.get("warmup_on_start", True),
    )

    trans_raw = raw.get("translation", {})
    gen_raw = trans_raw.get("generation", {})
    generation = GenerationConfig(
        max_new_tokens=gen_raw.get("max_new_tokens", 256),
        num_beams=gen_raw.get("num_beams", 1),
        length_penalty=gen_raw.get("length_penalty", 1.0),
        early_stopping=gen_raw.get("early_stopping", True),
    )

    translation = TranslationConfig(
        model_name=trans_raw.get("model_name", "facebook/m2m100_418M"),
        source_language=trans_raw.get("source_language", "zh"),
        target_language=trans_raw.get("target_language", "en"),
        device=trans_raw.get("device", "cuda"),
        cuda_device=trans_raw.get("cuda_device", 0),
        allow_cpu_fallback=trans_raw.get("allow_cpu_fallback", False),
        precision=trans_raw.get("precision", "auto"),
        batch_size=trans_raw.get("batch_size", 8),
        max_input_characters=trans_raw.get("max_input_characters", 4000),
        generation=generation,
        model_cache_dir=trans_raw.get("model_cache_dir"),
    )

    config = TranslationServerConfig(server=server, runtime=runtime, translation=translation)
    _validate(config)
    return config
