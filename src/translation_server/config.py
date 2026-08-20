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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from image_translation.translation.config import (
    GenerationConfig,
    QualityConfig,
    StructuredConfig,
    TranslationConfig,
)

_VALID_LOG_LEVELS = {"critical", "error", "warning", "info", "debug", "trace"}


@dataclass
class ServerConfig:
    """Translation server host configuration."""
    host: str = "127.0.0.1"
    port: int = 8091
    workers: int = 1
    log_level: str = "info"
    model_cache_dir: str = r"D:\Caches"


@dataclass
class RuntimeConfig:
    """Runtime behavior configuration."""
    warmup_on_start: bool = True


@dataclass
class TranslationServerConfig:
    """Root config for the standalone translation server."""
    server: ServerConfig = field(default_factory=ServerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    structured: StructuredConfig = field(default_factory=StructuredConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)

    def __post_init__(self) -> None:
        """Inject the server-owned cache root into shared translation config."""
        if self.translation.model_cache_dir is None:
            self.translation.model_cache_dir = self.server.model_cache_dir
        elif self.translation.model_cache_dir != self.server.model_cache_dir:
            raise ValueError(
                "translation.model_cache_dir is a legacy input; configure "
                "server.model_cache_dir so the server owns the cache root"
            )


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

    translation = config.translation
    cache_path = Path(server.model_cache_dir)
    if not cache_path.exists() or not cache_path.is_dir():
        raise ValueError(
            f"server.model_cache_dir must be an existing directory: {cache_path}"
        )
    if not os.access(cache_path, os.W_OK):
        raise ValueError(
            f"server.model_cache_dir is not writable: {cache_path}"
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


def _resolve_cache_dir(raw: object, config_file: Optional[Path]) -> str:
    """Resolve the configured model cache root.

    - None/empty -> the explicit production default D:\Caches.
    - Environment variables are expanded (os.path.expandvars — safe, no
      shell evaluation).
    - Relative paths resolve against the directory containing the selected
      config file (never the unpredictable current working directory).
    - The result is normalized to an absolute path.
    """
    if raw is None:
        return r"D:\Caches"
    value = str(raw).strip()
    if not value:
        return r"D:\Caches"
    expanded = os.path.expandvars(value)
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        base = config_file.parent if config_file is not None else Path.cwd()
        path = base / path
    return str(path.resolve())


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
        model_cache_dir=_resolve_cache_dir(
            server_raw.get(
                "model_cache_dir",
                raw.get("translation", {}).get(
                    "model_cache_dir",
                    raw.get("translation", {}).get("cache_dir"),
                ),
            ),
            resolved,
        ),
    )

    runtime_raw = raw.get("runtime", {})
    runtime = RuntimeConfig(
        warmup_on_start=runtime_raw.get("warmup_on_start", True),
    )

    structured_raw = raw.get("structured", {})
    structured = StructuredConfig(
        enabled=structured_raw.get("enabled", True),
        max_chapter_characters=structured_raw.get("max_chapter_characters", 100_000),
        max_segment_tokens=structured_raw.get("max_segment_tokens", 450),
        max_target_tokens=structured_raw.get("max_target_tokens", 400),
        context_window_tokens=structured_raw.get("context_window_tokens", 0),
        preserve_patterns=tuple(
            structured_raw.get("preserve_patterns", ())
        ),
        translatable_attributes=tuple(
            structured_raw.get("translatable_attributes", ())
        ),
        excluded_tags=tuple(structured_raw.get("excluded_tags", ("script", "style", "code", "pre"))),
        excluded_classes=tuple(structured_raw.get("excluded_classes", ("notranslate",))),
        segment_warning_seconds=structured_raw.get(
            "segment_warning_seconds",
            structured_raw.get("max_segment_seconds", 60.0),
        ),
        max_total_seconds=structured_raw.get("max_total_seconds", 600.0),
        max_retries_per_segment=structured_raw.get("max_retries_per_segment", 1),
        batch_size=structured_raw.get("batch_size", 4),
        concurrency=structured_raw.get("concurrency", 1),
    )

    trans_raw = raw.get("translation", {})
    gen_raw = trans_raw.get("generation", {})
    generation = GenerationConfig(
        max_new_tokens=gen_raw.get("max_new_tokens", 512),
        min_new_tokens=gen_raw.get("min_new_tokens", 1),
        target_token_multiplier=gen_raw.get("target_token_multiplier", 2.5),
        short_text_max_new_tokens=gen_raw.get(
            "short_text_max_new_tokens", 64
        ),
        num_beams=gen_raw.get("num_beams", 4),
        do_sample=gen_raw.get("do_sample", False),
        no_repeat_ngram_size=gen_raw.get("no_repeat_ngram_size"),
        length_penalty=gen_raw.get("length_penalty", 1.0),
        early_stopping=gen_raw.get("early_stopping", True),
        repetition_check=gen_raw.get("repetition_check", True),
        max_repeated_token_run=gen_raw.get("max_repeated_token_run", 3),
        max_repeated_ngram_ratio=gen_raw.get(
            "max_repeated_ngram_ratio", 0.35
        ),
        retry_on_degenerate_output=gen_raw.get(
            "retry_on_degenerate_output", True
        ),
        retry_num_beams=gen_raw.get("retry_num_beams", 1),
        retry_max_new_tokens=gen_raw.get("retry_max_new_tokens", 64),
    )
    quality_raw = raw.get("quality", {})
    quality = QualityConfig(
        unknown_token_policy=quality_raw.get("unknown_token_policy", "warn"),
    )

    legacy_cache = trans_raw.get("model_cache_dir", trans_raw.get("cache_dir"))
    if (
        legacy_cache is not None
        and "model_cache_dir" in server_raw
        and _resolve_cache_dir(legacy_cache, resolved)
        != server.model_cache_dir
    ):
        raise ValueError(
            "translation.model_cache_dir conflicts with server.model_cache_dir; "
            "remove the legacy translation cache setting"
        )
    translation = TranslationConfig(
        model_name=trans_raw.get(
            "model_name", "facebook/nllb-200-distilled-600M"
        ),
        model_family=trans_raw.get("model_family", "nllb"),
        model_revision=trans_raw.get("model_revision", "main"),
        source_language=trans_raw.get("source_language", "zho_Hans"),
        target_language=trans_raw.get("target_language", "eng_Latn"),
        device=trans_raw.get("device", "cuda"),
        cuda_device=trans_raw.get("cuda_device", 0),
        allow_cpu_fallback=trans_raw.get("allow_cpu_fallback", False),
        precision=trans_raw.get("precision", "auto"),
        batch_size=trans_raw.get("batch_size", 8),
        max_input_characters=trans_raw.get("max_input_characters", 4000),
        max_input_tokens=trans_raw.get("max_input_tokens", 1024),
        commercial_use=trans_raw.get("commercial_use", False),
        generation=generation,
        quality=quality,
        model_cache_dir=server.model_cache_dir,
        allow_model_download=trans_raw.get(
            "allow_model_download",
            not trans_raw.get("offline", trans_raw.get("local_files_only", False)),
        ),
        local_files_only=trans_raw.get(
            "offline", trans_raw.get("local_files_only", False)
        ),
    )

    config = TranslationServerConfig(
        server=server,
        runtime=runtime,
        structured=structured,
        translation=translation,
    )
    _validate(config)
    return config
