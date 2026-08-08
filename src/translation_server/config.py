"""Translation server configuration — FastAPI host settings + shared translation config."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from image_translation.translation.config import TranslationConfig


@dataclass
class ServerConfig:
    """Translation server host configuration."""
    host: str = "127.0.0.1"
    port: int = 8091
    workers: int = 1
    log_level: str = "info"


@dataclass
class TranslationServerConfig:
    """Root config for the standalone translation server."""
    server: ServerConfig = field(default_factory=ServerConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)


def load_server_config(path: Optional[Path] = None) -> TranslationServerConfig:
    """Load server configuration from a JSON file.

    Args:
        path: Path to config JSON. Uses built-in defaults if None.

    Returns:
        Validated TranslationServerConfig.

    Raises:
        FileNotFoundError: If path is provided but does not exist.
        ValueError: If JSON is malformed or validation fails.
    """
    if path is None:
        return TranslationServerConfig()

    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Server config not found: {p}")

    raw = json.loads(p.read_text(encoding="utf-8"))

    server_raw = raw.get("server", {})
    server = ServerConfig(
        host=server_raw.get("host", "127.0.0.1"),
        port=server_raw.get("port", 8091),
        workers=server_raw.get("workers", 1),
        log_level=server_raw.get("log_level", "info"),
    )

    trans_raw = raw.get("translation", {})
    gen_raw = trans_raw.get("generation", {})

    from image_translation.translation.config import GenerationConfig
    generation = GenerationConfig(
        max_new_tokens=gen_raw.get("max_new_tokens", 256),
        num_beams=gen_raw.get("num_beams", 4),
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

    return TranslationServerConfig(server=server, translation=translation)
