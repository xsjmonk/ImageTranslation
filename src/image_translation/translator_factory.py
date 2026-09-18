"""Factory for image-pipeline translators — bridges AppConfig to GPU/server backends."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .translation.base import Translator
from .translation.config import TranslationStyle
from .translation.factory import create_translator
from .translation.server_translator import ServerTranslator
from .translation.translator import NoopTranslator

if TYPE_CHECKING:
    from .config.models import TranslationConfig as AppTranslationConfig

logger = logging.getLogger(__name__)


def create_image_pipeline_translator(
    config: "AppTranslationConfig",
    repo_root: Optional[Path] = None,
) -> Translator:
    """Create a translator for the image localization pipeline."""
    if not config.enabled:
        logger.info("Translation disabled — using noop translator")
        return NoopTranslator()

    engine = (config.engine or "gpu").lower()
    if engine == "noop":
        return NoopTranslator()

    style = TranslationStyle(config.style) if config.style else TranslationStyle.PHRASE
    source_lang = _short_lang(config.source_language)
    target_lang = _short_lang(config.target_language)

    if engine == "server":
        return ServerTranslator(
            server_url=config.server_url,
            source_language=source_lang,
            target_language=target_lang,
            style=style,
            timeout_seconds=config.server_timeout_seconds,
        )

    if engine == "gpu":
        gpu_config = _load_gpu_translation_config(config, repo_root)
        return create_translator(gpu_config)

    raise ValueError(
        f"translation.engine must be one of noop, gpu, server; got '{config.engine}'"
    )


def _short_lang(code: str) -> str:
    return code.replace("-", "_").split("_")[0]


def _load_gpu_translation_config(config: "AppTranslationConfig", repo_root: Optional[Path]):
    from translation_server.config import load_server_config

    if config.gpu_config_path:
        path = Path(config.gpu_config_path)
    else:
        root = repo_root or _default_repo_root()
        path = root / "translation-server.config.json"

    if not path.exists():
        raise FileNotFoundError(
            "GPU translation config not found. Set translation.gpu_config_path "
            f"or place translation-server.config.json at: {path}"
        )

    server_config = load_server_config(path)
    logger.info(
        "Loaded GPU translator from %s (backend=%s)",
        path,
        server_config.translation.backend,
    )
    return server_config.translation


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
