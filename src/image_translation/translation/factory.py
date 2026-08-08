"""Translation factory — creates Translator instances from configuration."""

from __future__ import annotations

import logging

from .base import Translator
from .config import TranslationConfig

logger = logging.getLogger(__name__)


def create_translator(config: TranslationConfig) -> Translator:
    """Create a Translator instance based on configuration.

    Args:
        config: Validated TranslationConfig.

    Returns:
        A Translator implementation.

    Raises:
        ValueError: If the configured engine is unknown.
    """
    model_name = config.model_name.lower()

    if "m2m100" in model_name:
        from .m2m100_translator import M2M100Translator
        logger.info("Creating M2M100 translator for model: %s", config.model_name)
        return M2M100Translator(config)

    raise ValueError(
        f"Unknown translation engine for model '{config.model_name}'. "
        f"Supported engines: m2m100."
    )
