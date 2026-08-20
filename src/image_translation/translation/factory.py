"""Translation factory — creates Translator instances from configuration."""

from __future__ import annotations

from .base import Translator
from .config import TranslationConfig


def create_translator(config: TranslationConfig) -> Translator:
    """Create a Translator instance based on configuration.

    Args:
        config: Validated TranslationConfig.

    Returns:
        A Translator implementation.

    Raises:
        ValueError: If the configured engine is unknown.
    """
    from .seq2seq_translator import Seq2SeqTranslator

    return Seq2SeqTranslator(config)
