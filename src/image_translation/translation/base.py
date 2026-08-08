"""Translation base interface – replaceable translator protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

from ..models.text_region import TextRegion
from .models import TranslationResult


class Translator(ABC):
    """Abstract translator interface.

    Implementations translate text from source to target language.
    Clients depend on this abstraction, not concrete model classes.
    """

    # ---- Core string-based API (preferred for new code) ----

    @abstractmethod
    def translate_text(
        self, text: str, source_lang: str = "zh", target_lang: str = "en"
    ) -> TranslationResult:
        """Translate a single string.

        Args:
            text: Source text to translate.
            source_lang: Source language code (e.g. 'zh').
            target_lang: Target language code (e.g. 'en').

        Returns:
            TranslationResult with translated_text populated.
        """
        ...

    @abstractmethod
    def translate_batch_texts(
        self, texts: Sequence[str], source_lang: str = "zh", target_lang: str = "en"
    ) -> List[TranslationResult]:
        """Translate multiple strings in one batch.

        Args:
            texts: Source texts.
            source_lang: Source language code.
            target_lang: Target language code.

        Returns:
            List of TranslationResults, same order as input.
        """
        ...

    # ---- TextRegion-based API (backward-compatible with ImageTranslation pipeline) ----

    def translate(
        self, region: TextRegion, target_language: str = "en"
    ) -> Dict[str, Any]:
        """Translate a TextRegion (compatibility shim).

        The default implementation delegates to translate_text().
        Override only if TextRegion-specific handling is needed.
        """
        result = self.translate_text(
            text=region.source_text,
            source_lang=region.language or "zh",
            target_lang=target_language.replace("-", "_").split("_")[0],
        )
        return {
            "source_text": result.source_text,
            "translated_text": result.translated_text,
            "compact_text": result.compact_text,
            "literal_text": result.literal_text,
            "target_language": result.target_language,
        }

    def translate_batch(
        self, regions: List[TextRegion], target_language: str = "en"
    ) -> List[Dict[str, Any]]:
        """Translate multiple TextRegions."""
        texts = [r.source_text for r in regions]
        results = self.translate_batch_texts(
            texts=texts,
            source_lang="zh",
            target_lang=target_language.replace("-", "_").split("_")[0],
        )
        return [
            {
                "source_text": r.source_text,
                "translated_text": r.translated_text,
                "compact_text": r.compact_text,
                "literal_text": r.literal_text,
                "target_language": r.target_language,
            }
            for r in results
        ]

    # ---- Metadata ----

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable translator name."""
        ...

    @property
    def runtime_info(self) -> "TranslationRuntimeInfo":
        """Return runtime metadata about the loaded engine."""
        raise NotImplementedError
