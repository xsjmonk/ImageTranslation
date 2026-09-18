"""Default translator – a no-op placeholder that preserves interface contract.

Replace with a real cloud translation provider (Google, DeepL, etc.) later.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from typing import Sequence

from ..models.text_region import TextRegion
from .base import Translator
from .models import TranslationResult, TranslationRuntimeInfo

logger = logging.getLogger(__name__)


class NoopTranslator(Translator):
    """Placeholder translator that returns source text as translation.

    This satisfies the Translator interface without depending on any
    external translation API. Replace with a real implementation.
    """

    @property
    def name(self) -> str:
        return "noop"

    @property
    def runtime_info(self) -> TranslationRuntimeInfo:
        return TranslationRuntimeInfo(
            backend="noop",
            model_name="noop",
            device="none",
            precision="none",
            dtype="none",
            source_language="zh",
            target_language="en",
        )

    def translate_text(
        self,
        text: str,
        source_lang: str = "zh",
        target_lang: str = "en",
        style=None,
    ) -> TranslationResult:
        bracketed = f"[{text}]"
        return TranslationResult(
            source_text=text,
            translated_text=bracketed,
            compact_text=bracketed,
            literal_text=bracketed,
            source_language=source_lang,
            target_language=target_lang,
        )

    def translate_batch_texts(
        self,
        texts: Sequence[str],
        source_lang: str = "zh",
        target_lang: str = "en",
        max_new_tokens: int | None = None,
        style=None,
    ) -> List[TranslationResult]:
        return [
            self.translate_text(text, source_lang, target_lang, style) for text in texts
        ]

    def translate(self, region: TextRegion, target_language: str) -> Dict[str, Any]:
        return {
            "source_text": region.source_text,
            "translated_text": f"[{region.source_text}]",
            "compact_text": region.source_text,
            "literal_text": region.source_text,
            "target_language": target_language,
        }

    def translate_batch(
        self, regions: List[TextRegion], target_language: str
    ) -> List[Dict[str, Any]]:
        return [self.translate(r, target_language) for r in regions]
