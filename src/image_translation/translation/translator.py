"""Default translator – a no-op placeholder that preserves interface contract.

Replace with a real cloud translation provider (Google, DeepL, etc.) later.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..models.text_region import TextRegion
from .base import Translator

logger = logging.getLogger(__name__)


class NoopTranslator(Translator):
    """Placeholder translator that returns source text as translation.

    This satisfies the Translator interface without depending on any
    external translation API. Replace with a real implementation.
    """

    @property
    def name(self) -> str:
        return "noop"

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
