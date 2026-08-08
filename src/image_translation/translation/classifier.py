"""Text action classifier – decides translate/preserve/remove/review per region."""

from __future__ import annotations

import logging
import re
from typing import List

from ..config import TranslationConfig
from ..models.text_region import TextAction, TextRegion

logger = logging.getLogger(__name__)


def classify_regions(
    regions: List[TextRegion],
    config: TranslationConfig,
) -> List[TextRegion]:
    """Assign an action to every TextRegion based on configured rules.

    Classification logic:
    1. If preserve_already_target_language: detect English text → preserve.
    2. Match against preserve_terms (case-insensitive substring).
    3. Match against preserve_patterns (full-text regex).
    4. Default action from config (usually 'translate').
    5. Uncertain cases → 'review'.

    Args:
        regions: OCR results to classify.
        config: Translation configuration.

    Returns:
        The same regions with action and action_reason set.
    """
    compiled = config.compiled_patterns()

    for region in regions:
        text = region.source_text.strip()
        if not text:
            region.action = TextAction.remove
            region.action_reason = "empty_text"
            continue

        # Check if text is already in target language (basic heuristic)
        if config.preserve_already_target_language and _is_mostly_english(text):
            region.action = TextAction.preserve
            region.action_reason = "already_target_language"
            continue

        # Check preserve terms
        if _matches_preserve_terms(text, config.preserve_terms):
            region.action = TextAction.preserve
            region.action_reason = "preserve_term_match"
            continue

        # Check preserve patterns
        if _matches_preserve_patterns(text, compiled):
            region.action = TextAction.preserve
            region.action_reason = "preserve_pattern_match"
            continue

        # Default action
        region.action = TextAction(config.default_action)
        region.action_reason = "default_action"

    return regions


def _is_mostly_english(text: str) -> bool:
    """Heuristic: if >80% of characters are ASCII letters/digits/spaces, consider it English."""
    if not text:
        return False
    ascii_count = sum(1 for c in text if c.isascii() and (c.isalpha() or c.isdigit() or c.isspace()))
    return (ascii_count / len(text)) > 0.8


def _matches_preserve_terms(text: str, terms: List[str]) -> bool:
    """Case-insensitive substring match against preserve terms."""
    text_lower = text.lower()
    for term in terms:
        if term.lower() in text_lower:
            return True
    return False


def _matches_preserve_patterns(text: str, patterns: List[re.Pattern]) -> bool:
    """Full-string regex match against preserve patterns."""
    for pattern in patterns:
        if pattern.fullmatch(text.strip()):
            return True
    return False
