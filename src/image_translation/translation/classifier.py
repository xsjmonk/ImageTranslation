"""Text action classifier – decides translate/preserve/remove/review per region."""

from __future__ import annotations

import logging
import re
from typing import List

import numpy as np

from ..config.models import ClassificationConfig, TranslationConfig
from ..models.text_region import TextAction, TextRegion

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_SERIAL_MODEL_RE = re.compile(
    r"^(型号|款号|货号)?[A-Z]{0,4}[-_]?\d{2,}[A-Z0-9\-_/]*$",
    re.IGNORECASE,
)
_BRAND_LIKE_RE = re.compile(r"^[A-Z0-9\u4e00-\u9fff]{1,8}$")
_DESCRIPTIVE_MARKERS = (
    "展示",
    "说明",
    "特点",
    "功能",
    "材质",
    "颜色",
    "尺寸",
    "正面",
    "侧面",
    "背面",
    "细节",
    "加厚",
    "升级",
)


def classify_regions(
    regions: List[TextRegion],
    config: TranslationConfig,
    classification: ClassificationConfig,
) -> List[TextRegion]:
    """Assign an action to every TextRegion based on configured rules."""
    compiled = config.compiled_patterns()
    product_patterns = classification.compiled_product_patterns()
    review_patterns = classification.compiled_review_patterns()
    force_patterns = classification.compiled_force_patterns()
    identifier_patterns = classification.compiled_identifier_patterns()

    for region in regions:
        text = region.source_text.strip()
        if not text:
            region.action = TextAction.remove
            region.action_reason = "empty_text"
            region.action_confidence = 1.0
            continue

        if _matches_any(text, force_patterns):
            region.action = TextAction.translate
            region.action_reason = "force_translate_pattern"
            region.action_confidence = 0.95
            continue

        if _matches_any(text, review_patterns):
            region.action = TextAction.review
            region.action_reason = "review_pattern"
            region.action_confidence = 0.9
            continue

        if config.preserve_already_target_language and _is_mostly_english(text):
            region.action = TextAction.preserve
            region.action_reason = "already_target_language"
            region.action_confidence = 0.9
            continue

        if _matches_preserve_terms(text, config.preserve_terms):
            region.action = TextAction.preserve
            region.action_reason = "preserve_term_match"
            region.action_confidence = 0.95
            continue

        if _matches_preserve_patterns(text, compiled):
            region.action = TextAction.preserve
            region.action_reason = "preserve_pattern_match"
            region.action_confidence = 0.95
            continue

        if _matches_identifier(text, identifier_patterns):
            region.action = TextAction.preserve
            region.action_reason = "identifier_match"
            region.action_confidence = 0.9
            continue

        if _looks_like_serial_or_model(text):
            region.action = TextAction.preserve
            region.action_reason = "serial_or_model_label"
            region.action_confidence = 0.85
            continue

        if _matches_any(text, product_patterns):
            region.action = TextAction.review
            region.action_reason = "product_embedded_pattern"
            region.action_confidence = 0.8
            continue

        if not _CJK_RE.search(text):
            region.action = TextAction.preserve
            region.action_reason = "no_cjk_detected"
            region.action_confidence = 0.8
            continue

        if _is_descriptive_callout(text):
            region.action = TextAction(config.default_action)
            region.action_reason = "descriptive_callout"
            region.action_confidence = 0.7
            continue

        if _looks_like_logo(region, text, classification):
            region.action = TextAction.review
            region.action_reason = "likely_logo_or_trademark"
            region.action_confidence = 0.75
            continue

        if _looks_like_brand_mark(text, classification):
            region.action = TextAction.review
            region.action_reason = "likely_brand_mark"
            region.action_confidence = 0.7
            continue

        region.action = TextAction.review
        region.action_reason = "uncertain_cjk_text"
        region.action_confidence = 0.55

    return regions


def _is_descriptive_callout(text: str) -> bool:
    if len(text) >= 8:
        return True
    if any(marker in text for marker in _DESCRIPTIVE_MARKERS):
        return True
    if text.endswith("色") and len(text) <= 6:
        return True
    return False


def _looks_like_serial_or_model(text: str) -> bool:
    stripped = text.strip()
    if _SERIAL_MODEL_RE.match(stripped):
        return True
    if re.fullmatch(r"^[A-Z]{1,4}\d{2,6}$", stripped):
        return True
    return False


def _looks_like_brand_mark(text: str, classification: ClassificationConfig) -> bool:
    stripped = text.strip()
    if len(stripped) > classification.logo_max_chars:
        return False
    if " " in stripped:
        return False
    if _BRAND_LIKE_RE.fullmatch(stripped) and _CJK_RE.search(stripped):
        return True
    return False


def _is_mostly_english(text: str) -> bool:
    if not text:
        return False
    ascii_count = sum(
        1 for c in text if c.isascii() and (c.isalpha() or c.isdigit() or c.isspace())
    )
    return (ascii_count / len(text)) > 0.8


def _matches_preserve_terms(text: str, terms: List[str]) -> bool:
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in terms)


def _matches_preserve_patterns(text: str, patterns: List[re.Pattern]) -> bool:
    return any(pattern.fullmatch(text.strip()) for pattern in patterns)


def _matches_any(text: str, patterns: List[re.Pattern]) -> bool:
    return any(pattern.search(text.strip()) for pattern in patterns)


def _matches_identifier(text: str, patterns: List[re.Pattern]) -> bool:
    stripped = text.strip()
    if _URL_RE.match(stripped):
        return True
    return any(pattern.fullmatch(stripped) for pattern in patterns)


def _looks_like_logo(
    region: TextRegion,
    text: str,
    classification: ClassificationConfig,
) -> bool:
    """Conservative heuristic: short, wide regions may be logos or trademarks."""
    if len(text) > classification.logo_max_chars:
        return False
    poly = np.array(region.polygon, dtype=np.float32)
    xs = poly[:, 0]
    ys = poly[:, 1]
    width = float(np.max(xs) - np.min(xs))
    height = float(np.max(ys) - np.min(ys))
    if height <= 0:
        return False
    aspect = width / height
    if aspect >= classification.logo_aspect_ratio_min:
        return True
    if len(text) <= 4 and not _is_descriptive_callout(text):
        return True
    return False
