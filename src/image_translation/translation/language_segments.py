"""Deterministic Chinese/English classification and protected-span extraction.

Language policy (documented):
- pure English text (no CJK): preserved EXACTLY — replaced with protected
  placeholders before inference, restored from the original after inference
  (never model output);
- pure Chinese text: translated;
- mixed Chinese/English text: split into CJK spans (translated) and
  English/identifier spans (protected placeholders), so English embedded
  inside mixed content is preserved exactly;
- URLs, emails, product codes, version strings, measurements, and known
  brand/interface identifiers are protected as placeholders;
- a whole document is never classified by one global detector — every text
  span is classified independently.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Tuple

from .html_protection import ProtectionMap


class LanguageKind(str, Enum):
    ENGLISH = "english"
    CHINESE = "chinese"
    MIXED = "mixed"


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# A maximal CJK span: hanzi plus CJK punctuation (，。！？；：、""''《》【】（）…)
_CJK_SPAN_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f\uff00-\uffef]+"
)


def split_mixed_spans(text: str):
    """Split text into maximal (is_chinese, span) runs.

    A span is ``chinese`` when it contains hanzi or CJK punctuation;
    everything else (letters, digits, spaces, ASCII punctuation) is a
    non-Chinese span that will be PROTECTED (preserved exactly), never
    translated. Whitespace between CJK spans stays inside the CJK span.
    """
    spans = []
    pos = 0
    for m in _CJK_SPAN_RE.finditer(text):
        if m.start() > pos:
            spans.append((False, text[pos:m.start()]))
        spans.append((True, m.group(0)))
        pos = m.end()
    if pos < len(text):
        spans.append((False, text[pos:]))
    return spans

# Protected identifiers (applied inside mixed segments)
PROTECTED_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("url", re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)),
    ("www", re.compile(r"www\.[^\s<>\"']+", re.IGNORECASE)),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("product_code", re.compile(r"\b[A-Z]{2,6}-\d{2,}[A-Z0-9-]*\b")),
    ("interface", re.compile(r"\bUSB-C\b|\bHDMI\b|\bBluetooth\b|\bWi-Fi\b", re.IGNORECASE)),
    ("version", re.compile(r"\bWindows\s*\d+(?:\.\d+)?\b", re.IGNORECASE)),
    ("version_num", re.compile(r"\b\d+\.\d+(?:\.\d+)+\b")),
    ("measurement", re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|kg|g|ml|L|W|V|A|mAh|Wh|GHz|MHz|%)\b",
        re.IGNORECASE,
    )),
    ("model_no", re.compile(r"\b[A-Z]{2,4}[- ]?\d{3,}[A-Z0-9-]*\b")),
]


def classify(text: str) -> LanguageKind:
    """Classify a text span: english / chinese / mixed."""
    if not text:
        return LanguageKind.ENGLISH
    has_cjk = bool(_CJK_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    if has_cjk and has_latin:
        return LanguageKind.MIXED
    if has_cjk:
        return LanguageKind.CHINESE
    return LanguageKind.ENGLISH


def find_protected_spans(text: str):
    """Yield (start, end, kind, content) for every protected identifier.

    Overlapping matches are resolved first-match-wins, in document order.
    """
    matches = []
    for kind, pattern in PROTECTED_PATTERNS:
        for m in pattern.finditer(text):
            matches.append((m.start(), m.end(), kind, m.group(0)))

    # Merge overlaps keeping the earliest start (then longest)
    matches.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    merged = []
    last_end = -1
    for start, end, kind, content in matches:
        if start < last_end:
            continue
        merged.append((start, end, kind, content))
        last_end = end
    return merged


def protect_identifiers(text: str, pmap: ProtectionMap) -> str:
    """Replace protected identifiers with placeholders (for mixed segments).

    Args:
        text: The mixed-language text.
        pmap: ProtectionMap to reserve tokens on.

    Returns:
        Text with identifiers replaced by placeholders.
    """
    spans = find_protected_spans(text)
    if not spans:
        return text

    pieces = []
    cursor = 0
    for start, end, kind, content in spans:
        pieces.append(text[cursor:start])
        token = pmap.reserve(content, kind="span")
        pieces.append(token)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def is_english_only(text: str) -> bool:
    return classify(text) == LanguageKind.ENGLISH
