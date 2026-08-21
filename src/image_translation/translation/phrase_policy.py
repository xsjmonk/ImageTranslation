"""Pure phrase-style quality validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from .html_protection import PLACEHOLDER_TOKEN_RE

_SCAFFOLDING_RE = re.compile(
    r"^\s*(?:this\s+(?:is|product)|it\s+is)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class PhraseValidationResult:
    accepted: bool
    reasons: tuple[str, ...] = ()
    source_tokens: int = 0
    output_tokens: int = 0
    expansion_ratio: float = 0.0


def validate_phrase_output(
    source: str,
    output: str,
    *,
    max_expansion_ratio: float,
) -> PhraseValidationResult:
    """Check model-produced phrase text without rewriting it."""
    source_clean = PLACEHOLDER_TOKEN_RE.sub(" ", source)
    output_clean = PLACEHOLDER_TOKEN_RE.sub(" ", output)
    source_tokens = content_token_count(source_clean)
    output_tokens = content_token_count(output_clean)
    source_tokens = max(1, source_tokens)
    ratio = output_tokens / source_tokens
    reasons = []
    if ratio > max_expansion_ratio:
        reasons.append("output_expansion_exceeded")
    if _SCAFFOLDING_RE.search(output_clean):
        reasons.append("sentence_scaffolding_detected")
    return PhraseValidationResult(
        accepted=not reasons,
        reasons=tuple(reasons),
        source_tokens=source_tokens,
        output_tokens=output_tokens,
        expansion_ratio=ratio,
    )


def content_token_count(text: str) -> int:
    """Count CJK ideographs individually and Latin/numeric runs as units."""
    count = 0
    in_latin = False
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            count += 1
            in_latin = False
        elif char.isascii() and (char.isalnum() or char in "'-_"):
            if not in_latin:
                count += 1
            in_latin = True
        else:
            in_latin = False
    return count
