"""Protected spans, placeholder tokens, and placeholder validation.

Placeholders replace inline tags and protected identifiers inside text that
is sent to the translation model, so the model can never rewrite them as
HTML. After generation, placeholders are validated (exactly-once, source
order) and restored from the ORIGINAL node/spans — never from model output.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .exceptions import StructuredTranslationError

DEFAULT_PREFIX = "__ITRANSLATE_"

# Matches ANY placeholder-like token the project can produce: the default
# prefix (__ITRANSLATE_) and the stricter retry prefixes (__IT + 4 random
# alnum + _), each followed by <kind letter><4 digits>_. Used to reject
# model-invented placeholder-like tokens that are not registered.
_PLACEHOLDER_TOKEN_RE = re.compile(r"__IT[A-Za-z0-9]{0,16}_[A-Z]\d{4}_")
PLACEHOLDER_TOKEN_RE = _PLACEHOLDER_TOKEN_RE


def is_placeholder_token(value: str) -> bool:
    """Return whether value is one complete project placeholder token."""
    return bool(PLACEHOLDER_TOKEN_RE.fullmatch(value))


class ProtectedSpan:
    """A span of source content protected from translation."""

    __slots__ = ("token", "content", "kind")

    def __init__(self, token: str, content: str, kind: str = "span") -> None:
        self.token = token
        self.content = content
        self.kind = kind  # span | tag_start | tag_end

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ProtectedSpan({self.token!r}, {self.content!r}, {self.kind!r})"


class ProtectionMap:
    """token -> ProtectedSpan, with build/validate/restore operations."""

    def __init__(self, prefix: str = DEFAULT_PREFIX) -> None:
        self.prefix = prefix
        self._spans: Dict[str, ProtectedSpan] = {}
        self._counter = 0

    # -- building ------------------------------------------------------

    def reserve(self, content: str, kind: str = "span") -> str:
        """Create a new placeholder token for protected content."""
        self._counter += 1
        token = f"{self.prefix}{kind[:1].upper()}{self._counter:04d}_"
        self._spans[token] = ProtectedSpan(token, content, kind)
        return token

    @property
    def tokens(self) -> List[str]:
        return [s.token for s in self._spans.values()]

    def span(self, token: str) -> Optional[ProtectedSpan]:
        return self._spans.get(token)

    # -- validation ----------------------------------------------------

    def validate_output(
        self, output: str, expected_order: Optional[List[str]] = None
    ) -> Dict[str, object]:
        """Check every placeholder appears exactly once, in EXACT source order.

        ``expected_order`` is the placeholder order in the SOURCE text (a
        ProtectionMap reserves tokens in processing order, which can differ
        from source order when an identifier sits inside an English span).
        When omitted, reservation order is used (backwards compatible).

        Strict contract (documented, production requirement):
        - the complete known-placeholder sequence in the model output must
          equal ``expected_order`` EXACTLY — tags, entities, bare-ampersand
          runs, English spans, identifiers/model numbers/SKUs/URLs/emails/
          versions and protected terms included. Reordering within
          a tag interval is REJECTED: protected content must stay in its
          original source slot;
        - any missing, duplicated, altered, or unknown placeholder is
          rejected;
        - any model-invented placeholder-like token matching the project's
          placeholder syntax (including alternate retry prefixes) is
          rejected unless registered in this ProtectionMap.

        Returns {"ok": bool, "issues": [str,...], "order_ok": bool}.
        """
        issues: List[str] = []
        expected = self.tokens
        expected_order = expected_order if expected_order is not None else expected

        # 1) Reject model-invented placeholder-like tokens (any prefix
        #    variant: default and retry prefixes) not registered here.
        for match in _PLACEHOLDER_TOKEN_RE.finditer(output):
            tok = match.group(0)
            if tok not in self._spans:
                issues.append(f"unknown placeholder invented: {tok}")

        # 2) Missing / duplicated / altered known placeholders. Iterate the
        #    CALL's expected sequence (a block split across segments shares
        #    one ProtectionMap, so map-wide tokens of sibling segments must
        #    not be required here).
        for token in expected_order:
            count = output.count(token)
            if count == 0:
                issues.append(f"placeholder missing: {token}")
            elif count > 1:
                issues.append(f"placeholder duplicated ({count}x): {token}")

        # 3) EXACT full-sequence order: the known-placeholder sequence in
        #    first-occurrence order must equal the source order exactly.
        found_order: List[str] = []
        seen: set = set()
        for match in re.finditer(re.escape(self.prefix) + r"[A-Z]\d{4}_", output):
            tok = match.group(0)
            if tok in self._spans and tok not in seen:
                found_order.append(tok)
                seen.add(tok)

        order_ok = found_order == expected_order
        if not order_ok:
            issues.append(
                f"placeholder sequence changed: expected {expected_order}, "
                f"got {found_order}"
            )

        return {"ok": not issues, "issues": issues, "order_ok": order_ok}

    # -- restoration ---------------------------------------------------

    def restore(self, output: str) -> str:
        """Replace every placeholder with its original protected content."""
        result = output
        for token, span in self._spans.items():
            result = result.replace(token, span.content)
        return result

    def restore_split(self, output: str):
        """Split model output at EVERY placeholder token (tags, identifiers,
        protected English), in first-occurrence order.

        Returns (pieces, protected_sequence):
          pieces: list of text pieces between consecutive placeholders
                  (len == number of placeholders + 1);
          protected_sequence: list of ProtectedSpan in the order their
                  placeholders occur in the output.
        """
        tokens = self.tokens
        if not tokens:
            return [output], []

        pattern = "|".join(re.escape(t) for t in tokens)
        pieces = re.split(pattern, output)
        protected_sequence = [
            self._spans[m.group(0)]
            for m in re.finditer(pattern, output)
        ]
        return pieces, protected_sequence


def find_unknown_placeholders(
    output: str, protection_map: ProtectionMap
) -> List[str]:
    """Return placeholder-like tokens in ``output`` not registered in the map.

    Detects model-invented tokens matching the project's placeholder syntax
    — the default prefix and the stricter retry prefixes — that are not
    registered in ``protection_map``. Used as a fail-closed guard in both
    validation and reconstruction (defense in depth).
    """
    return [
        m.group(0)
        for m in _PLACEHOLDER_TOKEN_RE.finditer(output)
        if m.group(0) not in protection_map._spans
    ]


def assert_prefix_safe(text: str, prefix: str = DEFAULT_PREFIX) -> None:
    """Fail closed if the placeholder prefix could collide with source text."""
    if prefix in text:
        raise StructuredTranslationError(
            f"placeholder prefix {prefix!r} collides with source text; "
            f"cannot protect this document safely"
        )
