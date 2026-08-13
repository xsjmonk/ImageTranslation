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
        """Check every placeholder appears exactly once, in tag-relative order.

        ``expected_order`` is the placeholder order in the SOURCE text (a
        ProtectionMap reserves tokens in processing order, which can differ
        from source order when an identifier sits inside an English span).
        When omitted, reservation order is used (backwards compatible).

        Order contract (documented, per spec: "validate placeholder order
        relative to tags"):
        - tag placeholders must appear in source order;
        - every non-tag placeholder must stay inside the same tag interval
          (the number of tags before it) as in the source — the model may
          reorder protected spans WITHIN a tag interval (its own rendering),
          but never move them across tag boundaries.

        Returns {"ok": bool, "issues": [str,...], "order_ok": bool}.
        """
        issues: List[str] = []
        expected = self.tokens
        expected_order = expected_order if expected_order is not None else expected

        found_order: List[str] = []
        seen: set = set()
        # Scan for placeholder tokens in first-occurrence order
        for match in re.finditer(re.escape(self.prefix) + r"[A-Z]\d{4}_", output):
            tok = match.group(0)
            if tok in self._spans and tok not in seen:
                found_order.append(tok)
                seen.add(tok)

        for token in expected:
            count = output.count(token)
            if count == 0:
                issues.append(f"placeholder missing: {token}")
            elif count > 1:
                issues.append(f"placeholder duplicated ({count}x): {token}")

        # Tag-relative order checks
        def tag_sequence(tokens: List[str]) -> List[str]:
            return [t for t in tokens
                    if self._spans[t].kind in ("tag_start", "tag_end")]

        def tag_intervals(tokens: List[str]) -> Dict[str, int]:
            intervals: Dict[str, int] = {}
            tags_before = 0
            for t in tokens:
                if self._spans[t].kind in ("tag_start", "tag_end"):
                    tags_before += 1
                else:
                    intervals[t] = tags_before
            return intervals

        order_ok = True
        if tag_sequence(found_order) != tag_sequence(expected_order):
            order_ok = False
            issues.append(
                f"tag placeholder order changed: expected "
                f"{tag_sequence(expected_order)}, got {tag_sequence(found_order)}"
            )
        found_intervals = tag_intervals(found_order)
        expected_intervals = tag_intervals(expected_order)
        moved = [
            t for t in expected_intervals
            if found_intervals.get(t) != expected_intervals[t]
        ]
        if moved:
            order_ok = False
            issues.append(
                f"protected spans moved across tag boundaries: {moved}"
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


def assert_prefix_safe(text: str, prefix: str = DEFAULT_PREFIX) -> None:
    """Fail closed if the placeholder prefix could collide with source text."""
    if prefix in text:
        raise StructuredTranslationError(
            f"placeholder prefix {prefix!r} collides with source text; "
            f"cannot protect this document safely"
        )
