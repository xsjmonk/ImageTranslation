"""Structural and token-aware chapter segmentation with an explicit run model.

Segmentation order (documented):
1. document blocks: headings, paragraphs, list items, table cells, block
   quotes, divs/sections/articles;
2. sentence boundaries within a block (。！？；.!?…);
3. clause/punctuation boundaries (，,;：:);
4. conservative hard split only as a last resort.

Protected-span translation (documented):
- every text node is split into explicit ``Run`` records: ``chinese`` spans
  (translated), ``english_protected`` / ``identifier_protected`` spans and
  ``tag`` runs (replaced with collision-resistant tokens BEFORE inference
  and restored from the ORIGINAL text afterwards — the model can never
  rewrite English or identifiers);
- a ``Run`` carries node_id, kind, original raw text, protected text,
  placeholder token, source character offsets, and slot index;
- segments record the exact source layout: ``slots`` (text between
  placeholders) interleaved with ``placeholder_order`` (every placeholder in
  source order). Reconstruction maps model-output pieces back through the
  layout, never by string splitting alone;
- coverage invariants are verified at build time: concatenating all original
  runs reproduces the block text; every source text node is covered exactly
  once (or is a whitespace-only node preserved without being sent).

Context strategy (documented): M2M100's generate() has no reliable
context-injection API, so chapter-level context is NOT implemented
(``context_window_tokens`` must stay 0). Terminology consistency across
segments is guaranteed for protected terms (identical placeholders restore
identical text); for translatable Chinese terms it depends on model
determinism — deterministic generation (fixed beams) reproduces identical
output for identical input.
"""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .html_document import ElementNode, HTMLDocument, _ENTITY_MARKER_RE
from .html_protection import ProtectionMap
from .language_segments import (
    classify,
    split_mixed_spans,
    LanguageKind,
    protect_identifiers,
    find_protected_spans,
)

# Elements whose children form their own block
BLOCK_ELEMENTS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "td", "th", "caption", "blockquote", "div",
    "section", "article", "figure", "figcaption", "dt", "dd",
    "ul", "ol", "table", "tr", "tbody", "thead", "tfoot",
    "header", "footer", "main", "aside", "nav", "form",
}

# Inline elements: tags are protected, contained text is translated
INLINE_ELEMENTS = {
    "span", "strong", "em", "b", "i", "u", "a", "sub", "sup",
    "mark", "small", "label", "abbr", "cite", "q", "time",
    "kbd", "samp", "var", "br", "img", "wbr", "ins", "del",
}

SENTENCE_BOUNDARY = "。！？；!?;…"
CLAUSE_BOUNDARY = "，,：:、"

# Run kinds (explicit mixed-span data model)
RUN_CHINESE = "chinese"                  # translated
RUN_ENGLISH = "english_protected"        # preserved exactly (placeholder)
RUN_IDENTIFIER = "identifier_protected"  # preserved exactly (placeholder)
RUN_MODEL_NUMBER = "model_number_protected"  # user-configured preserve pattern
RUN_GLOSSARY = "glossary_protected"      # glossary term -> fixed target term
RUN_ENTITY = "entity_protected"          # character reference, exact spelling
RUN_TAG = "tag"                          # inline tag placeholder
RUN_ATTRIBUTE = "attribute"              # attribute value segment
RUN_WHITESPACE = "whitespace_only"       # whitespace-only node: preserved, not sent


@dataclass
class Run:
    """One explicit span of a segment, with full source metadata.

    Attributes:
        node_id: text node id; "tag" for tag runs; "attr:<elem>:<name>" for
            attribute runs.
        kind: RUN_CHINESE / RUN_ENGLISH / RUN_IDENTIFIER / RUN_MODEL_NUMBER /
            RUN_GLOSSARY / RUN_ENTITY / RUN_TAG / RUN_ATTRIBUTE /
            RUN_WHITESPACE.
        raw: original source content (text, serialized tag, or attribute
            value) — restored verbatim for protected runs. Entity runs carry
            the sentinel marker (the serializer converts markers back to the
            exact source spelling at the end).
        protected: what appears in ``Segment.source_text`` (placeholder token
            for protected runs, raw text for chinese runs).
        placeholder: the placeholder token, or None for translated runs.
        restore_text: text restored instead of ``raw`` (glossary runs restore
            the configured TARGET term; None = restore ``raw`` verbatim).
        translate: True only for chinese/attribute runs.
        offset_start / offset_end: character offsets within the block text
            (coverage invariant).
        sequence_index: run order within the segment.
        slot_index: for translate runs, the index of the model-output piece
            it consumes (equal to the number of placeholders before it).
    """

    node_id: str
    kind: str
    raw: str
    protected: str
    placeholder: Optional[str] = None
    restore_text: Optional[str] = None
    translate: bool = False
    offset_start: int = 0
    offset_end: int = 0
    sequence_index: int = 0
    slot_index: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "raw": self.raw,
            "placeholder": self.placeholder,
            "restore_text": self.restore_text,
            "translate": self.translate,
            "offset_start": self.offset_start,
            "offset_end": self.offset_end,
            "sequence_index": self.sequence_index,
        }


@dataclass
class Segment:
    """One translatable unit to be sent to the model."""

    document_id: str
    segment_id: str
    sequence_index: int
    source_node_ids: List[str]
    source_text: str
    protected_map: ProtectionMap
    source_language: str
    target_language: str
    token_count: int
    context_before_id: Optional[str] = None
    context_after_id: Optional[str] = None
    # Explicit run metadata in source order (text runs + tag runs).
    runs: List[Run] = field(default_factory=list)
    # Every placeholder token in source order (tags + protected runs).
    placeholder_order: List[str] = field(default_factory=list)
    # Text slots between placeholders: slot[i] is the translate run whose
    # output piece sits between placeholder[i-1] and placeholder[i]
    # (None = empty slot). Derived at build time.
    slots: List[Optional[Run]] = field(default_factory=list)
    # Block identity + original block text (coverage invariant).
    block_key: str = ""
    block_text: str = ""
    # Filled after translation:
    translated_text: str = ""
    output_pieces: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "segment_id": self.segment_id,
            "sequence_index": self.sequence_index,
            "source_node_ids": list(self.source_node_ids),
            "source_text": self.source_text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "token_count": self.token_count,
            "context_before_id": self.context_before_id,
            "context_after_id": self.context_after_id,
            "runs": [r.to_dict() for r in self.runs],
        }

    def text_runs(self) -> List[Run]:
        """Translate-capable runs in slot order."""
        return [r for r in self.runs if r.translate]


class Block:
    """An ordered run of text nodes and inline tags inside one block element."""

    def __init__(self) -> None:
        self.items: List[dict] = []
        # Original block text (text raws + serialized tags, in document
        # order, whitespace-only text nodes excluded — they are preserved
        # without being sent).
        self.block_text: str = ""
        self.block_key: str = ""

    @property
    def text_node_ids(self) -> List[str]:
        return [i["node_id"] for i in self.items if i["kind"] == "text"]


def _is_excluded(element: ElementNode, excluded_tags, excluded_classes) -> bool:
    if element.tag.lower() in excluded_tags:
        return True
    attrs = dict(element.attrs)
    if attrs.get("translate", "").strip().lower() == "no":
        return True
    classes = attrs.get("class", "").split()
    if any(c in excluded_classes for c in classes):
        return True
    return False


def collect_blocks(
    doc: HTMLDocument,
    excluded_tags=("script", "style", "code", "pre"),
    excluded_classes=("notranslate",),
) -> List[Block]:
    """Collect translatable blocks in document order.

    Mixed-language grouping (documented):
    - English-only text INSIDE a block that also contains Chinese is kept as
      an item; it is protected (placeholder) before inference and restored
      exactly afterwards — it never reaches the model as free text.
    - A block whose items are ALL English-only is not translatable and is
      dropped entirely (its text is preserved verbatim in the tree).
    - Whitespace-only text nodes are recorded but preserved without being
      sent (they are not part of ``block_text``).
    - Excluded elements' subtrees are skipped entirely (never sent to the
      model; covered by the structural fingerprint + content snapshot).
    """
    blocks: List[Block] = []
    current = Block()

    def push_block() -> None:
        nonlocal current
        current = Block()
        blocks.append(current)

    def walk(container: ElementNode) -> None:
        nonlocal current
        for child in container.children:
            if child.kind == "element":
                if _is_excluded(child, excluded_tags, excluded_classes):
                    continue  # subtree fully protected
                if child.tag in BLOCK_ELEMENTS:
                    push_block()
                    walk(child)
                    push_block()
                else:
                    # Inline or unknown container: protect tags, translate text
                    current.items.append(
                        {"kind": "tag", "element": child, "side": "start"}
                    )
                    walk(child)
                    current.items.append(
                        {"kind": "tag", "element": child, "side": "end"}
                    )
            elif child.kind == "text":
                text = child.text
                if not text.strip():
                    current.items.append(
                        {"kind": "whitespace", "node_id": child.id, "text": text}
                    )
                    continue  # preserved without being sent
                lang = classify(text)
                current.items.append(
                    {"kind": "text", "node_id": child.id, "text": text, "lang": lang}
                )

    blocks.append(current)
    walk(doc.root)

    def has_translatable(b: Block) -> bool:
        return any(
            i["kind"] == "text" and i.get("lang") != LanguageKind.ENGLISH
            for i in b.items
        )

    return [b for b in blocks if has_translatable(b)]


def _serialize_start_tag(element: ElementNode) -> str:
    from .html_document import _serialize_attrs
    return f"<{element.tag}{_serialize_attrs(element.attrs)}>"


def _serialize_start_tag(element: ElementNode,
                         translatable_attrs=frozenset()) -> str:
    from .html_document import VOID_ELEMENTS
    if element.raw_start is not None and not any(
        name in translatable_attrs for name, _value in element.attrs
    ):
        return element.raw_start
    if element.tag in VOID_ELEMENTS:
        return f"<{element.tag}>"
    attrs = "".join(
        f' {name}="{html_lib.escape(value, quote=True)}"'
        for name, value in element.attrs
    )
    return f"<{element.tag}{attrs}>"


def _serialize_end_tag(element: ElementNode) -> str:
    from .html_document import VOID_ELEMENTS
    if element.raw_end is not None:
        return element.raw_end
    if element.tag in VOID_ELEMENTS:
        return ""
    return f"</{element.tag}>"


def prepare_block(block: Block, pmap: ProtectionMap,
                  translatable_attrs=frozenset()) -> List[dict]:
    """Protect inline tags; keep text RAW (identifier/English protection is
    applied per slot after splitting, so split fallback can re-protect from
    scratch). Builds ``block.block_text`` from the original content.

    Items: {"kind": "text", "node_id", "text", "lang"} (raw text),
           {"kind": "tag", "token", "content"} (placeholder token + original
           tag), or {"kind": "whitespace", "node_id", "text"}.

    Tags use their EXACT source spelling (``<br>`` vs ``<br/>``, case) when
    the source was valid and the element carries no translatable attribute;
    otherwise the canonical serialization is used.
    """
    items: List[dict] = []
    text_parts: List[str] = []
    for item in block.items:
        if item["kind"] == "text":
            items.append({
                "kind": "text",
                "node_id": item["node_id"],
                "text": item["text"],
                "lang": item.get("lang", classify(item["text"])),
            })
            text_parts.append(item["text"])
        elif item["kind"] == "whitespace":
            items.append({"kind": "whitespace", "node_id": item["node_id"],
                          "text": item["text"]})
        else:
            element = item["element"]
            if item["side"] == "start":
                content = _serialize_start_tag(element, translatable_attrs)
                kind = "tag_start"
            else:
                content = _serialize_end_tag(element)
                kind = "tag_end"
            token = pmap.reserve(content, kind=kind)
            items.append({"kind": "tag", "token": token, "content": content})
            text_parts.append(content)
    block.block_text = "".join(text_parts)
    return items


def _glossary_boundary_char(ch: str) -> bool:
    """True when ch is a word character for glossary boundary purposes:
    latin alphanumerics and underscore ONLY.

    CJK ideographs are deliberately NOT word chars: Chinese text has no
    spaces, so a term like 充电器 naturally sits adjacent to other
    ideographs ('使用充电器。'); requiring a non-CJK neighbor would make
    the glossary unusable for Chinese. The corruption risk the boundary
    policy guards against is latin word embedding ('cat' inside 'catalog'),
    which this rule prevents. Documented in GlossaryEntry.
    """
    return bool(re.match(r"[A-Za-z0-9_]", ch))


def find_glossary_spans(text: str, entries, excluded_ranges=None) -> list:
    """Find (start, end, entry) occurrences of glossary terms in text.

    Boundary policy (documented):
    - entry.exact=True: whole-occurrence only — the term must not be embedded
      in a latin word (bounded by non-latin-alphanumerics or text edges);
      CJK ideograph neighbors are accepted (Chinese has no spaces);
    - entry.exact=False: explicit opt-in — matches anywhere (may split words).

    ``excluded_ranges``: (start, end) intervals in which matches are
    dropped (protected identifiers win over glossary terms). Config
    validation forbids overlapping glossary terms, so occurrences cannot
    overlap; this is double-checked defensively (earliest span wins).
    """
    excluded_ranges = excluded_ranges or []
    spans = []
    for entry in entries:
        term = entry.source
        for m in re.finditer(re.escape(term), text):
            if any(s <= m.start() and m.end() <= e for s, e in excluded_ranges):
                continue
            if entry.exact:
                left_ok = m.start() == 0 or not _glossary_boundary_char(
                    text[m.start() - 1]
                )
                right_ok = m.end() == len(text) or not _glossary_boundary_char(
                    text[m.end()]
                )
                if not (left_ok and right_ok):
                    continue
            spans.append((m.start(), m.end(), entry))
    spans.sort(key=lambda s: (s[0], -s[1]))
    result = []
    last_end = -1
    for start, end, entry in spans:
        if start < last_end:
            continue  # overlapping occurrence: earliest span wins
        result.append((start, end, entry))
        last_end = end
    return result


def build_text_runs(
    node_id: str,
    raw: str,
    pmap: ProtectionMap,
    offset_base: int,
    glossary=None,
    preserve_patterns=(),
) -> List[Run]:
    """Turn one raw text node into explicit runs.

    Policy:
    - character-reference sentinels (from the lexical layer) become
      ``entity_protected`` runs restored to their EXACT source spelling;
    - maximal CJK regions (including surrounding whitespace) -> translated
      chinese runs (one run per region — no adjacent translate runs);
    - configured glossary terms inside chinese regions -> protected runs
      restored to the configured TARGET term (terminology memory);
    - user-configured ``preserve_patterns`` (project model formats) ->
      ``model_number_protected`` runs, exact;
    - identifiers (URLs, codes, versions, measurements, ...) -> protected;
    - remaining non-Chinese spans (ordinary English) -> protected;
    - whitespace-only spans merge into an adjacent chinese region so the
      model renders spacing naturally (never double-restored).

    All protected spans get placeholders; their ``raw`` holds the original
    text and is restored verbatim after inference (glossary runs restore
    ``restore_text`` — the configured target term; entity runs carry the
    sentinel marker, which the serializer converts to the exact spelling).
    """
    runs: List[Run] = []
    cursor = 0
    pieces = re.split(_ENTITY_MARKER_RE, raw)
    i = 0
    while i < len(pieces):
        text_piece = pieces[i]
        if text_piece:
            runs.extend(_text_runs(
                node_id, text_piece, pmap, offset_base + cursor,
                glossary, preserve_patterns,
            ))
            cursor += len(text_piece)
        if i + 2 < len(pieces):
            nonce, idx = pieces[i + 1], pieces[i + 2]
            marker = f"\x02ITENT{nonce}{idx}\x03"
            token = pmap.reserve(marker, kind="entity")
            runs.append(Run(
                node_id=node_id,
                kind=RUN_ENTITY,
                raw=marker,
                protected=token,
                placeholder=token,
                offset_start=offset_base + cursor,
                offset_end=offset_base + cursor + len(marker),
                sequence_index=len(runs),
            ))
            cursor += len(marker)
            i += 3
        else:
            i += 1
    return runs


def _text_runs(
    node_id: str,
    raw: str,
    pmap: ProtectionMap,
    offset_base: int,
    glossary=None,
    preserve_patterns=(),
) -> List[Run]:
    """Region-splitting + run emission for entity-free text (see
    build_text_runs for the full policy)."""
    # 1) Regions: chinese regions absorb adjacent whitespace-only spans.
    #    Non-chinese regions are stripped of edge whitespace first, so the
    #    spaces around identifiers/English never become placeholder runs
    #    (M2M100 drops whitespace-only placeholders; the whitespace is
    #    carried by adjacent chinese runs' original edge whitespace instead).
    regions: List[tuple] = []  # (is_chinese, text)
    pending_ws = ""
    for is_chinese, span in split_mixed_spans(raw):
        if is_chinese:
            regions.append((True, pending_ws + span))
            pending_ws = ""
        elif not span.strip():
            pending_ws += span
        else:
            if pending_ws:
                regions.append((False, pending_ws))
                pending_ws = ""
            leading = span[: len(span) - len(span.lstrip())]
            trailing = span[len(span.rstrip()):]
            core = span.strip()
            if leading:
                regions.append((True, leading))
            regions.append((False, core))
            if trailing:
                regions.append((True, trailing))
    if pending_ws:
        if regions and regions[-1][0]:
            regions[-1] = (True, regions[-1][1] + pending_ws)
        else:
            regions.append((False, pending_ws))

    # 2) Merge adjacent chinese regions into single runs.
    merged: List[tuple] = []
    for is_chinese, text in regions:
        if is_chinese and merged and merged[-1][0]:
            merged[-1] = (True, merged[-1][1] + text)
        else:
            merged.append((is_chinese, text))

    # 3) Emit runs.
    runs: List[Run] = []
    cursor = 0
    for is_chinese, span in merged:
        # Glossary terms (terminology memory) become protected runs in BOTH
        # chinese and english regions; the gaps keep their normal handling.
        # Protected identifiers win over glossary terms (identifiers are
        # non-negotiable exact).
        excluded_ranges = []
        if not is_chinese:
            excluded_ranges = [
                (s, e) for s, e, _kind, _content in find_protected_spans(span)
            ]
        pos = 0
        for start, end, entry in find_glossary_spans(
            span, glossary or (), excluded_ranges=excluded_ranges
        ):
            if start > pos:
                _emit_region(runs, node_id, is_chinese, span[pos:start],
                             pmap, offset_base + cursor + pos,
                             preserve_patterns)
            token = pmap.reserve(entry.target, kind="glossary")
            runs.append(Run(
                node_id=node_id,
                kind=RUN_GLOSSARY,
                raw=entry.source,
                protected=token,
                placeholder=token,
                restore_text=entry.target,
                offset_start=offset_base + cursor + start,
                offset_end=offset_base + cursor + end,
                sequence_index=len(runs),
            ))
            pos = end
        if pos < len(span):
            _emit_region(runs, node_id, is_chinese, span[pos:],
                         pmap, offset_base + cursor + pos,
                         preserve_patterns)
        cursor += len(span)
    return runs


def _emit_region(runs: List[Run], node_id: str, is_chinese: bool, span: str,
                 pmap: ProtectionMap, offset_base: int,
                 preserve_patterns=()) -> None:
    """Emit normal runs for a region gap (no glossary terms inside)."""
    if is_chinese:
        runs.append(Run(
            node_id=node_id,
            kind=RUN_CHINESE,
            raw=span,
            protected=span,
            translate=True,
            offset_start=offset_base,
            offset_end=offset_base + len(span),
            sequence_index=len(runs),
        ))
        return

    # Non-Chinese span: user-configured preserve patterns first (project
    # model formats), then identifiers, then the remaining English.
    if preserve_patterns:
        matches = []
        for pat in preserve_patterns:
            for m in pat.finditer(span):
                matches.append((m.start(), m.end(), m.group(0)))
        matches.sort()
        selected = []
        last_end = -1
        for start, end, text in matches:
            if start < last_end:
                continue
            selected.append((start, end, text))
            last_end = end
        pos = 0
        for start, end, text in selected:
            if start > pos:
                _emit_english_span(runs, node_id, span[pos:start],
                                   pmap, offset_base + pos)
            token = pmap.reserve(text, kind="model")
            runs.append(Run(
                node_id=node_id,
                kind=RUN_MODEL_NUMBER,
                raw=text,
                protected=token,
                placeholder=token,
                offset_start=offset_base + start,
                offset_end=offset_base + end,
                sequence_index=len(runs),
            ))
            pos = end
        if pos < len(span):
            _emit_english_span(runs, node_id, span[pos:],
                               pmap, offset_base + pos)
        return

    _emit_english_span(runs, node_id, span, pmap, offset_base)


def _emit_english_span(runs: List[Run], node_id: str, span: str,
                       pmap: ProtectionMap, offset_base: int) -> None:
    """Protect identifiers, then protect the remaining English text."""
    protected = protect_identifiers(span, pmap)
    parts = re.split(r"(__ITRANSLATE_[A-Z]\d{4}_)", protected)
    cursor = 0
    for part in parts:
        if not part:
            continue
        existing = pmap.span(part)
        if existing is not None:
            # identifier placeholder already reserved
            runs.append(Run(
                node_id=node_id,
                kind=RUN_IDENTIFIER,
                raw=existing.content,
                protected=part,
                placeholder=part,
                offset_start=offset_base + cursor,
                offset_end=offset_base + cursor + len(existing.content),
                sequence_index=len(runs),
            ))
            cursor += len(existing.content)
        else:
            token = pmap.reserve(part, kind="english")
            runs.append(Run(
                node_id=node_id,
                kind=RUN_ENGLISH,
                raw=part,
                protected=token,
                placeholder=token,
                offset_start=offset_base + cursor,
                offset_end=offset_base + cursor + len(part),
                sequence_index=len(runs),
            ))
            cursor += len(part)
    return runs


def _last_boundary(text: str, limit: int, boundaries: str) -> int:
    """Index (exclusive) of the last boundary char within text[:limit], else -1."""
    best = -1
    for i in range(min(limit, len(text))):
        if text[i] in boundaries:
            best = i + 1
    return best


def _with_spacing(text: str, prefix: str = "__ITRANSLATE_") -> str:
    """Model-facing source: wrap every placeholder token in spaces.

    Empirically verified with M2M100: a placeholder directly adjacent to CJK
    text (or to another placeholder) gets re-segmented and can be dropped by
    the model; space-separated placeholders are preserved exactly. The extra
    spaces are model-side only — reconstruction maps output pieces onto the
    recorded slots, and spaces around tokens land in empty slots (dropped)
    or at piece edges (normalized by edge-whitespace preservation).
    """
    return re.sub(
        r"(" + re.escape(prefix) + r"[A-Z]\d{4}_)", r" \1 ", text
    )


def _hard_split_prefix(text: str, budget_tokens: int, measure) -> int:
    """Largest prefix length of text whose token count fits the budget."""
    if measure(text) <= budget_tokens:
        return len(text)
    lo, hi = 1, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if measure(text[:mid]) <= budget_tokens:
            lo = mid
        else:
            hi = mid - 1
    cut = lo
    # Never cut inside a placeholder token (trailing/leading underscore)
    while cut > 0 and (text[cut - 1] == "_" or (cut < len(text) and text[cut] == "_")):
        cut -= 1
    return max(cut, 1)


def segment_blocks(
    doc: HTMLDocument,
    blocks: List[Block],
    measure: Callable[[str], int],
    max_segment_tokens: int,
    document_id: str = "doc",
    source_language: str = "zh",
    target_language: str = "en",
    glossary=None,
    preserve_patterns=(),
    translatable_attrs=frozenset(),
) -> List[Segment]:
    """Turn blocks into model-sized segments, preserving document order.

    Greedy packing: runs and tag placeholders are accumulated in order; when
    the next text would exceed the budget, the text is split at a
    sentence/clause boundary (hard split last), and the segment is flushed.
    Tag placeholders are never split and always stay inside the segment that
    contains them. ``glossary`` entries become protected runs (terminology
    memory) — see build_text_runs.

    Build-time coverage invariants (fail closed with ValueError):
    - ``segment.source_text`` == spacing-normalized concatenation of every
      run's protected text;
    - concatenating all original runs of a block reproduces ``block.block_text``
      with contiguous, non-overlapping character offsets;
    - a translate run's slot index equals the number of placeholders before
      it, and slots are consistent 0..n-1.
    """
    segments: List[Segment] = []
    seq = 0
    block_seq = 0

    for block in blocks:
        pmap = ProtectionMap()
        # Throwaway map for token-measurement probes: probing must not
        # reserve real placeholder tokens (probe tokens are never in cur_src).
        probe_pmap = ProtectionMap()
        items = prepare_block(block, pmap, translatable_attrs=translatable_attrs)
        block.block_key = f"b{block_seq:04d}"
        block_seq += 1

        cur_src = ""
        cur_runs: List[Run] = []
        cur_has_text = False            # any translate run yet?
        # Original-text cursor within block_text (offsets for coverage).
        orig_cursor = 0

        def flush() -> None:
            nonlocal cur_src, cur_runs, cur_has_text, orig_cursor, seq
            if not cur_has_text:
                return
            source_text = _with_spacing(
                "".join(r.protected for r in cur_runs)
            )
            token_count = measure(source_text)
            runs = list(cur_runs)
            placeholders = [r.placeholder for r in runs if r.placeholder]
            slots: List[Optional[Run]] = [None] * (len(placeholders) + 1)
            for run in runs:
                if run.translate:
                    slots[run.slot_index] = run
            seg = Segment(
                document_id=document_id,
                segment_id=f"{document_id}:{seq:04d}",
                sequence_index=seq,
                source_node_ids=sorted({r.node_id for r in runs
                                        if r.node_id != "tag"}),
                source_text=source_text,
                protected_map=pmap,
                source_language=source_language,
                target_language=target_language,
                token_count=token_count,
                context_before_id=segments[-1].segment_id if segments else None,
                runs=runs,
                placeholder_order=placeholders,
                slots=slots,
                block_key=block.block_key,
                block_text=block.block_text,
            )
            segments.append(seg)
            seq += 1
            cur_src = ""
            cur_runs = []
            cur_has_text = False

        def append_tag_run(content: str, token: str) -> None:
            nonlocal cur_src, cur_runs, orig_cursor
            cur_src += token
            cur_runs.append(Run(
                node_id="tag",
                kind=RUN_TAG,
                raw=content,
                protected=token,
                placeholder=token,
                offset_start=orig_cursor,
                offset_end=orig_cursor + len(content),
                sequence_index=len(cur_runs),
            ))
            orig_cursor += len(content)

        def append_runs(runs: List[Run]) -> None:
            """Append text runs; slot_index = placeholders seen so far."""
            nonlocal cur_src, cur_runs, cur_has_text, orig_cursor
            ph_count = len([r for r in cur_runs if r.placeholder])
            for run in runs:
                if run.translate:
                    run.slot_index = ph_count
                    cur_has_text = True
                else:
                    ph_count += 1  # protected run consumes a placeholder slot
                cur_src += run.protected
                run.sequence_index = len(cur_runs)
                cur_runs.append(run)
            orig_cursor += sum(len(r.raw) for r in runs)

        for item in items:
            if item["kind"] == "tag":
                token = item["token"]
                if measure(_with_spacing(cur_src + token)) <= max_segment_tokens or not cur_has_text:
                    append_tag_run(item["content"], token)
                else:
                    flush()
                    append_tag_run(item["content"], token)
                continue

            if item["kind"] == "whitespace":
                continue  # preserved in the document; never sent

            # text item (raw text; protection applied per slot)
            text = item["text"]
            node_id = item["node_id"]

            # Adjacent-text guard: two text items with NO tag/placeholder
            # between them would produce two translate runs sharing one
            # model-output slot (the model renders their combined text as a
            # single piece that cannot be split between nodes). Flush the
            # current segment so every segment ends with a placeholder or
            # starts fresh — piece-to-run alignment stays 1:1.
            if cur_has_text and cur_runs and cur_runs[-1].translate:
                flush()

            def protected_for(fragment: str) -> str:
                probe_runs = build_text_runs(
                    node_id, fragment, probe_pmap, 0, glossary=glossary,
                    preserve_patterns=preserve_patterns,
                )
                return "".join(r.protected for r in probe_runs)

            def emit(fragment: str) -> None:
                append_runs(
                    build_text_runs(
                        node_id, fragment, pmap, orig_cursor, glossary=glossary,
                        preserve_patterns=preserve_patterns,
                    )
                )

            while text:
                # Adjacent-text guard (see above): after a split, the
                # remainder must start a fresh segment when the current one
                # ends with a translate run — one slot per run, always.
                if cur_has_text and cur_runs and cur_runs[-1].translate:
                    flush()
                probe = protected_for(text)
                if measure(_with_spacing(cur_src + probe)) <= max_segment_tokens:
                    emit(text)
                    text = ""
                else:
                    # Glossary terms are atomic: a cut must never land inside
                    # one (the same term must be glossary-mapped in exactly
                    # one segment — consistency by construction). Entity
                    # sentinel markers are equally atomic: a cut inside a
                    # marker would split one entity across two segments and
                    # corrupt the exact-spelling restoration.
                    term_spans = [
                        (s, e) for s, e, _entry
                        in find_glossary_spans(text, glossary or ())
                    ]
                    marker_spans = [
                        (m.start(), m.end())
                        for m in _ENTITY_MARKER_RE.finditer(text)
                    ]

                    def avoid_terms(cut: int) -> int:
                        for s, e in term_spans:
                            if s < cut < e:
                                # move to the nearer atomic boundary
                                return s if (cut - s) <= (e - cut) else e
                        for s, e in marker_spans:
                            if s < cut < e:
                                # Prefer the marker START: the prefix always
                                # fits the measured budget (a partial marker
                                # would corrupt exact-spelling restoration).
                                # Only when the marker starts the text (cut
                                # would make no progress) snap to its end.
                                return s if s > 0 else e
                        return cut

                    # Find the max prefix of `text` that fits the remaining budget
                    budget = max_segment_tokens
                    lo, hi = 1, len(text)
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if measure(_with_spacing(cur_src + protected_for(text[:mid]))) <= budget:
                            lo = mid
                        else:
                            hi = mid - 1
                    max_prefix = avoid_terms(lo)
                    cut = _last_boundary(text, max_prefix, SENTENCE_BOUNDARY)
                    if cut <= 0:
                        cut = _last_boundary(text, max_prefix, CLAUSE_BOUNDARY)
                    cut = avoid_terms(cut)
                    if cut <= 0:
                        # No boundary within the remaining budget: flush the
                        # current segment; the remainder may fit alone.
                        flush()
                        if measure(_with_spacing(protected_for(text))) <= max_segment_tokens:
                            emit(text)
                            text = ""
                        else:
                            # Hard split the remainder so each piece fits alone
                            cut = avoid_terms(_hard_split_prefix(
                                text, max_segment_tokens,
                                lambda frag: measure(_with_spacing(protected_for(frag))),
                            ))
                            if (cut <= 0 or cut >= len(text)
                                    or measure(_with_spacing(protected_for(text[:cut])))
                                    > max_segment_tokens):
                                raise ValueError(
                                    f"indivisible text unit exceeds token budget "
                                    f"(node {node_id}): no prefix fits "
                                    f"{max_segment_tokens} tokens"
                                )
                            emit(text[:cut])
                            text = text[cut:]
                        continue
                    if cut <= 0 or cut >= len(text):
                        raise ValueError(
                            f"indivisible text unit exceeds token budget "
                            f"(node {node_id})"
                        )
                    # Flush current segment, then start the next with the prefix
                    flush()
                    emit(text[:cut])
                    text = text[cut:]
        flush()

    # Global context links: last segment of block N -> first of block N+1
    for i in range(len(segments) - 1):
        if segments[i].context_after_id is None:
            segments[i].context_after_id = segments[i + 1].segment_id

    _validate_coverage(segments)
    return segments


def _validate_coverage(segments: List[Segment]) -> None:
    """Fail closed on run-metadata invariants (change 2)."""
    from collections import defaultdict

    by_block: "defaultdict[str, List[Segment]]" = defaultdict(list)
    for seg in segments:
        by_block[seg.block_key].append(seg)

    for block_key, block_segs in by_block.items():
        block_text = block_segs[0].block_text
        # 1) concatenating all original runs reproduces the block text
        cat = "".join(r.raw for s in block_segs for r in s.runs)
        if cat != block_text:
            raise ValueError(
                f"block {block_key}: coverage invariant failed: concatenated "
                "runs do not reproduce the block text"
            )
        # 2) offsets are contiguous, non-overlapping, and cover the block
        offsets = sorted(
            (r.offset_start, r.offset_end) for s in block_segs for r in s.runs
        )
        pos = 0
        for start, end in offsets:
            if start != pos:
                raise ValueError(
                    f"block {block_key}: coverage invariant failed: run offset "
                    f"gap at {pos} (next run starts at {start})"
                )
            if end < start:
                raise ValueError(
                    f"block {block_key}: coverage invariant failed: negative "
                    "run span"
                )
            pos = end
        if pos != len(block_text):
            raise ValueError(
                f"block {block_key}: coverage invariant failed: runs cover "
                f"{pos} of {len(block_text)} block chars"
            )

    # 3) every segment's source is the spacing-normalized concatenation of
    #    run.protected (model-facing spaces around placeholders)
    for seg in segments:
        cat = _with_spacing("".join(r.protected for r in seg.runs))
        if cat != seg.source_text:
            raise ValueError(
                f"segment {seg.segment_id}: source_text does not match "
                f"concatenated protected runs"
            )
        # 4) slot/placeholder layout matches
        if len(seg.placeholder_order) + 1 != len(seg.slots):
            raise ValueError(
                f"segment {seg.segment_id}: slot/placeholder layout mismatch "
                f"({len(seg.slots)} slots vs {len(seg.placeholder_order)} "
                f"placeholders)"
            )
        for i, run in enumerate(seg.slots):
            if run is not None and not run.translate:
                raise ValueError(
                    f"segment {seg.segment_id}: non-translate run in slot {i}"
                )
        # 5) slot indices are unique, in range, and consistent with the layout
        slot_idx = [r.slot_index for r in seg.runs if r.translate]
        if len(slot_idx) != len(set(slot_idx)):
            raise ValueError(
                f"segment {seg.segment_id}: duplicate slot indices"
            )
        for i, run in enumerate(seg.slots):
            if run is not None and run.slot_index != i:
                raise ValueError(
                    f"segment {seg.segment_id}: slot {i} holds run with "
                    f"slot_index {run.slot_index}"
                )
