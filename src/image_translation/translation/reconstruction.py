"""Ordered merge of translated segments back into the document, with
fail-closed integrity checks.

Reconstruction contract (documented):
- segment outputs are split at EVERY placeholder token (tags, protected
  English, identifiers) in first-occurrence order; the pieces align 1:1 with
  the segment's recorded slots (``slots[i]`` is the text between
  placeholder[i-1] and placeholder[i]);
- protected runs (English, identifiers, tags) are restored from their ORIGINAL
  raw text — never from model output;
- chinese runs consume exactly one model-output piece each; a node split
  across segments/runs accumulates pieces in segment order;
- model output in an EMPTY slot (no source text at that position) is
  dropped — it has no source position to map to (logged at debug);
- translatable-attribute segments are applied to their element (values
  replaced only for allowlisted attributes);
- the structural fingerprint (tag nesting, attrs minus translatable values,
  comments, doctypes, text-node IDs) must be identical before and after;
  excluded (never-translated) text must be byte-identical;
- any mismatch fails closed with StructuredTranslationError. No partial
  output is ever returned.
- text inserted into text nodes is escaped by the serializer on output, so
  model-generated markup can never become executable tags.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from .chapter_chunking import Segment
from .exceptions import StructuredTranslationError
from .html_document import HTMLDocument

logger = logging.getLogger(__name__)


def rebuild_document(
    doc: HTMLDocument,
    segments: List[Segment],
    fingerprint_before: str,
    attr_segments: Optional[List[Segment]] = None,
    excluded_tags=("script", "style", "code", "pre"),
    excluded_classes=("notranslate",),
    translatable_attrs: Set[str] = frozenset(),
) -> str:
    """Merge translated segments into the document and validate.

    Returns the serialized translated HTML.

    Raises:
        StructuredTranslationError: On placeholder issues, alignment
            mismatches, excluded-content changes, or fingerprint changes.
    """
    attr_segments = attr_segments or []
    node_content: Dict[str, str] = {}

    # Snapshot excluded (never-translated) text content; it must be byte-identical
    excluded_ids = doc.excluded_text_node_ids(excluded_tags, excluded_classes)
    excluded_snapshot = {
        nid: node.text
        for nid, node in ((nid, doc.get_text_node(nid)) for nid in excluded_ids)
        if node is not None
    }

    for seg in segments:
        pieces, protected_sequence = seg.protected_map.restore_split(
            seg.translated_text
        )

        # Alignment: pieces must map 1:1 onto the recorded slots, and the
        # placeholder sequence must match the source layout exactly.
        if len(pieces) != len(seg.slots):
            raise StructuredTranslationError(
                f"segment {seg.segment_id}: output piece count {len(pieces)} "
                f"does not match slot count {len(seg.slots)} (placeholders "
                f"were not preserved in order)"
            )
        expected_order = seg.placeholder_order
        # Tag-relative order validation (same contract as validate_output):
        # tags in source order; non-tag spans stay in the same tag interval.
        # Within an interval the model may reorder spans — pieces are then
        # mapped positionally (the model's own rendering).
        def _kind(tok):
            span = seg.protected_map.span(tok)
            return span.kind if span is not None else ""

        tag_expected = [t for t in expected_order if _kind(t) in ("tag_start", "tag_end")]
        tag_found = [s.token for s in protected_sequence
                     if s.kind in ("tag_start", "tag_end")]
        if tag_found != tag_expected:
            raise StructuredTranslationError(
                f"segment {seg.segment_id}: tag placeholder order changed; "
                f"refusing to reconstruct"
            )

        def _intervals(tokens):
            intervals = {}
            tags_before = 0
            for t in tokens:
                if _kind(t) in ("tag_start", "tag_end"):
                    tags_before += 1
                else:
                    intervals[t] = tags_before
            return intervals

        found_intervals = _intervals([s.token for s in protected_sequence])
        expected_intervals = _intervals(expected_order)
        moved = [t for t in expected_intervals
                 if found_intervals.get(t) != expected_intervals[t]]
        if moved:
            raise StructuredTranslationError(
                f"segment {seg.segment_id}: protected spans moved across "
                f"tag boundaries: {moved}; refusing to reconstruct"
            )

        # Walk the runs in source order: chinese runs consume their piece
        # (slot order == piece order); protected runs restore their ORIGINAL
        # raw text; model output in empty slots is dropped.
        # Walk the runs in source order: chinese runs consume the piece at
        # their slot index (slot order == piece order, empty slots included);
        # protected runs restore their ORIGINAL raw text.
        for run in seg.runs:
            if run.node_id == "tag":
                continue
            if run.translate:
                piece = pieces[run.slot_index]
                # Edge-whitespace normalization: ALL source whitespace around
                # a translated run is carried by the run's own original edge
                # whitespace or by adjacent protected runs' raw text — the
                # model's edge whitespace is always redundant. Model edges
                # are stripped, then the ORIGINAL edge whitespace is restored
                # (leading/trailing computed on disjoint slices so a
                # whitespace-only run never double-counts its space).
                leading = run.raw[: len(run.raw) - len(run.raw.lstrip())]
                rest = run.raw[len(leading):]
                trailing = rest[len(rest.rstrip()):]
                piece = leading + piece.strip() + trailing
                node_content[run.node_id] = node_content.get(run.node_id, "") + piece
            else:
                # Protected run (English/identifier/glossary): restore the
                # ORIGINAL text — glossary runs restore their configured
                # TARGET term (restore_text); everything else restores raw.
                node_content[run.node_id] = node_content.get(
                    run.node_id, ""
                ) + (run.restore_text or run.raw)
        for i, slot in enumerate(seg.slots):
            if slot is None and pieces[i].strip():
                logger.debug(
                    "segment %s: dropping model output in empty slot: %r",
                    seg.segment_id, pieces[i],
                )

    # Apply translated text to the document nodes
    for node in doc.text_nodes():
        if node.id in node_content:
            node.text = node_content[node.id]

    # Apply translated attribute values (allowlisted only)
    for seg in attr_segments:
        if len(seg.runs) != 1 or not seg.runs[0].translate:
            raise StructuredTranslationError(
                f"attribute segment {seg.segment_id} must contain exactly one "
                f"translate run"
            )
        run = seg.runs[0]
        if run.node_id != "tag" and not run.node_id.startswith("attr:"):
            raise StructuredTranslationError(
                f"attribute segment {seg.segment_id} has invalid run id "
                f"{run.node_id!r}"
            )
        if run.node_id.startswith("attr:"):
            _prefix, elem_id, attr_name = run.node_id.split(":", 2)
            if attr_name not in translatable_attrs:
                raise StructuredTranslationError(
                    f"attribute segment {seg.segment_id} targets "
                    f"non-allowlisted attribute {attr_name!r}"
                )
        else:
            elem_id, attr_name = None, None

        pieces, _ = seg.protected_map.restore_split(seg.translated_text)
        if len(pieces) != 1:
            raise StructuredTranslationError(
                f"attribute segment {seg.segment_id}: expected 1 piece, "
                f"got {len(pieces)}"
            )
        if elem_id is not None:
            elem = doc.get_element_node(elem_id)
            if elem is None:
                raise StructuredTranslationError(
                    f"attribute segment {seg.segment_id}: element {elem_id} "
                    f"not found"
                )
            attrs = list(elem.attrs)
            for i, (name, _value) in enumerate(attrs):
                if name == attr_name:
                    attrs[i] = (name, pieces[0])
                    break
            else:
                raise StructuredTranslationError(
                    f"attribute segment {seg.segment_id}: attribute "
                    f"{attr_name} missing on element {elem_id}"
                )
            elem.attrs = attrs

    translated_html = doc.serialize()

    # Excluded content must be byte-identical
    for nid, expected in excluded_snapshot.items():
        node = doc.get_text_node(nid)
        if node is None or node.text != expected:
            raise StructuredTranslationError(
                f"excluded (never-translated) content changed for node {nid}; "
                f"refusing to return corrupted document"
            )

    # Fail closed on structural drift
    fingerprint_after = doc.fingerprint(translatable_attrs=translatable_attrs)
    if fingerprint_after != fingerprint_before:
        raise StructuredTranslationError(
            "structural fingerprint changed after translation; "
            "refusing to return corrupted document"
        )

    return translated_html
