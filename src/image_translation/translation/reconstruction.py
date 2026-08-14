"""Ordered merge of translated segments back into the document, with
fail-closed integrity checks.

Reconstruction contract (documented):
- segment outputs are split at EVERY placeholder token (tags, protected
  English, identifiers) in first-occurrence order; the pieces align 1:1 with
  the segment's recorded slots (``slots[i]`` is the text between
  placeholder[i-1] and placeholder[i]);
- STRICT source-order contract: the complete protected-token sequence in the
  model output must equal the segment's source ``placeholder_order`` EXACTLY
  — tags, entities, bare-ampersand runs, English spans, identifiers, and
  glossary terms included. Reordering within a tag interval is rejected:
  protected content stays in its original source slot and translated pieces
  map to their source slots from the exact sequence, never positionally
  after a reordered placeholder sequence;
- model-invented placeholder-like tokens (default or retry prefixes) that
  are not registered in the segment's ProtectionMap fail closed; they can
  never leak into the output;
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
from .html_protection import find_unknown_placeholders

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
        # Fail closed on model-invented placeholder-like tokens (default or
        # retry prefixes) not registered in this segment's protection map —
        # they must never leak into the output as text.
        unknown = find_unknown_placeholders(seg.translated_text, seg.protected_map)
        if unknown:
            raise StructuredTranslationError(
                f"segment {seg.segment_id}: unknown placeholder(s) invented "
                f"by the model: {unknown}; refusing to reconstruct"
            )

        pieces, protected_sequence = seg.protected_map.restore_split(
            seg.translated_text
        )

        # Alignment: pieces must map 1:1 onto the recorded slots, and the
        # placeholder sequence must match the source layout EXACTLY.
        if len(pieces) != len(seg.slots):
            raise StructuredTranslationError(
                f"segment {seg.segment_id}: output piece count {len(pieces)} "
                f"does not match slot count {len(seg.slots)} (placeholders "
                f"were not preserved in order)"
            )
        expected_order = seg.placeholder_order
        # Strict full-sequence contract: the complete protected-token
        # sequence in the model output must equal the source placeholder
        # order exactly — tags, entities, bare-ampersand runs, English spans,
        # identifiers, and glossary terms included. Reordering within a tag
        # interval is REJECTED: protected content must stay in its original
        # source slot, and translated pieces map to their source slots
        # positionally from the exact sequence.
        found_order = [s.token for s in protected_sequence]
        if found_order != expected_order:
            raise StructuredTranslationError(
                f"segment {seg.segment_id}: protected placeholder sequence "
                f"differs from source order (expected {expected_order}, "
                f"got {found_order}); refusing to reconstruct"
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

        pieces, protected_sequence = seg.protected_map.restore_split(seg.translated_text)
        seq = [s.token for s in protected_sequence]
        if seq == seg.placeholder_order:
            # Normal path: protected identifiers restored from the source
            # map at their exact positions (never from model output).
            value = ""
            for i, piece in enumerate(pieces):
                value += piece
                if i < len(protected_sequence):
                    value += protected_sequence[i].content
        elif not seq and len(pieces) == 1:
            # Split-fallback path: the per-run outputs were already
            # validated and their protected identifiers restored from the
            # source map; the rebuilt value contains no placeholder tokens.
            value = pieces[0]
        else:
            raise StructuredTranslationError(
                f"attribute segment {seg.segment_id}: protected sequence "
                f"{seq} does not match {seg.placeholder_order} and is not "
                f"a validated fallback output; refusing to reconstruct"
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
            # The raw source spelling carried the ORIGINAL attribute value;
            # the translated value must be serialized instead.
            elem.raw_start = None
            elem.raw_end = None

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
