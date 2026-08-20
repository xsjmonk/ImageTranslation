"""Structured (HTML-aware) translation orchestration.

Public entry point for the API and CLI. Never imports FastAPI.

Pipeline (documented):
1. size validation (configurable max chapter size; no silent truncation);
2. parse with the html5lib-based document model;
3. structural fingerprint before translation (translatable-attr values
   excluded — they legitimately change);
4. collect translatable blocks (mixed-language grouping: English inside a
   mixed block is PROTECTED, never sent to the model as free text; excluded
   subtrees are never sent);
5. token-aware segmentation with an explicit run model (truncation=False
   measurement; every English/identifier/tag span has a placeholder);
6. TRUE bounded first-pass batching: segments are grouped (configurable
   batch_size, source order preserved) and each batch is sent to the shared
   translator's translate_batch_texts() ONCE (never HTTP), with the REQUEST
   source/target languages propagated to every model call and the batch
   target budget taken as the MAX of the per-segment budgets (no budget is
   silently lowered);
7. per-segment placeholder validation of every batch item (strict complete
   sequence: every placeholder exactly once, in source order; unknown/
   invented placeholder tokens rejected) — only failed items are retried
   individually (fresh placeholder namespace), then per-chinese-run split
   fallback; successful items are never re-sent; an unrecoverable segment
   fails the request closed with no partial output;
8. ordered reconstruction from run metadata (never string splitting) with
   fail-closed fingerprint/excluded checks.

Protected-span strategy (documented): ordinary English and identifiers are
replaced with collision-resistant placeholders BEFORE inference and restored
from the ORIGINAL text afterwards. The model can never rewrite, drop, or
paraphrase English — exact preservation does not depend on post-hoc
detection. Context is NOT supplied to the model: the configured sequence-to-sequence model's generate() has no
reliable context-injection API, so context_window_tokens is reserved and
MUST stay 0. Terminology consistency across segments is guaranteed for
protected terms; for translatable Chinese terms it depends on deterministic
generation (fixed beams).
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .base import Translator
from .chapter_chunking import (
    RUN_ATTRIBUTE,
    RUN_IDENTIFIER,
    RUN_TAG,
    Segment,
    collect_blocks,
    segment_blocks,
)
from .config import StructuredConfig, TranslationConfig
from .exceptions import (
    BatchItemError,
    StructuredTranslationError,
    TranslationQualityError,
)
from .html_document import HTMLDocument
from .html_protection import DEFAULT_PREFIX, ProtectionMap, assert_prefix_safe
from .language_segments import LanguageKind, classify, protect_identifiers
from .reconstruction import rebuild_document

logger = logging.getLogger(__name__)

# Documented quantized target-budget buckets. Segments are grouped by
# (language pair, bucket(required_budget)); the batch's max_new_tokens is
# the bucket, which is NEVER below any member's required budget (buckets
# only raise up to the next boundary) and never exceeds max_target_tokens.
# Bucketing keeps short segments from running with a large max_new_tokens
# while preventing pathological group fragmentation (every distinct
# 2.5x-derived budget becoming its own single-item batch).
TARGET_BUDGET_BUCKETS = (64, 128, 192, 256, 320, 400)


def _budget_bucket(budget: int) -> int:
    """Smallest documented bucket >= budget (never lowers; beyond the
    buckets the exact budget is used)."""
    for b in TARGET_BUDGET_BUCKETS:
        if budget <= b:
            return b
    return budget


def _validate_batch_results(results, expected_count: int) -> List[str]:
    """Validate a translator result collection at the shared boundary.

    - ``None`` / a scalar / a non-sized object are rejected with
      StructuredTranslationError (no raw TypeError/AttributeError escapes);
    - the exact result count is validated BEFORE any zip()/indexing;
    - each item's ``translated_text`` must be a non-null string; malformed
      items raise BatchItemError carrying the bad indices and the validated
      outputs of the good neighbors so the caller can recover only the
      affected inputs (valid neighbors are never re-sent).

    Returns the validated translated strings in input order.
    """
    if results is None:
        raise StructuredTranslationError(
            "translator returned no result collection"
        )
    try:
        actual_count = len(results)
    except (TypeError, AttributeError) as exc:
        raise StructuredTranslationError(
            f"translator returned an invalid result collection: "
            f"{type(results).__name__}"
        ) from exc
    if actual_count != expected_count:
        raise StructuredTranslationError(
            f"translator returned {actual_count} results for "
            f"{expected_count} inputs"
        )
    bad_indices: List[int] = []
    valid_outputs: Dict[int, str] = {}
    for index, result in enumerate(results):
        value = getattr(result, "translated_text", None)
        if not isinstance(value, str):
            bad_indices.append(index)
            continue
        valid_outputs[index] = value
    if bad_indices:
        raise BatchItemError(
            f"translator returned non-string translated_text at batch "
            f"item(s) {bad_indices}: "
            f"{[type(results[i]).__name__ for i in bad_indices]}",
            bad_indices,
            valid_outputs,
        )
    return [valid_outputs[i] for i in range(actual_count)]

# CJK ideographs only (for repeated-term reporting)
_CJK_GRAM_RE = re.compile(r"^[\u4e00-\u9fff]+$")


def _collect_repeated_terms(
    doc: HTMLDocument, excluded_ids: set, min_count: int = 3, top_n: int = 20
) -> Dict[str, int]:
    """Repeated CJK bigrams/trigrams in TRANSLATABLE text (informational).

    Excluded content is never scanned. Reported terms are NOT replaced —
    only configured protected terminology entries drive replacement.
    """
    from collections import Counter

    counter: "Counter[str]" = Counter()
    for node in doc.text_nodes():
        if node.id in excluded_ids:
            continue
        text = node.text
        for n in (2, 3):
            for i in range(len(text) - n + 1):
                gram = text[i:i + n]
                if _CJK_GRAM_RE.fullmatch(gram):
                    counter[gram] += 1
    return {
        term: count
        for term, count in counter.most_common(top_n)
        if count >= min_count
    }


def _collect_identifier_occurrences(segments: List[Segment], cap: int = 50) -> dict:
    """Protected identifier record: content -> {occurrences, segments[]}
    (identifiers are exact by construction; recorded for audit)."""
    result: Dict[str, dict] = {}
    for seg in segments:
        for run in seg.runs:
            if run.kind != RUN_IDENTIFIER:
                continue
            info = result.setdefault(
                run.raw, {"occurrences": 0, "segments": []}
            )
            info["occurrences"] += 1
            if seg.segment_id not in info["segments"]:
                info["segments"].append(seg.segment_id)
    # cap the number of distinct identifiers reported
    if len(result) > cap:
        return dict(list(result.items())[:cap])
    return result


@dataclass
class StructuredTranslationResult:
    """Outcome of one structured translation request."""

    translated_html: str
    correlation_id: str
    segment_count: int
    total_source_tokens: int
    total_target_tokens: int
    retry_count: int
    fallback_count: int
    protected_run_count: int
    duration_seconds: float
    fingerprint_ok: bool
    excluded_text_nodes: int
    translated_attributes: int
    source_language: str
    target_language: str
    batch_count: int = 0
    batch_generation_budget: int = 0
    sum_requested_target_tokens: int = 0
    batch_metrics: List[dict] = field(default_factory=list)
    segments: List[dict] = field(default_factory=list)
    # Machine-readable metrics (JSON-serializable): terminology map,
    # repeated terms, budgets, validation status. See to_dict().
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Machine-readable result (segment count, source tokens, target
        budget, protected-run count, terminology occurrences, elapsed time,
        retry/fallback count, validation status)."""
        return {
            "correlation_id": self.correlation_id,
            "segment_count": self.segment_count,
            "total_source_tokens": self.total_source_tokens,
            "total_target_tokens": self.total_target_tokens,
            "sum_requested_target_tokens": self.sum_requested_target_tokens,
            "batch_generation_budget": self.batch_generation_budget,
            "protected_run_count": self.protected_run_count,
            "terminology": self.metrics.get("terminology", {}),
            "repeated_terms": self.metrics.get("repeated_terms", {}),
            "duration_seconds": round(self.duration_seconds, 3),
            "retry_count": self.retry_count,
            "fallback_count": self.fallback_count,
            "batch_count": self.batch_count,
            "batch_metrics": self.batch_metrics,
            "validation": "ok",
            "fingerprint_ok": self.fingerprint_ok,
            "excluded_text_nodes": self.excluded_text_nodes,
            "translated_attributes": self.translated_attributes,
            "source_language": self.source_language,
            "target_language": self.target_language,
        }


class StructuredTranslator:
    """Orchestrates HTML-aware translation over an existing Translator."""

    def __init__(
        self,
        translator: Translator,
        config: StructuredConfig,
        translation_config: Optional[TranslationConfig] = None,
        document_id: str = "doc",
    ) -> None:
        self._translator = translator
        self._config = config
        self._translation_config = translation_config or TranslationConfig()
        self._document_id = document_id

    # ------------------------------------------------------------------

    def translate(
        self,
        html: str,
        source_lang: str = "zh",
        target_lang: str = "en",
    ) -> StructuredTranslationResult:
        """Translate an HTML chapter; returns the reconstructed document.

        Raises:
            StructuredTranslationError: size violation, parse failure,
                protection failure, deadline exceeded, or reconstruction
                validation failure.
        """
        start = time.monotonic()
        correlation_id = uuid.uuid4().hex[:12]
        cfg = self._config
        deadline = start + cfg.max_total_seconds

        if not isinstance(html, str):
            raise StructuredTranslationError("HTML input must be a string")
        if len(html) > cfg.max_chapter_characters:
            raise StructuredTranslationError(
                f"document exceeds max_chapter_characters "
                f"({len(html)} > {cfg.max_chapter_characters}); "
                f"no silent truncation is performed"
            )

        # --- parse + fingerprint ---
        try:
            doc = HTMLDocument(html)
        except ValueError as e:
            raise StructuredTranslationError(str(e)) from e

        assert_prefix_safe(html, DEFAULT_PREFIX)
        translatable_attrs = set(cfg.translatable_attributes)
        fingerprint_before = doc.fingerprint(translatable_attrs=translatable_attrs)
        excluded_ids = doc.excluded_text_node_ids(
            cfg.excluded_tags, cfg.excluded_classes
        )

        # --- collect + segment ---
        blocks = collect_blocks(
            doc,
            excluded_tags=cfg.excluded_tags,
            excluded_classes=cfg.excluded_classes,
        )
        # Token measurement uses the EXACT tokenizer already loaded by the
        # injected translator (authoritative cache policy, revision, and
        # resolved snapshot; truncation=False). The model loads lazily on
        # the first measurement and is then reused for inference — no
        # second tokenizer copy, no independent Hugging Face access.
        def measure(text: str) -> int:
            return self._translator.measure_source_tokens(text, source_lang)
        try:
            segments = segment_blocks(
                doc,
                blocks,
                measure=measure,
                max_segment_tokens=cfg.max_segment_tokens,
                document_id=self._document_id,
                source_language=source_lang,
                target_language=target_lang,
                preserve_patterns=tuple(
                    re.compile(p) for p in cfg.preserve_patterns
                ),
                translatable_attrs=set(cfg.translatable_attributes),
            )
        except ValueError as e:
            raise StructuredTranslationError(str(e)) from e

        # --- translatable attributes become segments too ---
        attr_segments = self._collect_attribute_segments(
            doc, translatable_attrs, measure, source_lang, target_lang
        )
        all_segments = segments + attr_segments

        protected_run_count = sum(
            1 for s in all_segments for r in s.runs if not r.translate
        )

        logger.info(
            "[STRUCTURED] correlation=%s segments=%d attr_segments=%d blocks=%d "
            "source_chars=%d src=%s tgt=%s protected_runs=%d",
            correlation_id, len(segments), len(attr_segments), len(blocks),
            len(html), source_lang, target_lang, protected_run_count,
        )

        # --- translate segments in TRUE bounded batches (first pass) ---
        # Segments are grouped into batches (configurable batch_size,
        # source order preserved) and each batch is sent to the shared
        # translator's translate_batch_texts() ONCE. Groups are formed by
        # (language pair, quantized target-budget bucket) so short segments
        # never run with an unnecessarily large max_new_tokens: the group's
        # generation budget is the documented bucket, never below any
        # member's required budget (never lowered, never truncated).
        # Group order is first-seen within the chunk; every segment object
        # retains its own sequence index and outputs are restored onto the
        # ORIGINAL segment objects, so no correctness mechanism relies on
        # dictionary/group iteration order. Every result is validated
        # independently with the strict complete placeholder sequence;
        # only failed items are retried individually (stricter prefix),
        # then split fallback, then fail closed. Batch cardinality is
        # checked explicitly after every batch call — a mismatch is
        # recovered per segment or fails closed; results are never
        # silently zipped away. No concurrent model calls.
        retry_count = 0
        fallback_count = 0
        total_source_tokens = 0
        total_target_tokens = 0
        batch_generation_budget = 0
        sum_requested_target_tokens = 0
        batch_metrics: List[dict] = []
        batch_count = 0
        batch_size = cfg.batch_size

        for start_idx in range(0, len(all_segments), batch_size):
            if time.monotonic() > deadline:
                raise StructuredTranslationError(
                    f"deadline exceeded (max_total_seconds="
                    f"{cfg.max_total_seconds}); request aborted between "
                    f"batches; no partial output returned"
                )
            chunk = all_segments[start_idx : start_idx + batch_size]
            for seg in chunk:
                total_source_tokens += seg.token_count

            # Per-segment REQUIRED target budgets (never lowered) and the
            # documented quantized bucket actually passed to generation.
            budgets = {
                id(seg): self._target_budget(seg.token_count)
                for seg in chunk
            }
            buckets = {
                id(seg): _budget_bucket(budgets[id(seg)])
                for seg in chunk
            }
            sum_requested_target_tokens += sum(budgets.values())

            # Group by (language pair, budget bucket) in first-seen order.
            # Segments keep their sequence_index; outputs are restored onto
            # the original segment objects afterwards. No correctness
            # mechanism relies on group iteration order.
            groups: List[tuple] = []
            group_key_index: Dict[tuple, int] = {}
            for seg in chunk:
                key = (
                    (seg.source_language, seg.target_language),
                    buckets[id(seg)],
                )
                if key not in group_key_index:
                    group_key_index[key] = len(groups)
                    groups.append(
                        (seg.source_language, seg.target_language,
                         buckets[id(seg)], [])
                    )
                groups[group_key_index[key]][3].append(seg)

            for src_lang, tgt_lang, group_budget, group_segments in groups:
                # Each budget group is one model batch call.
                batch_count += 1
                batch_start = time.monotonic()
                texts = [s.source_text for s in group_segments]
                source_tokens_in_group = sum(s.token_count for s in group_segments)
                batch_generation_budget += group_budget * len(group_segments)

                try:
                    outputs = self._call_model(
                        texts, src_lang, tgt_lang, group_budget, deadline
                    )
                    item_outputs: Optional[dict] = {
                        i: out for i, out in enumerate(outputs)
                    }
                except BatchItemError as e:
                    # Malformed item(s): the bad indices and the validated
                    # outputs of the GOOD items are carried by the error, so
                    # only the affected inputs are recovered individually —
                    # successful neighbors are never re-sent.
                    logger.warning(
                        "batch item(s) %s malformed for correlation=%s "
                        "batch=%d; recovering those item(s) individually, "
                        "preserving %d valid neighbor(s)",
                        e.bad_indices, correlation_id, batch_count,
                        len(e.valid_outputs),
                    )
                    item_outputs: Optional[dict] = {
                        i: e.valid_outputs.get(i)
                        for i in range(len(group_segments))
                    }
                except StructuredTranslationError:
                    # The whole batch call failed (model crash or deadline):
                    # every segment of the batch is retried individually.
                    logger.warning(
                        "batch call failed for correlation=%s batch=%d; "
                        "retrying %d segment(s) individually",
                        correlation_id, batch_count, len(group_segments),
                    )
                    item_outputs = None

                if item_outputs is not None and len(item_outputs) != len(group_segments):
                    # Explicit cardinality invariant: never zip away missing
                    # or surplus results. Recover every affected segment
                    # individually; if recovery cannot prove each segment
                    # translated exactly once, the request fails closed.
                    logger.warning(
                        "batch result count %d does not match input count %d "
                        "for correlation=%s batch=%d; recovering each "
                        "segment individually",
                        len(item_outputs), len(group_segments),
                        correlation_id, batch_count,
                    )
                    item_outputs = None

                if item_outputs is None:
                    pending = [(seg, None) for seg in group_segments]
                else:
                    pending = []
                    for idx, seg in enumerate(group_segments):
                        out = item_outputs.get(idx)
                        if out is None:
                            # This item's batch result was missing or
                            # malformed: recovery starts a fresh call.
                            pending.append((seg, None))
                            continue
                        check = seg.protected_map.validate_output(
                            out, expected_order=seg.placeholder_order
                        )
                        if check["ok"]:
                            seg.translated_text = out
                            total_target_tokens += budgets[id(seg)]
                            logger.info(
                                "[STRUCTURED] correlation=%s seg=%s tokens=%d "
                                "target_budget=%d (batch=%d)",
                                correlation_id, seg.segment_id, seg.token_count,
                                budgets[id(seg)], batch_count,
                            )
                        else:
                            # First-pass attempt failed validation: recovery
                            # must treat this output as the failed attempt 1.
                            pending.append((seg, out))

                for seg, failed_out in pending:
                    if time.monotonic() > deadline:
                        raise StructuredTranslationError(
                            f"deadline exceeded (max_total_seconds="
                            f"{cfg.max_total_seconds}); request aborted "
                            f"during retry/fallback of {seg.segment_id}; "
                            f"no partial output returned"
                        )
                    ok, translated, retried, fell_back = self._recover_segment(
                        seg, budgets[id(seg)], deadline, first_text=failed_out
                    )
                    retry_count += retried
                    fallback_count += fell_back
                    if not ok:
                        raise StructuredTranslationError(
                            f"segment {seg.segment_id} could not be translated "
                            f"safely after retries and split fallback"
                        )
                    seg.translated_text = translated
                    total_target_tokens += budgets[id(seg)]

                batch_metrics.append({
                    "batch_index": batch_count,
                    "items": len(group_segments),
                    "max_target_budget": group_budget,
                    "per_segment_budgets": [budgets[id(s)] for s in group_segments],
                    "per_segment_buckets": [buckets[id(s)] for s in group_segments],
                    "source_tokens": source_tokens_in_group,
                    "elapsed_seconds": round(time.monotonic() - batch_start, 3),
                })

        # --- machine-readable metrics (computed on the ORIGINAL document) ---
        repeated_terms = _collect_repeated_terms(doc, excluded_ids)

        # --- reconstruct + validate ---
        translated_html = rebuild_document(
            doc,
            segments,
            fingerprint_before,
            attr_segments=attr_segments,
            excluded_tags=cfg.excluded_tags,
            excluded_classes=cfg.excluded_classes,
            translatable_attrs=translatable_attrs,
        )

        # --- machine-readable metrics ---
        identifier_occurrences = _collect_identifier_occurrences(all_segments)
        metrics = {
            "identifiers": identifier_occurrences,
            "repeated_terms": repeated_terms,
        }

        duration = time.monotonic() - start
        logger.info(
            "[STRUCTURED] correlation=%s done segments=%d batches=%d "
            "retries=%d fallbacks=%d duration=%.2fs",
            correlation_id, len(all_segments), batch_count,
            retry_count, fallback_count, duration,
        )

        return StructuredTranslationResult(
            translated_html=translated_html,
            correlation_id=correlation_id,
            segment_count=len(all_segments),
            total_source_tokens=total_source_tokens,
            total_target_tokens=total_target_tokens,
            retry_count=retry_count,
            fallback_count=fallback_count,
            batch_count=batch_count,
            batch_generation_budget=batch_generation_budget,
            sum_requested_target_tokens=sum_requested_target_tokens,
            batch_metrics=batch_metrics,
            protected_run_count=protected_run_count,
            duration_seconds=duration,
            fingerprint_ok=True,
            excluded_text_nodes=len(excluded_ids),
            translated_attributes=len(attr_segments),
            source_language=source_lang,
            target_language=target_lang,
            segments=[s.to_dict() for s in all_segments],
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Translatable attributes
    # ------------------------------------------------------------------

    def _collect_attribute_segments(
        self,
        doc: HTMLDocument,
        translatable_attrs: set,
        measure,
        source_lang: str,
        target_lang: str,
    ) -> List[Segment]:
        """Attribute values (allowlist only) with CJK content -> segments.

        URL/code/style attributes are never included: they are not in the
        allowlist. Attribute translations get stable element ids, token
        budgets, placeholder protection, and reconstruction validation.
        """
        if not translatable_attrs:
            return []
        segments: List[Segment] = []
        seq = 0
        for elem in doc.element_nodes():
            if elem.id == "#root":
                continue
            for name, value in elem.attrs:
                if name not in translatable_attrs:
                    continue
                if classify(value) == LanguageKind.ENGLISH:
                    continue
                pmap = ProtectionMap()
                protected = protect_identifiers(value, pmap)
                run_id = f"attr:{elem.id}:{name}"
                token_count = measure(protected)
                if token_count > self._config.max_segment_tokens:
                    raise StructuredTranslationError(
                        f"attribute {name} value of element {elem.id} exceeds "
                        f"max_segment_tokens ({token_count} > "
                        f"{self._config.max_segment_tokens}); attributes are "
                        f"not hard-split"
                    )
                run = _make_run(
                    node_id=run_id,
                    kind=RUN_ATTRIBUTE,
                    raw=value,
                    protected=protected,
                    translate=True,
                )
                run.slot_index = 0
                # The attribute value's protected identifiers are recorded
                # in placeholder_order (protect_identifiers reserves in
                # document order), so strict validation covers them: a model
                # that drops/duplicates/reorders/invents them fails.
                attr_placeholder_order = pmap.tokens
                seg = Segment(
                    document_id=self._document_id,
                    segment_id=f"attr:{self._document_id}:{seq:04d}",
                    sequence_index=seq,
                    source_node_ids=[run_id],
                    source_text=protected,
                    protected_map=pmap,
                    source_language=source_lang,
                    target_language=target_lang,
                    token_count=token_count,
                    runs=[run],
                    placeholder_order=attr_placeholder_order,
                    slots=[run],
                    block_key=f"attr{seq:04d}",
                    block_text=value,
                )
                segments.append(seg)
                seq += 1
        return segments

    # ------------------------------------------------------------------

    def _target_budget(self, source_tokens: int) -> int:
        """Safe per-segment target budget (configurable, never assumed 256)."""
        cfg = self._config
        return min(cfg.max_target_tokens, max(64, int(source_tokens * 2.5)))

    def _translate_segment(
        self, seg: Segment, target_budget: int, deadline: float
    ) -> tuple:
        """Translate one segment with its own model call (single path).

        Backward-compatible wrapper kept for callers outside the batched
        first pass: makes one model call for the segment, then recovers via
        retry/fallback if validation fails.

        Returns (ok, translated_text, retry_count, fallback_count).
        """
        src_lang = seg.source_language
        tgt_lang = seg.target_language
        text = self._call_model(
            [seg.source_text], src_lang, tgt_lang, target_budget, deadline
        )[0]
        return self._recover_segment(
            seg, target_budget, deadline, first_text=text
        )

    def _recover_segment(
        self,
        seg: Segment,
        target_budget: int,
        deadline: float,
        first_text: Optional[str] = None,
    ) -> tuple:
        """Validate ``first_text`` (or make a fresh single call), then
        retry with a stricter placeholder prefix, then split fallback.

        Used by the batched first pass for items that failed validation and
        for whole-batch call failures. Successful items are never re-sent.

        Returns (ok, translated_text, retry_count, fallback_count).
        """
        text = first_text
        if text is None:
            text = self._call_model(
                [seg.source_text], seg.source_language, seg.target_language,
                target_budget, deadline,
            )[0]
        check = seg.protected_map.validate_output(
            text, expected_order=seg.placeholder_order
        )
        if check["ok"]:
            return True, text, 0, 0

        logger.debug("segment %s validation issues: %s", seg.segment_id, check["issues"])

        # Retry: stricter placeholder representation (fresh random prefix)
        for attempt in range(1, self._config.max_retries_per_segment + 1):
            new_text = self._retry_stricter_prefix(seg, target_budget, deadline)
            if new_text is not None:
                return True, new_text, attempt, 0

        # Split fallback: translate each chinese run alone (protected runs
        # and tags are re-interleaved verbatim)
        translated = self._split_fallback(seg, target_budget, deadline)
        if translated is not None:
            return True, translated, self._config.max_retries_per_segment, 1

        return False, "", self._config.max_retries_per_segment, 1

    def _retry_stricter_prefix(
        self, seg: Segment, target_budget: int, deadline: float
    ) -> Optional[str]:
        """Re-translate with a fresh random placeholder prefix.

        Deterministic models may reproduce the same output; this is a
        documented, cheap stricter-representation attempt.
        """
        import random
        import string

        new_prefix = "__IT" + "".join(
            random.choices(string.ascii_uppercase + string.digits, k=4)
        ) + "_"
        if new_prefix in seg.source_text:
            return None

        remapped = {}
        for token in seg.protected_map.tokens:
            remapped[token] = new_prefix + token[len(DEFAULT_PREFIX):]

        new_source = seg.source_text
        new_pmap = ProtectionMap(prefix=new_prefix)
        for old, new in remapped.items():
            new_source = new_source.replace(old, new)
            span = seg.protected_map.span(old)
            new_pmap._spans[new] = span.__class__(new, span.content, span.kind)
            new_pmap._counter += 1

        text = self._call_model(
            [new_source], seg.source_language, seg.target_language,
            target_budget, deadline,
        )[0]
        check = new_pmap.validate_output(
            text, expected_order=[remapped[t] for t in seg.placeholder_order]
        )
        if not check["ok"]:
            logger.debug(
                "stricter-prefix retry for %s still failing: %s",
                seg.segment_id, check["issues"],
            )
            return None
        for old, new in remapped.items():
            text = text.replace(new, old)
        return text
    def _split_fallback(
        self, seg: Segment, target_budget: int, deadline: float
    ) -> Optional[str]:
        """Translate each chinese run independently (no surrounding
        placeholders); protected runs and tags are re-interleaved verbatim.

        The output is built in source layout order: translate runs consume a
        model call each, protected runs restore their ORIGINAL raw text.
        """
        pieces: List[str] = []
        for run in seg.runs:
            if not run.translate:
                continue  # tags/protected runs restored from raw
            if not run.raw.strip():
                # Whitespace-only translate run (edge-whitespace carrier):
                # nothing to translate — the layout restores its raw spacing
                # verbatim (edge-whitespace normalization).
                pieces.append(run.raw)
                continue
            pmap = ProtectionMap()
            protected = protect_identifiers(run.raw, pmap)
            text = self._call_model(
                [protected], seg.source_language, seg.target_language,
                target_budget, deadline,
            )[0]
            check = pmap.validate_output(text)
            if not check["ok"]:
                return None
            pieces.append(pmap.restore(text))

        # Rebuild in layout order: slot pieces interleaved with placeholders
        piece_idx = 0
        result = ""
        for run in seg.runs:
            if run.translate:
                result += pieces[piece_idx]
                piece_idx += 1
            else:
                result += run.protected  # placeholder token (restored later)
        return result

    def _call_model(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        target_budget: int,
        deadline: float,
    ) -> List[str]:
        """Call the shared translator's batch interface (never HTTP).

        The REQUEST source/target languages are propagated to the model;
        forced_bos_token_id is resolved from target_lang by the engine.
        """
        if not texts:
            return []
        if time.monotonic() > deadline:
            raise StructuredTranslationError(
                f"deadline exceeded (max_total_seconds="
                f"{self._config.max_total_seconds}); request aborted"
            )
        start = time.monotonic()
        try:
            results = self._translator.translate_batch_texts(
                texts,
                source_lang=source_lang,
                target_lang=target_lang,
                max_new_tokens=target_budget,
            )
        except TranslationQualityError:
            raise
        except Exception as e:
            logger.exception("model call failed: %s", e)
            raise StructuredTranslationError("model call failed") from e
        # Shared-boundary validation: rejects None / scalar / non-sized
        # collections (no raw TypeError/AttributeError escapes), validates
        # the exact result count BEFORE any zip()/indexing, and validates
        # every item's translated_text is a non-null string — malformed
        # items raise BatchItemError carrying the good neighbors' outputs.
        out: List[str] = _validate_batch_results(results, len(texts))
        elapsed = time.monotonic() - start
        if elapsed > self._config.segment_warning_seconds:
            logger.warning(
                "segment batch took %.1fs (warning threshold %.1fs)",
                elapsed, self._config.segment_warning_seconds,
            )
        return out


def _make_run(node_id: str, kind: str, raw: str, protected: str,
              translate: bool) -> "Run":
    from .chapter_chunking import Run
    return Run(
        node_id=node_id,
        kind=kind,
        raw=raw,
        protected=protected,
        translate=translate,
        offset_end=len(raw),
    )


def translate_html(
    html: str,
    translator: Translator,
    structured_config: StructuredConfig,
    translation_config: Optional[TranslationConfig] = None,
    source_lang: str = "zh",
    target_lang: str = "en",
    document_id: str = "doc",
) -> StructuredTranslationResult:
    """Convenience one-call entry point for the structured path."""
    st = StructuredTranslator(
        translator, structured_config, translation_config, document_id=document_id
    )
    return st.translate(html, source_lang=source_lang, target_lang=target_lang)
