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
6. batch translation through the existing translator (never HTTP), with the
   REQUEST source/target languages propagated to every model call;
7. per-segment placeholder validation (every placeholder exactly once, in
   source order) with retry (fresh placeholder namespace) and per-chinese-run
   split fallback;
8. ordered reconstruction from run metadata (never string splitting) with
   fail-closed fingerprint/excluded checks.

Protected-span strategy (documented): ordinary English and identifiers are
replaced with collision-resistant placeholders BEFORE inference and restored
from the ORIGINAL text afterwards. The model can never rewrite, drop, or
paraphrase English — exact preservation does not depend on post-hoc
detection. Context is NOT supplied to the model: M2M100's generate() has no
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
    RUN_GLOSSARY,
    RUN_IDENTIFIER,
    RUN_TAG,
    Segment,
    collect_blocks,
    segment_blocks,
)
from .config import GlossaryEntry, StructuredConfig, TranslationConfig
from .exceptions import StructuredTranslationError
from .html_document import HTMLDocument
from .html_protection import DEFAULT_PREFIX, ProtectionMap, assert_prefix_safe
from .language_segments import LanguageKind, classify, protect_identifiers
from .reconstruction import rebuild_document

logger = logging.getLogger(__name__)

_tokenizer_cache: Dict[tuple, object] = {}

# CJK ideographs only (for repeated-term reporting)
_CJK_GRAM_RE = re.compile(r"^[\u4e00-\u9fff]+$")


def _collect_repeated_terms(
    doc: HTMLDocument, excluded_ids: set, min_count: int = 3, top_n: int = 20
) -> Dict[str, int]:
    """Repeated CJK bigrams/trigrams in TRANSLATABLE text (informational).

    Excluded content is never scanned. Reported terms are NOT replaced —
    only configured glossary entries drive replacement.
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


def _collect_glossary_occurrences(segments: List[Segment]) -> dict:
    """Terminology memory record: glossary term -> {target, exact,
    occurrences, segments[]}. Grouped by run.raw (the source term)."""
    result: Dict[str, dict] = {}
    for seg in segments:
        for run in seg.runs:
            if run.kind != RUN_GLOSSARY:
                continue
            info = result.setdefault(
                run.raw,
                {
                    "target": run.restore_text,
                    "exact": True,
                    "occurrences": 0,
                    "segments": [],
                },
            )
            info["occurrences"] += 1
            if seg.segment_id not in info["segments"]:
                info["segments"].append(seg.segment_id)
    return result


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


def _get_measure_tokenizer(model_name: str, model_cache_dir: Optional[str]):
    """Lazily load the tokenizer used ONLY for token measurement."""
    key = (model_name, model_cache_dir)
    if key not in _tokenizer_cache:
        from transformers import M2M100Tokenizer
        kwargs = {}
        if model_cache_dir:
            kwargs["cache_dir"] = model_cache_dir
        _tokenizer_cache[key] = M2M100Tokenizer.from_pretrained(model_name, **kwargs)
    return _tokenizer_cache[key]


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
            "protected_run_count": self.protected_run_count,
            "terminology": self.metrics.get("terminology", {}),
            "repeated_terms": self.metrics.get("repeated_terms", {}),
            "duration_seconds": round(self.duration_seconds, 3),
            "retry_count": self.retry_count,
            "fallback_count": self.fallback_count,
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
        glossary = list(cfg.glossary)
        measure = lambda text: len(
            _get_measure_tokenizer(
                self._translation_config.model_name,
                self._translation_config.model_cache_dir,
            )(text, truncation=False)["input_ids"]
        )
        try:
            segments = segment_blocks(
                doc,
                blocks,
                measure=measure,
                max_segment_tokens=cfg.max_segment_tokens,
                document_id=self._document_id,
                source_language=source_lang,
                target_language=target_lang,
                glossary=glossary,
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

        # --- translate segments in batches ---
        retry_count = 0
        fallback_count = 0
        total_source_tokens = 0
        total_target_tokens = 0

        batch_size = 4
        for start_idx in range(0, len(all_segments), batch_size):
            chunk = all_segments[start_idx : start_idx + batch_size]
            for seg in chunk:
                if time.monotonic() > deadline:
                    raise StructuredTranslationError(
                        f"deadline exceeded (max_total_seconds="
                        f"{cfg.max_total_seconds}); request aborted between "
                        f"segments; no partial output returned"
                    )
                total_source_tokens += seg.token_count
                target_budget = self._target_budget(seg.token_count)

                ok, translated, retried, fell_back = self._translate_segment(
                    seg, target_budget, deadline
                )
                retry_count += retried
                fallback_count += fell_back
                if not ok:
                    raise StructuredTranslationError(
                        f"segment {seg.segment_id} could not be translated "
                        f"safely after retries and split fallback"
                    )
                seg.translated_text = translated
                total_target_tokens += target_budget
                logger.info(
                    "[STRUCTURED] correlation=%s seg=%s tokens=%d target_budget=%d",
                    correlation_id, seg.segment_id, seg.token_count, target_budget,
                )

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

        # --- terminology consistency validation (node-scoped) ---
        # Every configured glossary term must have been replaced by its
        # target term wherever it occurred in TRANSLATABLE content. Counts
        # are scoped to the translated text nodes ONLY — pre-existing target
        # text or the source term in excluded/untouched HTML can never mask
        # a lost glossary occurrence (a global substring count could).
        glossary_occurrences = _collect_glossary_occurrences(all_segments)
        translatable_node_ids = {
            r.node_id
            for s in all_segments
            for r in s.runs
            if r.node_id != "tag"
        }
        for term, info in glossary_occurrences.items():
            target = info["target"]
            in_translated = sum(
                node.text.count(target)
                for node in doc.text_nodes()
                if node.id in translatable_node_ids
            )
            if in_translated < info["occurrences"]:
                raise StructuredTranslationError(
                    f"terminology consistency check failed: term {term!r} "
                    f"mapped to {target!r} in {info['occurrences']} "
                    f"occurrences, but the translated text nodes contain it "
                    f"only {in_translated} times"
                )

        # --- machine-readable metrics ---
        identifier_occurrences = _collect_identifier_occurrences(all_segments)
        metrics = {
            "terminology": {
                "glossary": glossary_occurrences,
                "identifiers": identifier_occurrences,
            },
            "repeated_terms": repeated_terms,
        }

        duration = time.monotonic() - start
        logger.info(
            "[STRUCTURED] correlation=%s done segments=%d duration=%.2fs",
            correlation_id, len(all_segments), duration,
        )

        return StructuredTranslationResult(
            translated_html=translated_html,
            correlation_id=correlation_id,
            segment_count=len(all_segments),
            total_source_tokens=total_source_tokens,
            total_target_tokens=total_target_tokens,
            retry_count=retry_count,
            fallback_count=fallback_count,
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
                    placeholder_order=[],
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
        """Translate one segment with validation, retries, and fallback.

        Returns (ok, translated_text, retry_count, fallback_count).
        """
        src_lang = seg.source_language
        tgt_lang = seg.target_language

        # Attempt 1: normal path (placeholders as built)
        text = self._call_model(
            [seg.source_text], src_lang, tgt_lang, target_budget, deadline
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
        except Exception as e:
            logger.exception("model call failed: %s", e)
            raise StructuredTranslationError("model call failed") from e
        elapsed = time.monotonic() - start
        if elapsed > self._config.segment_warning_seconds:
            logger.warning(
                "segment batch took %.1fs (warning threshold %.1fs)",
                elapsed, self._config.segment_warning_seconds,
            )
        return [r.translated_text for r in results]


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
