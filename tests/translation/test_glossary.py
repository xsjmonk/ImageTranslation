"""Terminology memory (glossary) tests — chapter-scoped terminology map."""

from __future__ import annotations

import re

import pytest

from image_translation.translation.chapter_chunking import find_glossary_spans
from image_translation.translation.config import GlossaryEntry, StructuredConfig
from image_translation.translation.exceptions import StructuredTranslationError
from image_translation.translation.structured_translation import (
    StructuredTranslator,
)

from tests.translation.test_one_round_revision import FakeTranslator


class TestGlossaryConfig:
    def test_empty_source_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            GlossaryEntry("", "Charger")

    def test_empty_target_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            GlossaryEntry("充电器", "  ")

    def test_identity_rejected(self):
        with pytest.raises(ValueError, match="must differ"):
            GlossaryEntry("充电器", "充电器")

    def test_duplicate_sources_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            StructuredConfig(glossary=(
                GlossaryEntry("充电器", "Charger"),
                GlossaryEntry("充电器", "Charger2"),
            ))

    def test_overlapping_terms_rejected(self):
        with pytest.raises(ValueError, match="overlap"):
            StructuredConfig(glossary=(
                GlossaryEntry("充电", "Charge"),
                GlossaryEntry("充电器", "Charger"),
            ))

    def test_invalid_entry_type_rejected(self):
        with pytest.raises(ValueError, match="GlossaryEntry"):
            StructuredConfig(glossary=(("充电器", "Charger"),))


class TestBoundaryPolicy:
    def test_latin_embedding_not_matched_exact(self):
        """exact=True: 'cat' must not match inside 'catalog'."""
        text = "catalog category cat"
        spans = find_glossary_spans(text, (GlossaryEntry("cat", "FELINE"),))
        # only the standalone 'cat' matches
        assert len(spans) == 1
        start, end, _entry = spans[0]
        assert text[start:end] == "cat"

    def test_cjk_neighbors_accepted(self):
        """Chinese has no spaces: 充电器 in '使用充电器。' matches."""
        spans = find_glossary_spans(
            "使用充电器。", (GlossaryEntry("充电器", "Charger"),)
        )
        assert len(spans) == 1

    def test_exact_false_matches_inside_words(self):
        spans = find_glossary_spans(
            "catalog", (GlossaryEntry("cat", "FELINE", exact=False),)
        )
        assert len(spans) == 1

    def test_boundary_at_text_edges(self):
        spans = find_glossary_spans(
            "充电器", (GlossaryEntry("充电器", "Charger"),)
        )
        assert len(spans) == 1

    def test_punctuation_boundaries(self):
        spans = find_glossary_spans(
            "（充电器）'充电器'，充电器。",
            (GlossaryEntry("充电器", "Charger"),),
        )
        assert len(spans) == 3


class TestGlossaryTranslation:
    def test_term_consistent_across_segments(self):
        """The same configured term maps to the same target in EVERY
        segment — consistent by construction."""
        fake = FakeTranslator()
        cfg = StructuredConfig(max_segment_tokens=60, glossary=(
            GlossaryEntry("充电器", "Charger"),
        ))
        # 12 paragraphs -> multiple segments; the term appears twice in each
        html = "".join(
            f"<p>第{i}段：本充电器支持快充，充电器需要定期维护。</p>"
            for i in range(12)
        )
        res = StructuredTranslator(fake, cfg, None).translate(html)
        out = res.translated_html
        assert res.segment_count > 1
        # every occurrence became the exact target
        assert out.count("Charger") == 12 * 2
        assert "充电器" not in out

    def test_terminology_metrics_recorded(self):
        fake = FakeTranslator()
        cfg = StructuredConfig(glossary=(
            GlossaryEntry("充电器", "Charger"),
            GlossaryEntry("防水面料", "Waterproof Fabric"),
        ))
        html = (
            "<p>充电器采用防水面料。</p>"
            "<p>充电器兼容型号 ABC-123。</p>"
            "<p>Visit https://example.com 获取充电器信息。</p>"
        )
        res = StructuredTranslator(fake, cfg, None).translate(html)
        term = res.to_dict()["terminology"]["glossary"]
        assert term["充电器"]["target"] == "Charger"
        assert term["充电器"]["occurrences"] == 3
        assert len(term["充电器"]["segments"]) >= 1
        assert term["防水面料"]["occurrences"] == 1
        # identifiers recorded too
        ids = res.to_dict()["terminology"]["identifiers"]
        assert "ABC-123" in ids
        assert any("https://" in k for k in ids)
        # machine-readable result has all required fields
        d = res.to_dict()
        for key in ("segment_count", "total_source_tokens", "total_target_tokens",
                    "protected_run_count", "terminology", "duration_seconds",
                    "retry_count", "fallback_count", "validation"):
            assert key in d, f"missing metric {key}"

    def test_consistency_validation_fails_closed(self, monkeypatch):
        """If a glossary target is lost from the translated nodes, the
        request fails (even though the serialized string is fine)."""
        fake = FakeTranslator()
        cfg = StructuredConfig(glossary=(GlossaryEntry("充电器", "Charger"),))
        with monkeypatch.context() as m:
            from image_translation.translation import structured_translation as st
            orig = st.rebuild_document
            def broken(doc, *a, **k):
                out = orig(doc, *a, **k)
                # lose the glossary target inside the TRANSLATED node only
                for node in doc.text_nodes():
                    if "EN:" in node.text:  # translated node (fake wraps CJK)
                        node.text = node.text.replace("Charger", "X", 1)
                return out
            m.setattr(st, "rebuild_document", broken)
            with pytest.raises(StructuredTranslationError, match="terminology"):
                StructuredTranslator(fake, cfg, None).translate(
                    "<p>充电器支持快充。</p>"
                )

    def test_excluded_target_text_cannot_mask_lost_occurrence(self, monkeypatch):
        """Node-scoped validation: the target term pre-existing in EXCLUDED
        HTML must not mask a lost glossary occurrence (a global substring
        count would pass; the node-scoped count still fails)."""
        fake = FakeTranslator()
        cfg = StructuredConfig(glossary=(GlossaryEntry("充电器", "Charger"),))
        # excluded block already contains the target term "Charger"
        html = (
            "<p>充电器支持快充。</p>"
            "<div class='notranslate'>Charger 保持不变</div>"
        )
        with monkeypatch.context() as m:
            from image_translation.translation import structured_translation as st
            orig = st.rebuild_document
            def broken(doc, *a, **k):
                out = orig(doc, *a, **k)
                # lose the glossary occurrence ONLY in the translated node;
                # the excluded div keeps its "Charger" (would mask a global
                # substring count)
                for node in doc.text_nodes():
                    if "EN:" in node.text:  # translated node (fake wraps CJK)
                        node.text = node.text.replace("Charger", "X", 1)
                return out
            m.setattr(st, "rebuild_document", broken)
            with pytest.raises(StructuredTranslationError, match="terminology"):
                StructuredTranslator(fake, cfg, None).translate(html)

    def test_untouched_english_target_lookalike_cannot_mask_loss(self, monkeypatch):
        """A target-looking string in UNTOUCHED English (an all-English
        block that is never sent to the model) must not mask a lost glossary
        occurrence in the translated nodes."""
        fake = FakeTranslator()
        cfg = StructuredConfig(glossary=(GlossaryEntry("充电器", "Charger"),))
        html = (
            "<p>充电器支持快充。</p>"
            "<p>The Charger is ready for shipping.</p>"  # all-English block
        )
        with monkeypatch.context() as m:
            from image_translation.translation import structured_translation as st
            orig = st.rebuild_document
            def broken(doc, *a, **k):
                out = orig(doc, *a, **k)
                # lose the glossary occurrence ONLY in the translated node;
                # the untouched English block keeps its "Charger" lookalike
                for node in doc.text_nodes():
                    if "EN:" in node.text:  # translated node (fake wraps CJK)
                        node.text = node.text.replace("Charger", "X", 1)
                return out
            m.setattr(st, "rebuild_document", broken)
            with pytest.raises(StructuredTranslationError, match="terminology"):
                StructuredTranslator(fake, cfg, None).translate(html)

    def test_latin_term_not_corrupted_in_english_text(self):
        """exact=True protects latin words: glossary 'USB' must not rewrite
        'USB-C' as 'Universal Serial Bus-C'."""
        fake = FakeTranslator()
        cfg = StructuredConfig(glossary=(GlossaryEntry("USB", "Universal Serial Bus"),))
        res = StructuredTranslator(fake, cfg, None).translate(
            "<p>通过 USB-C 接口连接。USB 支持快充。</p>"
        )
        out = res.translated_html
        assert "USB-C" in out          # untouched (embedded in a longer token)
        assert "Universal Serial Bus" in out  # standalone USB replaced
