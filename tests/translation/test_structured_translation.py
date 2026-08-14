"""End-to-end structured translation tests with a fake translator (no GPU)."""

from __future__ import annotations

import re

import pytest

from image_translation.translation.base import Translator
from image_translation.translation.config import StructuredConfig, TranslationConfig
from image_translation.translation.exceptions import StructuredTranslationError
from image_translation.translation.models import TranslationResult
from image_translation.translation.structured_translation import (
    StructuredTranslator,
    translate_html,
)

import image_translation.translation.structured_translation as st_mod


class FakeTranslator(Translator):
    """Emulates the model: prefixes 'EN:', preserves placeholders.

    drop_tag_calls: call indices (1-based) on which ALL placeholders are
    dropped, to exercise the retry/fallback paths.
    """

    def __init__(self, drop_tag_calls=()):
        self.call_count = 0
        self.drop_tag_calls = set(drop_tag_calls)

    @property
    def name(self) -> str:
        return "fake"

    @property
    def runtime_info(self):
        return None


    def measure_source_tokens(self, text: str, source_lang: str = "zh") -> int:
        """Token count used by HTML segmentation (no model call)."""
        return max(1, (len(text) + 1) // 2)
    def translate_text(self, text, source_lang="zh", target_lang="en", max_new_tokens=None):
        return self.translate_batch_texts(
            [text], source_lang, target_lang, max_new_tokens
        )[0]

    def translate_batch_texts(
        self, texts, source_lang="zh", target_lang="en", max_new_tokens=None
    ):
        out = []
        for t in texts:
            self.call_count += 1
            translated = re.sub(
                r"[\u4e00-\u9fff]+", lambda m: "EN:" + m.group(0), t
            )
            if self.call_count in self.drop_tag_calls:
                # Drop ALL placeholder tokens, including retry prefixes
                translated = re.sub(r"__IT[A-Z0-9]*_[A-Z]\d{4}_", "", translated)
            out.append(
                TranslationResult(
                    source_text=t, translated_text=translated,
                    model_name="fake", device="cpu",
                )
            )
        return out


@pytest.fixture(autouse=True)
def fake_measure(monkeypatch):
    """Keep unit tests hermetic: no HF tokenizer/network."""
    class FakeTok:
        def __call__(self, text, truncation=False):
            return {"input_ids": list(range(max(1, (len(text) + 1) // 2)))}
    monkeypatch.setattr(st_mod, "_get_measure_tokenizer", lambda *a, **k: FakeTok())


MIXED_HTML = """<h1>产品介绍</h1>
<p>这是一款 <strong>加厚防水面料</strong> 制作的背包，适合 daily use。</p>
<p>Visit https://example.com/x?q=1 for details。USB-C 接口支持快充。</p>
<div class="notranslate">保持不变 ABC-123</div>
<script>var x = "中文";</script>
<p>English only paragraph stays.</p>
"""


class TestStructuredTranslate:
    def test_mixed_html(self):
        fake = FakeTranslator()
        res = translate_html(MIXED_HTML, fake, StructuredConfig(), TranslationConfig())
        out = res.translated_html

        assert res.fingerprint_ok
        assert res.segment_count >= 3
        assert res.retry_count == 0
        assert res.fallback_count == 0

        # Structure preserved
        assert out.count("<h1>") == 1 and out.count("</h1>") == 1
        assert "<strong>" in out and "</strong>" in out
        # Excluded content untouched
        assert "保持不变 ABC-123" in out
        assert 'var x = "中文"' in out
        # English-only untouched
        assert "English only paragraph stays." in out
        # Protected identifiers survive
        assert "https://example.com/x?q=1" in out
        assert "USB-C" in out
        # Translational text was sent to the model (EN: prefix)
        assert "EN:产品介绍" in out

    def test_plain_text_input_works(self):
        fake = FakeTranslator()
        res = translate_html("加厚防水面料，耐磨耐用。", fake, StructuredConfig(), TranslationConfig())
        assert res.translated_html == "EN:加厚防水面料，EN:耐磨耐用。"
        assert res.segment_count == 1

    def test_empty_html_ok(self):
        fake = FakeTranslator()
        res = translate_html("", fake, StructuredConfig(), TranslationConfig())
        assert res.translated_html == ""

    def test_chapter_size_limit(self):
        fake = FakeTranslator()
        cfg = StructuredConfig(max_chapter_characters=50)
        with pytest.raises(StructuredTranslationError, match="max_chapter_characters"):
            translate_html("x" * 51, fake, cfg, TranslationConfig())

    def test_placeholder_drop_retry_succeeds(self):
        """Model drops placeholders on the first call -> stricter-prefix retry."""
        fake = FakeTranslator(drop_tag_calls={1})
        res = translate_html(
            "<p>前 <strong>中</strong> 后</p>", fake, StructuredConfig(), TranslationConfig()
        )
        out = res.translated_html
        assert res.retry_count == 1
        assert res.fallback_count == 0
        assert "<strong>" in out and "</strong>" in out
        assert out == "<p>EN:前 <strong>EN:中</strong> EN:后</p>"

    def test_placeholder_drop_uses_split_fallback(self):
        """Model drops placeholders on attempt + retry -> per-run fallback."""
        fake = FakeTranslator(drop_tag_calls={1, 2})
        res = translate_html(
            "<p>前 <strong>中</strong> 后</p>", fake, StructuredConfig(), TranslationConfig()
        )
        out = res.translated_html
        assert res.fallback_count >= 1
        assert "<strong>" in out and "</strong>" in out
        assert out == "<p>EN:前 <strong>EN:中</strong> EN:后</p>"

    def test_persistent_placeholder_drop_fails_closed(self):
        """Model always drops placeholders in an ATTRIBUTE segment -> error,
        never corrupted output. (Text chinese runs have no placeholders, so
        the split fallback always succeeds for them; attribute values can
        contain identifiers, where a dropping model has no safe path.)"""
        fake = FakeTranslator(drop_tag_calls=set(range(1, 100)))
        cfg = StructuredConfig(translatable_attributes=("alt",))
        with pytest.raises(StructuredTranslationError, match="could not be translated"):
            translate_html(
                '<img alt="看 https://example.com 这里">',
                fake, cfg, TranslationConfig(),
            )

    def test_no_translation_no_model_calls(self):
        fake = FakeTranslator()
        translate_html(
            "<p>English only</p><div class='notranslate'>中文保留</div>",
            fake, StructuredConfig(), TranslationConfig(),
        )
        assert fake.call_count == 0

    def test_metrics_recorded(self):
        fake = FakeTranslator()
        res = translate_html(MIXED_HTML, fake, StructuredConfig(), TranslationConfig())
        assert res.correlation_id
        assert res.total_source_tokens > 0
        assert res.total_target_tokens > 0
        assert res.excluded_text_nodes == 2
        assert res.duration_seconds >= 0
        assert len(res.segments) == res.segment_count
        assert all("segment_id" in s for s in res.segments)

    def test_target_budget_capped(self):
        fake = FakeTranslator()
        cfg = StructuredConfig(max_target_tokens=100)
        # 500-char text -> source tokens ~250 -> budget = min(100, 625) = 100
        st = StructuredTranslator(fake, cfg, TranslationConfig())
        st.translate("加厚防水面料。" * 100)
        assert st._target_budget(250) == 100

    def test_context_refs_recorded(self):
        fake = FakeTranslator()
        res = translate_html("<p>一</p><p>二</p><p>三</p>", fake, StructuredConfig(), TranslationConfig())
        segs = res.segments
        assert segs[0]["context_after_id"] == segs[1]["segment_id"]
        assert segs[1]["context_before_id"] == segs[0]["segment_id"]
