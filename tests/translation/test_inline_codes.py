"""Inline-code extraction tests: exact entity/tag spelling preservation.

Covers the required entity/break matrix, mixed model numbers, configurable
preserve patterns, adversarial fake models, long-chapter completeness, and
fail-closed reconstruction.
"""

import re
import pytest

from image_translation.translation.base import Translator
from image_translation.translation.chapter_chunking import (
    RUN_ENTITY,
    RUN_MODEL_NUMBER,
    collect_blocks,
    segment_blocks,
)
from image_translation.translation.config import (
    GlossaryEntry,
    StructuredConfig,
    TranslationConfig,
)
from image_translation.translation.exceptions import StructuredTranslationError
from image_translation.translation.html_document import HTMLDocument
from image_translation.translation.models import TranslationResult
from image_translation.translation.structured_translation import (
    StructuredTranslator,
)


class FakeTranslator(Translator):
    """Wraps every CJK run as ``EN:<cjk>``; keeps placeholders intact."""

    call_count = 0

    @property
    def name(self):
        return "fake"

    @property
    def runtime_info(self):
        return None

    def translate_text(self, text, source_lang="zh", target_lang="en",
                       max_new_tokens=None):
        return self.translate_batch_texts(
            [text], source_lang, target_lang, max_new_tokens
        )[0]

    def translate_batch_texts(self, texts, source_lang="zh", target_lang="en",
                              max_new_tokens=None):
        out = []
        for t in texts:
            parts = re.split(r"(__ITRANSLATE_[A-Z]\d{4}_)", t)
            parts = [
                re.sub(r"[\u4e00-\u9fff]+", lambda m: "EN:" + m.group(0), p)
                if not p.startswith("__ITRANSLATE_") else p
                for p in parts
            ]
            out.append(TranslationResult(
                source_text=t, translated_text="".join(parts),
                model_name="fake", device="cpu",
            ))
        return out


ENTITY_CASES = [
    ("<p>中文&nbsp;English</p>", "&nbsp;"),
    ("<p>中文&#160;English</p>", "&#160;"),
    ("<p>中文&#xA0;English</p>", "&#xA0;"),
    ("<p>中文&amp;English</p>", "&amp;"),
    ("<p>中文&lt;br&gt;English</p>", "&lt;br&gt;"),
    ("<p>中文<br>English</p>", "<br>"),
    ("<p>中文<br/>English</p>", "<br/>"),
]


class TestEntityAndBreakPreservation:
    @pytest.mark.parametrize("html,code", ENTITY_CASES)
    def test_exact_spelling_preserved(self, html, code):
        res = StructuredTranslator(
            FakeTranslator(), StructuredConfig(), TranslationConfig()
        ).translate(html)
        out = res.translated_html
        # exact source spelling present at the original position
        assert code in out
        assert out.startswith("<p>") and out.endswith("</p>")
        # Chinese still translated
        assert "EN:" in out
        # no OTHER entity spelling is substituted for this one
        for other_html, other_code in ENTITY_CASES:
            if other_code != code and other_code not in (code,):
                assert other_code not in out, f"{other_code!r} leaked into {out!r}"

    def test_entity_spellings_stay_distinct(self):
        html = ("<p>a&nbsp;b&#160;c&#xA0;d</p>")
        out = StructuredTranslator(
            FakeTranslator(), StructuredConfig(), TranslationConfig()
        ).translate(html).translated_html
        assert out == "<p>a&nbsp;b&#160;c&#xA0;d</p>"

    def test_lt_br_gt_never_becomes_markup(self):
        html = "<p>中文&lt;br&gt;English</p>"
        out = StructuredTranslator(
            FakeTranslator(), StructuredConfig(), TranslationConfig()
        ).translate(html).translated_html
        assert "&lt;br&gt;" in out
        assert out.count("<br>") == 0  # never a real break tag

    def test_amp_and_literal_ampersand_distinct(self):
        html = "<p>AT&amp;T 与 AT&T 电话</p>"
        out = StructuredTranslator(
            FakeTranslator(), StructuredConfig(), TranslationConfig()
        ).translate(html).translated_html
        assert "AT&amp;T" in out
        assert "AT&T" in out

    def test_required_example_shape(self):
        """The exact structural shape from the task specification."""
        html = "<p>型号 ABC-123&nbsp;采用<strong>加厚防水面料</strong><br>适合 daily use。</p>"
        out = StructuredTranslator(
            FakeTranslator(), StructuredConfig(), TranslationConfig()
        ).translate(html).translated_html
        # entities, tags, model number all exact
        assert "&nbsp;" in out
        assert "<strong>" in out and "</strong>" in out
        assert "<br>" in out
        assert "ABC-123" in out
        # only eligible Chinese differs (EN: prefix marks translation)
        assert "EN:采用" in out and "EN:适合" in out

    def test_br_vs_br_slash_distinct(self):
        out = StructuredTranslator(
            FakeTranslator(), StructuredConfig(), TranslationConfig()
        ).translate("<p>a<br>b<br/>c</p>").translated_html
        assert out == "<p>a<br>b<br/>c</p>"

    def test_script_entities_untouched(self):
        html = '<script>var s = "&nbsp;&lt;br&gt;";</script><p>中文</p>'
        out = StructuredTranslator(
            FakeTranslator(), StructuredConfig(), TranslationConfig()
        ).translate(html).translated_html
        assert 'var s = "&nbsp;&lt;br&gt;";' in out
        assert "EN:" in out


MIXED_MODEL_HTML = (
    "<p>型号 ABC-123、X1000、MK-Ⅱ 与 iPhone 16 Pro 均支持 USB-C。</p>"
    "<p>Model ABC-123 uses the USB-C interface，兼容 Windows 11。</p>"
)


class TestMixedModelNumbers:
    def test_model_numbers_exact(self):
        out = StructuredTranslator(
            FakeTranslator(), StructuredConfig(), TranslationConfig()
        ).translate(MIXED_MODEL_HTML).translated_html
        for ident in ("ABC-123", "X1000", "MK-Ⅱ", "iPhone 16 Pro", "USB-C",
                      "Windows 11"):
            assert ident in out, f"{ident!r} lost in {out!r}"
        assert "EN:" in out  # Chinese translated

    def test_entity_in_model_text(self):
        html = "<p>型号 X1000&nbsp;与 iPhone 16 Pro 兼容。</p>"
        out = StructuredTranslator(
            FakeTranslator(), StructuredConfig(), TranslationConfig()
        ).translate(html).translated_html
        assert "X1000&nbsp;" in out
        assert "iPhone 16 Pro" in out


class TestPreservePatterns:
    PATTERNS = (r"[A-Z]{2,}/\d{4}", r"Q\d+-\d{4}")

    def test_configured_patterns_protected(self):
        cfg = StructuredConfig(preserve_patterns=self.PATTERNS)
        html = "<p>零件 ABC/1234 与 Q7-2024 需要更换。</p>"
        out = StructuredTranslator(
            FakeTranslator(), cfg, TranslationConfig()
        ).translate(html).translated_html
        assert "ABC/1234" in out
        assert "Q7-2024" in out
        assert "EN:零件" in out

    def test_patterns_produce_model_number_runs(self):
        doc = HTMLDocument("<p>中文&nbsp;ABC/1234 Q7-2024</p>")
        segments = segment_blocks(
            doc, collect_blocks(doc), lambda t: max(1, (len(t) + 1) // 2), 450,
            preserve_patterns=tuple(re.compile(p) for p in self.PATTERNS),
        )
        kinds = [r.kind for r in segments[0].runs]
        assert RUN_ENTITY in kinds
        assert RUN_MODEL_NUMBER in kinds
        assert kinds.count(RUN_MODEL_NUMBER) == 2

    def test_invalid_pattern_rejected(self):
        with pytest.raises(ValueError, match="not a valid regex"):
            StructuredConfig(preserve_patterns=("[unclosed",))

    def test_empty_pattern_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            StructuredConfig(preserve_patterns=("  ",))

    def test_patterns_do_not_break_without_config(self):
        html = "<p>零件 ABC/1234 需要更换。</p>"
        out = StructuredTranslator(
            FakeTranslator(), StructuredConfig(), TranslationConfig()
        ).translate(html).translated_html
        # without the pattern the string is still English-protected (exact)
        assert "ABC/1234" in out


_TOKEN_RE = r"__IT[A-Z0-9]*_[A-Z]\d{4}_"


class TestAdversarialEntityFakes:
    def test_fake_dropping_entity_placeholder_recovered(self):
        """A model that drops entity placeholders fails validation on every
        attempt; the split fallback restores entities from the source map —
        the output is exact, never partial."""
        class DropEntityFake(FakeTranslator):
            def translate_batch_texts(self, texts, source_lang="zh",
                                      target_lang="en", max_new_tokens=None):
                out = []
                for t in texts:
                    self.call_count += 1
                    parts = re.split(r"(" + _TOKEN_RE + r")", t)
                    kept = [p for p in parts if not p.startswith("__IT")]
                    translated = re.sub(
                        r"[\u4e00-\u9fff]+",
                        lambda m: "EN:" + m.group(0), "".join(kept),
                    )
                    out.append(TranslationResult(
                        source_text=t, translated_text=translated,
                        model_name="fake", device="cpu",
                    ))
                return out

        html = "<p>中文&nbsp;English</p>"
        fake = DropEntityFake()
        res = StructuredTranslator(
            fake, StructuredConfig(), TranslationConfig()
        ).translate(html)
        assert res.translated_html == "<p>EN:中文&nbsp;English</p>"
        assert res.fallback_count >= 1

    def test_fake_duplicating_entity_placeholder_recovered(self):
        """Duplicated entity placeholders fail validation; the fallback
        restores exactly one occurrence per source position."""
        class DupEntityFake(FakeTranslator):
            def translate_batch_texts(self, texts, source_lang="zh",
                                      target_lang="en", max_new_tokens=None):
                out = []
                for t in texts:
                    self.call_count += 1
                    parts = re.split(r"(" + _TOKEN_RE + r")", t)
                    dup = []
                    for p in parts:
                        dup.append(p)
                        if p.startswith("__IT") and p[2:3].isalpha():
                            dup.append(p)  # duplicate every token
                    translated = re.sub(
                        r"[\u4e00-\u9fff]+",
                        lambda m: "EN:" + m.group(0), "".join(dup),
                    )
                    out.append(TranslationResult(
                        source_text=t, translated_text=translated,
                        model_name="fake", device="cpu",
                    ))
                return out

        html = "<p>中文&nbsp;English</p>"
        fake = DupEntityFake()
        res = StructuredTranslator(
            fake, StructuredConfig(), TranslationConfig()
        ).translate(html)
        assert res.translated_html == "<p>EN:中文&nbsp;English</p>"
        assert res.fallback_count >= 1

    def test_split_never_cuts_inside_entity_marker(self):
        """Small budgets force splits; entity markers must stay atomic and
        whitespace-only carrier runs must never be sent to the model."""
        html = (
            "<p>型号 X13&nbsp;与 X1300&#160;兼容，功耗 &lt; 5W&amp;稳定，"
            "支持 USB-C。<br/>更多说明&#xA0;见文档。</p>"
        )
        for budget in (30, 40, 60):
            res = StructuredTranslator(
                FakeTranslator(), StructuredConfig(max_segment_tokens=budget),
                TranslationConfig(),
            ).translate(html)
            out = res.translated_html
            assert res.segment_count > 1
            assert out.count("&nbsp;") == 1
            assert out.count("&#160;") == 1
            assert out.count("&#xA0;") == 1
            assert out.count("&amp;") == 1
            assert out.count("&lt;") == 1
            assert "&gt;" not in out  # only <br/> end-tag-less markup here
            assert "<br/>" in out
            assert "EN:型号" in out and "EN:见文档。" in out
            # no partial/split markers may leak into the output
            assert "\x02" not in out and "\x03" not in out

    def test_mandatory_matrix_link_case(self):
        """Required regression input: inline <strong> + <a> with an attribute
        entity — all codes exact, Chinese translated in place on both sides."""
        html = (
            '<p>中文<strong>加厚</strong>English'
            '<a href="/x?a=1&amp;b=2">链接</a>结尾</p>'
        )
        out = StructuredTranslator(
            FakeTranslator(), StructuredConfig(), TranslationConfig()
        ).translate(html).translated_html
        # exact markup + attribute entity spelling
        assert "<strong>EN:加厚</strong>" in out
        assert 'href="/x?a=1&amp;b=2"' in out
        assert "English" in out
        # Chinese translated in place on BOTH sides of the markup
        assert "EN:中文" in out and "EN:结尾" in out
        assert "EN:链接" in out

    def test_model_emitted_br_becomes_escaped_text(self):
        """A fake model emitting <br> as ordinary text can never create a
        real break tag — the serializer escapes it."""
        class EmitBrFake(FakeTranslator):
            def translate_batch_texts(self, texts, source_lang="zh",
                                      target_lang="en", max_new_tokens=None):
                out = []
                for t in texts:
                    self.call_count += 1
                    parts = re.split(r"(" + _TOKEN_RE + r")", t)
                    parts = [
                        re.sub(r"[\u4e00-\u9fff]+",
                               lambda m: "EN:" + m.group(0) + "<br>&nbsp;", p)
                        if not p.startswith("__IT") else p
                        for p in parts
                    ]
                    out.append(TranslationResult(
                        source_text=t, translated_text="".join(parts),
                        model_name="fake", device="cpu",
                    ))
                return out

        out = StructuredTranslator(
            EmitBrFake(), StructuredConfig(), TranslationConfig()
        ).translate("<p>中文English</p>").translated_html
        assert "&lt;br&gt;" in out          # escaped to text
        assert "&amp;nbsp;" in out          # escaped to text
        assert len(re.findall(r"<br(?!/)>", out)) == 0  # never a real tag

    def test_crashing_model_fails_closed_no_partial(self):
        """If every attempt (and the fallback) crashes, the request fails
        with a structured error — no partial HTML is returned."""
        class CrashFake(FakeTranslator):
            def translate_batch_texts(self, texts, source_lang="zh",
                                      target_lang="en", max_new_tokens=None):
                raise RuntimeError("gpu exploded")

        html = "<p>中文&nbsp;English</p>"
        with pytest.raises(StructuredTranslationError):
            StructuredTranslator(
                CrashFake(), StructuredConfig(), TranslationConfig()
            ).translate(html)

    def test_fake_reordering_tags_is_recovered(self):
        """A model that swaps <strong> tokens fails validation and is
        recovered via the split fallback — the output never has reordered
        tags, and nothing partial is returned."""
        class ReorderTagFake(FakeTranslator):
            def translate_batch_texts(self, texts, source_lang="zh",
                                      target_lang="en", max_new_tokens=None):
                out = []
                for t in texts:
                    self.call_count += 1
                    parts = re.split(r"(" + _TOKEN_RE + r")", t)
                    toks = [p for p in parts if p.startswith("__IT") and "T" in p]
                    toks.reverse()  # swap the <strong> and </strong> tokens
                    swapped = []
                    ti = 0
                    for p in parts:
                        if p.startswith("__IT") and "T" in p:
                            swapped.append(toks[ti])
                            ti += 1
                        else:
                            swapped.append(p)
                    translated = "".join(swapped)
                    out.append(TranslationResult(
                        source_text=t, translated_text=translated,
                        model_name="fake", device="cpu",
                    ))
                return out

        html = "<p>中文<strong>加厚</strong>English</p>"
        res = StructuredTranslator(
            ReorderTagFake(), StructuredConfig(), TranslationConfig()
        ).translate(html)
        out = res.translated_html
        # tags end up in SOURCE order (fallback recovery), never swapped
        assert out.index("<strong>") < out.index("</strong>")
        assert "English" in out
        assert res.fallback_count >= 1

    def test_model_emitted_entity_text_is_escaped(self):
        class EmitEntityFake(FakeTranslator):
            def translate_batch_texts(self, texts, source_lang="zh",
                                      target_lang="en", max_new_tokens=None):
                out = []
                for t in texts:
                    self.call_count += 1
                    # model tries to emit &nbsp; as free text
                    parts = re.split(r"(__ITRANSLATE_[A-Z]\d{4}_)", t)
                    parts = [
                        re.sub(r"[\u4e00-\u9fff]+",
                               lambda m: "EN:" + m.group(0) + "&nbsp;", p)
                        if not p.startswith("__ITRANSLATE_") else p
                        for p in parts
                    ]
                    out.append(TranslationResult(
                        source_text=t, translated_text="".join(parts),
                        model_name="fake", device="cpu",
                    ))
                return out

        html = "<p>中文English</p>"
        out = StructuredTranslator(
            EmitEntityFake(), StructuredConfig(), TranslationConfig()
        ).translate(html).translated_html
        # model-emitted entity must be escaped to TEXT — never a real entity
        assert "&amp;nbsp;" in out
        assert out.count("&nbsp;") == 0


LONG_PARAS = "\n".join(
    f"<p>第 {i} 段：型号 X{i}00&nbsp;采用<strong>加厚防水面料</strong>，"
    f"兼容 Windows 11&amp;12，适合 daily use。<br/>更多说明&#160;见文档。</p>"
    for i in range(1, 56)
)


class TestLongChapterInlineCodes:
    def test_entities_and_codes_complete_ordered(self):
        html = (
            "<h1>产品总览&nbsp;Overview</h1>"
            + LONG_PARAS
            + "<script>var x = '&nbsp;';</script>"
        )
        cfg = StructuredConfig()
        res = StructuredTranslator(
            FakeTranslator(), cfg, TranslationConfig()
        ).translate(html)
        out = res.translated_html
        assert len(out) > 4000
        assert "EN:" in out
        # entity spellings exact, every occurrence present
        # (56 = 55 paragraphs + h1; the excluded <script> keeps its own
        # literal '&nbsp;' text, counted separately in the raw script)
        assert out.count("&nbsp;") == 57
        assert out.count("&#160;") == 55
        assert out.count("&amp;") == 55
        assert out.count("<br/>") == 55
        # identifiers exact
        for i in (1, 10, 55):
            assert f"X{i}00" in out
            assert f"Windows 11&amp;12" in out
        # no truncation, no loss, no reordering: last paragraph's identifier
        # comes after the first paragraph's identifier
        assert out.find("X5500") > out.find("X100")
        assert out.count("EN:段") == 55
        # excluded script byte-identical
        assert "var x = '&nbsp;';" in out
        # every segment has contiguous unique ids (validated at build time)

    def test_fingerprint_stable_with_entities(self):
        """Real invariants: the result's fingerprint check is True, the
        OUTPUT document's structural fingerprint equals the SOURCE
        document's, and the fingerprint is not vacuous (tampering changes
        it)."""
        res = StructuredTranslator(
            FakeTranslator(), StructuredConfig(), TranslationConfig()
        ).translate(LONG_PARAS)
        assert res.fingerprint_ok is True
        fp_src = HTMLDocument(LONG_PARAS).fingerprint()
        fp_out = HTMLDocument(res.translated_html).fingerprint()
        assert fp_src == fp_out
        # fingerprint must actually detect structural tampering
        tampered = HTMLDocument(LONG_PARAS)
        strong = next(
            (e for e in tampered.element_nodes()
             if e.tag == "strong" and e.tag != "#document"), None
        )
        assert strong is not None
        strong.tag = "em"
        assert tampered.fingerprint() != fp_src
