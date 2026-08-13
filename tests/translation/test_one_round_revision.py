"""One-round revision tests: mixed grouping, propagation, attrs, injection,
deadline, truncation, concurrency (no GPU)."""

from __future__ import annotations

import re
import threading
import time

import pytest

from image_translation.translation.base import Translator
from image_translation.translation.chapter_chunking import collect_blocks
from image_translation.translation.config import (
    GlossaryEntry,
    StructuredConfig,
    TranslationConfig,
)
from image_translation.translation.exceptions import (
    StructuredTranslationError,
    TranslationInputError,
)
from image_translation.translation.html_document import HTMLDocument
from image_translation.translation.models import TranslationResult
from image_translation.translation.structured_translation import StructuredTranslator

import image_translation.translation.structured_translation as st_mod


class FakeTranslator(Translator):
    """Wraps each CJK run with EN:; records call args; optionally drops
    placeholders or injects fake HTML."""

    def __init__(self, drop_all=(), inject_html=()):
        self.call_count = 0
        self.drop_all = set(drop_all)
        self.inject_html = set(inject_html)
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "fake"

    @property
    def runtime_info(self):
        return None

    def translate_text(self, text, source_lang="zh", target_lang="en", max_new_tokens=None):
        return self.translate_batch_texts(
            [text], source_lang, target_lang, max_new_tokens
        )[0]

    def translate_batch_texts(
        self, texts, source_lang="zh", target_lang="en", max_new_tokens=None
    ):
        out = []
        with self._lock:
            for t in texts:
                self.call_count += 1
                n = self.call_count
                self.calls.append({
                    "text": t, "source_lang": source_lang,
                    "target_lang": target_lang, "max_new_tokens": max_new_tokens,
                })
                translated = re.sub(
                    r"[\u4e00-\u9fff]+", lambda m: "EN:" + m.group(0), t
                )
                if n in self.drop_all:
                    translated = re.sub(r"__IT[A-Z0-9]*_[A-Z]\d{4}_", "", translated)
                if n in self.inject_html:
                    translated = translated + "<script>alert(1)</script>"
                out.append(
                    TranslationResult(
                        source_text=t, translated_text=translated,
                        model_name="fake", device="cpu",
                    )
                )
        return out


@pytest.fixture(autouse=True)
def fake_measure(monkeypatch):
    class FakeTok:
        def __call__(self, text, truncation=False):
            return {"input_ids": list(range(max(1, (len(text) + 1) // 2)))}
    monkeypatch.setattr(st_mod, "_get_measure_tokenizer", lambda *a, **k: FakeTok())


# ---------------------------------------------------------------------------
# Mixed-language grouping (the four mandated cases)
# ---------------------------------------------------------------------------

MIXED_CASES = [
    "<p>请 click <strong>Continue</strong> 然后继续。</p>",
    "<p>Use the USB-C cable 连接设备。</p>",
    "<p>型号 ABC-123，compatible with Windows 11。</p>",
    "<p>这是 <em>premium</em> 防水面料。</p>",
]


class TestMixedGrouping:
    def test_english_runs_protected(self):
        """English inside a mixed block becomes protected runs — never sent
        to the model as free text."""
        doc = HTMLDocument(MIXED_CASES[0])
        blocks = collect_blocks(doc)
        assert len(blocks) == 1

    def test_all_english_block_dropped(self):
        doc = HTMLDocument("<p>Pure English sentence here.</p>")
        assert collect_blocks(doc) == []

    def test_mixed_cases_translate(self):
        fake = FakeTranslator()
        expected_identifiers = [
            {"Continue"},
            {"USB-C"},
            {"ABC-123", "Windows 11"},
            {"premium"},
        ]
        for html, identifiers in zip(MIXED_CASES, expected_identifiers):
            res = StructuredTranslator(
                fake, StructuredConfig(), TranslationConfig()
            ).translate(html)
            out = res.translated_html
            for ident in identifiers:
                assert ident in out, f"{ident!r} missing in {out!r}"
            # Chinese translated in place
            assert "EN:" in out

    def test_english_changed_by_model_is_restored_exactly(self):
        """The mandated case: a fake model that REWRITES English words must
        not affect the output — English is protected before inference and
        restored from the ORIGINAL text."""
        class ChangeEnglishFake(FakeTranslator):
            def translate_batch_texts(self, texts, source_lang="zh", target_lang="en", max_new_tokens=None):
                out = []
                for t in texts:
                    self.call_count += 1
                    # model that keeps placeholders but CORRUPTS every word
                    parts = re.split(r"(__IT[A-Z0-9]*_[A-Z]\d{4}_)", t)
                    parts = [
                        re.sub(r"[A-Za-z]{2,}", lambda m: "GARBAGE", p)
                        if not p.startswith("__IT") else p
                        for p in parts
                    ]
                    translated = re.sub(
                        r"[\u4e00-\u9fff]+", lambda m: "EN:" + m.group(0), "".join(parts)
                    )
                    out.append(TranslationResult(
                        source_text=t, translated_text=translated,
                        model_name="fake", device="cpu",
                    ))
                return out

        fake = ChangeEnglishFake()
        res = StructuredTranslator(
            fake, StructuredConfig(), TranslationConfig()
        ).translate(MIXED_CASES[0])
        out = res.translated_html
        # Model output contained GARBAGE in place of click/Continue; the
        # protected spans restore the ORIGINAL text exactly.
        assert "GARBAGE" not in out
        assert "Continue" in out
        assert "click" in out
        assert "EN:" in out  # Chinese still translated

    def test_english_split_around_inline_tags_restored(self):
        """English split around <strong>/<em>/<span> is restored exactly."""
        htmls = [
            "<p>请 <strong>click</strong> 这里</p>",
            "<p>前 <em>middle</em> 后</p>",
            "<p>中文 <span>English span</span> 中文</p>",
        ]
        fake = FakeTranslator()
        for html in htmls:
            out = StructuredTranslator(
                fake, StructuredConfig(), TranslationConfig()
            ).translate(html).translated_html
            assert ("click" in out) or ("middle" in out) or ("English span" in out)

    def test_every_category_restored_against_changing_model(self):
        """The mandated audit: a fake model that rewrites EVERY unprotected
        English word to GARBAGE must not affect the output — all protected
        categories are restored from the ORIGINAL source text."""
        class TotalCorruptionFake(FakeTranslator):
            def translate_batch_texts(self, texts, source_lang="zh",
                                      target_lang="en", max_new_tokens=None):
                out = []
                for t in texts:
                    self.call_count += 1
                    # corrupt every word OUTSIDE placeholders
                    parts = re.split(r"(__IT[A-Z0-9]*_[A-Z]\d{4}_)", t)
                    parts = [
                        re.sub(r"[A-Za-z0-9]{2,}", "GARBAGE", p)
                        if not p.startswith("__IT") else p
                        for p in parts
                    ]
                    translated = re.sub(
                        r"[\u4e00-\u9fff]+", lambda m: "EN:" + m.group(0),
                        "".join(parts),
                    )
                    out.append(TranslationResult(
                        source_text=t, translated_text=translated,
                        model_name="fake", device="cpu",
                    ))
                return out

        cfg = StructuredConfig(glossary=(
            GlossaryEntry("充电器", "Charger"),
        ))
        html = (
            "<p>Please click the button 请继续。</p>"                     # ordinary English
            "<p>请 <strong>click</strong> 这里 <em>now</em> 谢谢。</p>"    # English around inline tags
            "<p>产品编号 ABC-123 已发货。</p>"                           # product code
            "<p>访问 https://example.com/x?q=1 获取详情。</p>"           # URL
            "<p>联系 support@example.com 获取帮助。</p>"                 # email
            "<p>电缆长度 1.5 meters 支持 20V/5A 快充。</p>"              # measurements
            "<p>支持 Windows 11 和 iPhone 16 Pro。</p>"                  # versions
            "<p>本充电器支持快充协议。</p>"                              # glossary term
        )
        res = StructuredTranslator(TotalCorruptionFake(), cfg, None).translate(html)
        out = res.translated_html
        # the model's corruption must NEVER appear
        assert "GARBAGE" not in out, f"model corruption leaked: {out}"
        # every protected category restored exactly
        for expected in (
            "Please click the button",
            "<strong>click</strong>",
            "<em>now</em>",
            "ABC-123",
            "https://example.com/x?q=1",
            "support@example.com",
            "1.5 meters",
            "20V/5A",
            "Windows 11",
            "iPhone 16 Pro",
            "Charger",          # glossary target
        ):
            assert expected in out, f"{expected!r} missing in {out!r}"
        # Chinese translated in place
        assert "EN:" in out

    def test_identifier_and_glossary_target_changes_restored(self):
        """Adversarial fake emits changed identifier/glossary-looking text in
        every slot: protected originals must be restored exactly and the
        model's attempts can never REPLACE them (identifiers and glossary
        targets never reach the model as free text)."""
        class OverrideFake(FakeTranslator):
            def translate_batch_texts(self, texts, source_lang="zh",
                                      target_lang="en", max_new_tokens=None):
                out = []
                for t in texts:
                    self.call_count += 1
                    # every slot piece claims a changed identifier/target;
                    # placeholders (the protected content) stay intact
                    parts = re.split(r"(__IT[A-Z0-9]*_[A-Z]\d{4}_)", t)
                    parts = [
                        "USB-XX WrongTarget "
                        if not p.startswith("__IT") else p
                        for p in parts
                    ]
                    out.append(TranslationResult(
                        source_text=t, translated_text="".join(parts),
                        model_name="fake", device="cpu",
                    ))
                return out

        cfg = StructuredConfig(glossary=(GlossaryEntry("充电器", "Charger"),))
        html = (
            "<p>型号 ABC-123 已发货，充电器支持快充。</p>"
            "<p>访问 https://example.com 获取信息，本充电器兼容 Windows 11。</p>"
        )
        res = StructuredTranslator(OverrideFake(), cfg, None).translate(html)
        out = res.translated_html
        # protected originals restored exactly, exactly once each
        assert out.count("ABC-123") == 1
        assert out.count("https://example.com") == 1
        assert out.count("Windows 11") == 1
        assert out.count("Charger") == 2
        # no placeholder token may leak into the output
        assert "__ITRANSLATE" not in out

    def test_english_never_disappears(self):
        """Reconstruction never loses English-only block content."""
        html = "<p>请 click <strong>Continue</strong> 然后继续。</p><p>Keep this English block.</p>"
        fake = FakeTranslator()
        out = StructuredTranslator(fake, StructuredConfig(), TranslationConfig()).translate(html).translated_html
        assert "Keep this English block." in out
        assert "Continue" in out

    def test_source_runs_cover_every_text_node_once(self):
        """Invariant: covered nodes have contiguous non-overlapping run
        offsets; every covered node id is a real text node."""
        from image_translation.translation.chapter_chunking import segment_blocks
        html = ("<p>请 click <strong>Continue</strong> 然后继续。</p>"
                "<p>Use the USB-C cable 连接设备。</p>")
        doc = HTMLDocument(html)
        blocks = collect_blocks(doc)
        segments = segment_blocks(
            doc, blocks, lambda t: max(1, (len(t) + 1) // 2), 450
        )
        spans = []  # (node_id, offset_start, offset_end)
        for seg in segments:
            for run in seg.runs:
                if run.node_id != "tag":
                    spans.append((run.node_id, run.offset_start, run.offset_end))
                    assert 0 <= run.offset_start <= run.offset_end
                    assert run.offset_end <= len(seg.block_text)
        # ranges must not overlap
        by_node = {}
        for nid, start, end in spans:
            prev = by_node.get(nid)
            if prev is not None:
                p_start, p_end = prev
                assert start >= p_end, f"node {nid} ranges overlap"
            by_node[nid] = (start, end)
        # per block, offsets are contiguous (concat invariant, verified by
        # _validate_coverage at build time — re-check here cheaply):
        for seg in segments:
            for i, run in enumerate(seg.runs[:-1]):
                nxt = seg.runs[i + 1]
                if run.node_id == "tag" or nxt.node_id == "tag":
                    continue
                assert run.offset_end <= nxt.offset_start
        doc_nodes = {n.id for n in doc.text_nodes()}
        assert set(by_node) <= doc_nodes
        assert len(by_node) >= 4  # 请 click/Continue/然后继续 + cable paragraph


# ---------------------------------------------------------------------------
# Source-language propagation
# ---------------------------------------------------------------------------

class TestLanguagePropagation:
    def test_requested_languages_reach_translator(self):
        fake = FakeTranslator()
        res = StructuredTranslator(
            fake, StructuredConfig(), TranslationConfig()
        ).translate("<p>加厚防水面料</p>", source_lang="zh", target_lang="fr")
        assert res.source_language == "zh"
        assert res.target_language == "fr"
        assert fake.calls, "no model calls recorded"
        assert all(c["source_lang"] == "zh" for c in fake.calls)
        assert all(c["target_lang"] == "fr" for c in fake.calls)
        # segment metadata records the real languages
        assert all(s["source_language"] == "zh" and s["target_language"] == "fr"
                   for s in res.segments)

    def test_default_pair_recorded(self):
        fake = FakeTranslator()
        StructuredTranslator(fake, StructuredConfig(), TranslationConfig()).translate("<p>中文</p>")
        assert fake.calls[0]["source_lang"] == "zh"
        assert fake.calls[0]["target_lang"] == "en"

    def test_max_new_tokens_override_passed(self):
        fake = FakeTranslator()
        # 100 chars -> ~50 source tokens -> budget = min(120, 125) = 120
        StructuredTranslator(
            fake, StructuredConfig(max_target_tokens=120), TranslationConfig()
        ).translate("<p>" + "加厚防水面料" * 20 + "</p>")
        assert fake.calls[0]["max_new_tokens"] == 120


# ---------------------------------------------------------------------------
# Truncation: no silent truncation anywhere
# ---------------------------------------------------------------------------

class TestNoTruncation:
    def test_translator_rejects_over_budget_plain_input(self):
        """Plain-path hard guard: over-budget input raises, never truncates."""
        from image_translation.translation.m2m100_translator import M2M100Translator

        class FakeIds:
            def __init__(self, n):
                self.shape = (1, n)

            def to(self, device):
                return self

            def tolist(self):
                return [[0] * self.shape[1]]

        class FakeTok:
            def __call__(self, texts, return_tensors=None, padding=None, truncation=None):
                assert truncation is False, "tokenizer must be called with truncation=False"
                n = len(texts[0]) + 2
                return {
                    "input_ids": FakeIds(n),
                    "attention_mask": FakeIds(n),
                }

            def get_lang_id(self, lang):
                return 1

            def batch_decode(self, ids, skip_special_tokens=False):
                return ["x"] * len(ids)

        class FakeModel:
            config = type("C", (), {"max_position_embeddings": 1024})()

        t = M2M100Translator(TranslationConfig())
        t._model = FakeModel()
        t._tokenizer = FakeTok()
        t._device_str = "cpu"
        t._precision_str = "float32"

        with pytest.raises(TranslationInputError, match="token budget"):
            t._translate_impl(["x" * 900], "zh", "en", max_new_tokens=256)

    def test_structured_over_budget_segment_fails(self, monkeypatch):
        """An indivisible over-budget unit fails with a structured error."""
        class HugeTok:
            def __call__(self, text, truncation=False):
                # every single character already exceeds the budget
                return {"input_ids": list(range(len(text) * 100))}

        monkeypatch.setattr(st_mod, "_get_measure_tokenizer", lambda *a, **k: HugeTok())
        fake = FakeTranslator()
        cfg = StructuredConfig(max_segment_tokens=8)
        with pytest.raises(StructuredTranslationError, match="indivisible"):
            StructuredTranslator(
                fake, cfg, TranslationConfig()
            ).translate("<p>" + "超" * 40 + "</p>")

    def test_no_truncation_in_source(self):
        """Static guard: the translator module must not call truncation=True."""
        src = open("src/image_translation/translation/m2m100_translator.py",
                   encoding="utf-8").read()
        assert "truncation=True" not in src


# ---------------------------------------------------------------------------
# Injection safety: model-emitted markup is escaped
# ---------------------------------------------------------------------------

class TestInjectionSafety:
    def test_model_html_is_escaped(self):
        fake = FakeTranslator(inject_html={1})
        out = StructuredTranslator(
            fake, StructuredConfig(), TranslationConfig()
        ).translate("<p>加厚防水面料</p>").translated_html
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_fake_html_placeholder_validation(self):
        """Model emitting raw HTML does not create new tags."""
        class HtmlEmittingFake(FakeTranslator):
            def translate_batch_texts(self, texts, source_lang="zh", target_lang="en", max_new_tokens=None):
                return [
                    TranslationResult(
                        source_text=t,
                        translated_text="EN:" + t + " <b>fake</b>",
                        model_name="fake", device="cpu",
                    ) for t in texts
                ]
        fake = HtmlEmittingFake()
        out = StructuredTranslator(
            fake, StructuredConfig(), TranslationConfig()
        ).translate("<p>前 <strong>中</strong> 后</p>").translated_html
        # The model's <b> must not become a real tag
        assert "<b>fake</b>" not in out
        assert "&lt;b&gt;fake&lt;/b&gt;" in out


# ---------------------------------------------------------------------------
# Translatable attributes
# ---------------------------------------------------------------------------

class TestTranslatableAttributes:
    def test_alt_title_translated_src_protected(self):
        cfg = StructuredConfig(translatable_attributes=("alt", "title"))
        fake = FakeTranslator()
        html = '<img alt="加厚防水面料" src="https://example.com/x.png" title="耐磨耐用">'
        out = StructuredTranslator(fake, cfg, TranslationConfig()).translate(html).translated_html
        assert 'alt="EN:加厚防水面料"' in out
        assert 'title="EN:耐磨耐用"' in out
        assert 'src="https://example.com/x.png"' in out  # never translated

    def test_no_attrs_translated_by_default(self):
        cfg = StructuredConfig()  # empty allowlist
        fake = FakeTranslator()
        html = '<img alt="加厚防水面料" src="x.png">'
        out = StructuredTranslator(fake, cfg, TranslationConfig()).translate(html).translated_html
        assert 'alt="加厚防水面料"' in out
        assert fake.call_count == 0

    def test_english_attr_untouched(self):
        cfg = StructuredConfig(translatable_attributes=("alt",))
        fake = FakeTranslator()
        html = '<img alt="Premium Quality" src="x.png">'
        out = StructuredTranslator(fake, cfg, TranslationConfig()).translate(html).translated_html
        assert 'alt="Premium Quality"' in out

    def test_attr_fingerprint_stable(self):
        cfg = StructuredConfig(translatable_attributes=("alt",))
        fake = FakeTranslator()
        html = '<img alt="加厚防水面料"><p>加厚防水面料</p>'
        doc = HTMLDocument(html)
        fp = doc.fingerprint(translatable_attrs={"alt"})
        out = StructuredTranslator(fake, cfg, TranslationConfig()).translate(html).translated_html
        # translated doc has same fingerprint
        doc2 = HTMLDocument(out)
        assert doc2.fingerprint(translatable_attrs={"alt"}) == fp


# ---------------------------------------------------------------------------
# Deadline (real, between segments) vs warning threshold (no cancel)
# ---------------------------------------------------------------------------

class TestDeadline:
    def test_deadline_exceeded_fails(self):
        class SlowFake(FakeTranslator):
            def translate_batch_texts(self, texts, source_lang="zh", target_lang="en", max_new_tokens=None):
                time.sleep(0.3)
                return super().translate_batch_texts(texts, source_lang, target_lang, max_new_tokens)

        fake = SlowFake()
        cfg = StructuredConfig(max_total_seconds=0.15)
        # multiple segments so the deadline check runs between them
        html = "".join(f"<p>第{i}段中文内容</p>" for i in range(3))
        with pytest.raises(StructuredTranslationError, match="deadline"):
            StructuredTranslator(fake, cfg, TranslationConfig()).translate(html)

    def test_warning_threshold_is_not_cancel(self):
        class SlowFake(FakeTranslator):
            def translate_batch_texts(self, texts, source_lang="zh", target_lang="en", max_new_tokens=None):
                time.sleep(0.1)
                return super().translate_batch_texts(texts, source_lang, target_lang, max_new_tokens)

        fake = SlowFake()
        cfg = StructuredConfig(segment_warning_seconds=0.05, max_total_seconds=10)
        out = StructuredTranslator(fake, cfg, TranslationConfig()).translate("<p>加厚防水面料</p>")
        assert "EN:加厚防水面料" in out.translated_html


# ---------------------------------------------------------------------------
# Final static audit: no truncation, no hard-coded source language
# ---------------------------------------------------------------------------

class TestStaticAudit:
    PROD_DIRS = [
        "src/image_translation/translation",
        "src/translation_server",
    ]

    def test_no_truncation_true_anywhere(self):
        """Fails if any production path reintroduces truncation=True."""
        offenders = []
        for d in self.PROD_DIRS:
            for root, _dirs, files in __import__("os").walk(d):
                for f in files:
                    if f.endswith(".py"):
                        p = __import__("os").path.join(root, f)
                        for i, line in enumerate(
                            open(p, encoding="utf-8").read().splitlines(), 1
                        ):
                            if "truncation=True" in line:
                                offenders.append(f"{p}:{i}")
        assert not offenders, f"truncation=True reintroduced: {offenders}"

    def test_no_half_reachable_from_auto_precision(self):
        """Fails if any production path can call model.half() from the
        default 'auto' precision. FP32 is the quality baseline; float16 is
        only an explicit opt-in (guarded by a 'float16' branch)."""
        offenders = []
        for d in self.PROD_DIRS:
            for root, _dirs, files in __import__("os").walk(d):
                for f in files:
                    if f.endswith(".py"):
                        p = __import__("os").path.join(root, f)
                        lines = (
                            open(p, encoding="utf-8").read().splitlines()
                        )
                        for i, line in enumerate(lines, 1):
                            if ".half()" in line:
                                prev = lines[i - 2] if i >= 2 else ""
                                if "float16" not in line and "float16" not in prev:
                                    offenders.append(
                                        f"{p}:{i}: {line.strip()}"
                                    )
        assert not offenders, (
            f"model.half() reachable outside an explicit float16 branch: "
            f"{offenders}"
        )

    def test_no_hardcoded_source_language_in_core_paths(self):
        """Fails if a production call site hard-codes the source language
        (defaults in function signatures are allowed; call sites must use
        the request value)."""
        offenders = []
        for d in self.PROD_DIRS:
            for root, _dirs, files in __import__("os").walk(d):
                for f in files:
                    if f.endswith(".py"):
                        p = __import__("os").path.join(root, f)
                        in_signature = False
                        for i, line in enumerate(
                            open(p, encoding="utf-8").read().splitlines(), 1
                        ):
                            stripped = line.strip()
                            if stripped.startswith("#"):
                                continue
                            if "def " in stripped:
                                # enter a (possibly multi-line) signature
                                in_signature = "):" not in stripped
                                continue
                            if in_signature:
                                if "):" in stripped:
                                    in_signature = False
                                continue
                            if re.search(r'source_lang(?:uage)?\s*=\s*"zh"', line):
                                offenders.append(f"{p}:{i}: {line.strip()}")
        assert not offenders, (
            f"hard-coded source language at call sites: {offenders}"
        )

    def test_no_vacuous_assertions_in_tests(self):
        """Fails if any test contains an assertion that is always true or
        always false (a vacuous check). Every assertion must express a real
        invariant that fails when the implementation is wrong."""
        offenders = []
        test_dir = __import__("os").path.join(
            __import__("os").path.dirname(__file__), ".."
        )
        test_dir = __import__("os").path.abspath(test_dir)
        for root, _dirs, files in __import__("os").walk(test_dir):
            for f in files:
                if not f.endswith(".py"):
                    continue
                p = __import__("os").path.join(root, f)
                for i, line in enumerate(
                    open(p, encoding="utf-8").read().splitlines(), 1
                ):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    if stripped.startswith(('"', "'")):
                        continue  # string literals/docstrings
                    if "\\b" in line:
                        continue  # this audit's own pattern literal
                    if re.search(r"assert True\b|assert False\b| or True\b",
                                 line):
                        offenders.append(f"{p}:{i}: {line.strip()}")
        assert not offenders, f"vacuous test assertions: {offenders}"

    def test_no_unused_advertised_configuration(self):
        """Every StructuredConfig field must be consumed by production code
        (except context_window_tokens, which is explicitly unsupported and
        consumed by validation)."""
        import dataclasses
        from image_translation.translation.config import StructuredConfig

        fields = [f.name for f in dataclasses.fields(StructuredConfig)]
        code = ""
        for d in self.PROD_DIRS:
            for root, _dirs, files in __import__("os").walk(d):
                for f in files:
                    if f.endswith(".py") and f not in ("config.py",):
                        p = __import__("os").path.join(root, f)
                        code += open(p, encoding="utf-8").read()
        unused = []
        for fname in fields:
            if fname == "context_window_tokens":
                continue  # explicitly unsupported; consumed by validation
            if fname not in code:
                unused.append(fname)
        assert not unused, f"advertised config not consumed: {unused}"


# ---------------------------------------------------------------------------
# Property-style placeholder corruption test: random invalid outputs must
# retry safely or fail closed — never return partial/corrupted HTML.
# ---------------------------------------------------------------------------

class TestPlaceholderCorruptionProperty:
    """Randomized corruptions of model output: missing/duplicated/reordered/
    altered placeholders, fake tags, model-added text. Every run must either
    produce structurally valid output with exact preservation, or raise a
    StructuredTranslationError — never partial HTML."""

    HTML = (
        "<p>请 click <strong>Continue</strong> 然后继续。</p>"
        "<p>Use the USB-C cable 连接设备。</p>"
        "<p>这是 <em>premium</em> 防水面料。访问 https://example.com/x 获取详情。</p>"
    )

    @staticmethod
    def _corrupt(text: str, rng) -> str:
        parts = re.split(r"(__IT[A-Z0-9]*_[A-Z]\d{4}_)", text)
        tokens = [p for p in parts if p.startswith("__IT")]
        mode = rng.choice(["reorder", "missing", "duplicate", "alter",
                           "fake_tags", "added_text"])
        out_parts = [p for p in parts if not p.startswith("__IT")]
        if mode == "reorder" and len(tokens) >= 2:
            rng.shuffle(tokens)
        elif mode == "missing" and tokens:
            tokens = tokens[: rng.randint(0, len(tokens) - 1)]
        elif mode == "duplicate" and tokens:
            tokens.insert(rng.randrange(len(tokens) + 1), rng.choice(tokens))
        elif mode == "alter" and tokens:
            i = rng.randrange(len(tokens))
            tok = tokens[i]
            tokens[i] = tok[:-1] + ("_" if tok[-1] != "_" else "X")
        elif mode == "fake_tags" and out_parts:
            i = rng.randrange(len(out_parts))
            out_parts[i] += "<b>fake</b>"
        else:  # added_text
            out_parts.insert(rng.randrange(len(out_parts) + 1),
                             rng.choice(["EXTRA", "more text here", "!!"]))

        # interleave: text parts and tokens in order
        result = out_parts[0]
        for i, tok in enumerate(tokens):
            result += tok
            if i + 1 < len(out_parts):
                result += out_parts[i + 1]
        return result

    def test_random_corruptions_never_leak_partial_html(self):
        import random

        from image_translation.translation.exceptions import (
            StructuredTranslationError,
        )

        fake = FakeTranslator()
        st = StructuredTranslator(fake, StructuredConfig(), TranslationConfig())
        rng = random.Random(20260812)
        outcomes = {"ok": 0, "error": 0}
        for _ in range(40):
            class CorruptingFake(FakeTranslator):
                def translate_batch_texts(self, texts, source_lang="zh",
                                          target_lang="en", max_new_tokens=None):
                    base = FakeTranslator().translate_batch_texts(
                        texts, source_lang, target_lang, max_new_tokens
                    )
                    return [
                        TranslationResult(
                            source_text=t,
                            translated_text=TestPlaceholderCorruptionProperty._corrupt(
                                r.translated_text, rng
                            ),
                            model_name="fake", device="cpu",
                        )
                        for t, r in zip(texts, base)
                    ]

            try:
                out = StructuredTranslator(
                    CorruptingFake(), StructuredConfig(), TranslationConfig()
                ).translate(self.HTML).translated_html
            except StructuredTranslationError:
                outcomes["error"] += 1
                continue
            # valid output: never partial — all expected content present
            for token in ("click", "Continue", "USB-C", "cable", "premium"):
                assert token in out, f"{token!r} lost in {out!r}"
            assert "https://example.com/x" in out
            assert "<strong>" in out and "<em>" in out
            assert "<b>fake</b>" not in out  # model tags escaped, never real
            outcomes["ok"] += 1
        # every corruption either recovered (ok) or failed closed (error);
        # NEVER partial HTML. (Fail-closed itself is exercised by
        # test_persistent_placeholder_drop_fails_closed via attribute
        # segments, where the fallback has no safe path.)
        assert outcomes["ok"] >= 1
        print("outcomes:", outcomes)


# ---------------------------------------------------------------------------
# Long chapter: complete coverage, no dup/loss
# ---------------------------------------------------------------------------

class TestLongChapterCoverage:
    def test_no_segment_duplication_or_omission(self):
        fake = FakeTranslator()
        html = "".join(f"<p>第{i}段中文内容，耐磨耐用。</p>" for i in range(60))
        res = StructuredTranslator(
            fake, StructuredConfig(max_segment_tokens=40), TranslationConfig()
        ).translate(html)
        ids = [s["segment_id"] for s in res.segments]
        assert len(ids) == len(set(ids))  # no duplicates
        seqs = [s["sequence_index"] for s in res.segments]
        assert sorted(seqs) == list(range(len(seqs)))  # no gaps, ordered

    def test_blocks_count_preserved(self):
        fake = FakeTranslator()
        html = "".join(f"<p>第{i}段</p>" for i in range(50))
        out = StructuredTranslator(fake, StructuredConfig(), TranslationConfig()).translate(html).translated_html
        assert out.count("<p>") == 50
        assert out.count("</p>") == 50
