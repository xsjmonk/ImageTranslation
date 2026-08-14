"""Production batching tests: true first-pass batching of segments.

Proves:
- one batch call carries multiple segments, source order preserved;
- batch items are validated independently; failed items are retried
  individually (stricter prefix) and then split fallback; successful items
  are never re-sent;
- whole-batch crashes fall back to per-segment calls;
- long chapters are processed with fewer model calls than the sequential
  baseline, with exact markup/entity/identifier preservation;
- direct module and HTTP API produce equivalent HTML through the same
  batched path.
"""

from __future__ import annotations

import re
import time

import pytest

from image_translation.translation.base import Translator
from image_translation.translation.config import StructuredConfig, TranslationConfig
from image_translation.translation.exceptions import StructuredTranslationError
from image_translation.translation.models import TranslationResult
from image_translation.translation.structured_translation import StructuredTranslator

import image_translation.translation.structured_translation as st_mod

_TOKEN_RE = r"__IT[A-Z0-9]*_[A-Z]\d{4}_"


class RecordingFake(Translator):
    """Records batch call sizes and corrupts outputs deterministically.

    ``corrupt_items``: 1-based item indices (across ALL calls) whose output
    is corrupted in the given mode (swap/drop/duplicate/invent).
    ``crash_batches``: 1-based call indices (only for calls with len>1) that
    raise RuntimeError, simulating a whole-batch model crash.
    """

    def __init__(self, corrupt_items=(), mode="swap", crash_batches=(),
                 count_behaviors=None):
        self.corrupt_items = set(corrupt_items)
        self.mode = mode
        self.crash_batches = set(crash_batches)
        # call index -> 'zero' | 'short' | 'extra' | 'none' | 'malformed'
        self.count_behaviors = count_behaviors or {}
        self.call_count = 0
        self.call_sizes: list[int] = []
        self.call_texts: list[list[str]] = []
        self.budgets: list = []
        self.item_count = 0
        self.elapsed = 0.0
    @property
    def name(self) -> str:
        return "recording-fake"

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
        start = time.monotonic()
        self.call_count += 1
        self.call_sizes.append(len(texts))
        self.call_texts.append(list(texts))
        self.budgets.append(max_new_tokens)
        if self.call_count in self.crash_batches:
            self.elapsed += time.monotonic() - start
            raise RuntimeError("gpu exploded (batch)")

        out = []
        for t in texts:
            self.item_count += 1
            n = self.item_count
            translated = re.sub(
                r"[\u4e00-\u9fff]+", lambda m: "EN:" + m.group(0), t
            )
            if n in self.corrupt_items:
                translated = self._corrupt(translated)
            out.append(TranslationResult(
                source_text=t, translated_text=translated,
                model_name="fake", device="cpu",
            ))
        behavior = self.count_behaviors.get(self.call_count)
        if behavior == "zero":
            out = []
        elif behavior == "short":
            out = out[:-1]
        elif behavior == "extra":
            out = out + [out[0]]
        elif behavior == "none":
            out[0] = None
        elif behavior == "malformed":
            out[0] = TranslationResult(
                source_text=texts[0], translated_text=None,
                model_name="fake", device="cpu",
            )
        self.elapsed += time.monotonic() - start
        return out

    def _corrupt(self, text: str) -> str:
        parts = re.split(r"(" + _TOKEN_RE + r")", text)
        tokens = [p for p in parts if re.fullmatch(_TOKEN_RE, p)]
        body = [p for p in parts if not re.fullmatch(_TOKEN_RE, p)]

        def rebuild(toks) -> str:
            result = body[0]
            for i, tok in enumerate(toks):
                result += tok
                if i + 1 < len(body):
                    result += body[i + 1]
            return result

        if self.mode == "swap" and len(tokens) >= 2:
            tokens[0], tokens[1] = tokens[1], tokens[0]
            return rebuild(tokens)
        if self.mode == "drop" and tokens:
            return rebuild(tokens[:-1])
        if self.mode == "duplicate" and tokens:
            return rebuild(tokens + [tokens[0]])
        if self.mode == "invent":
            return text + "__ITRANSLATE_X9999_"
        return text


@pytest.fixture(autouse=True)
def fake_measure(monkeypatch):
    class FakeTok:
        def __call__(self, text, truncation=False):
            return {"input_ids": list(range(max(1, (len(text) + 1) // 2)))}
    monkeypatch.setattr(st_mod, "_get_measure_tokenizer", lambda *a, **k: FakeTok())


MIXED_CHAPTER = """<h1>产品介绍</h1>
<p>这是<strong>加厚防水面料</strong>制作的背包，型号 ABC-12345，支持 USB-C 快充。</p>
<p>Visit https://example.com/x?q=1 for details。中文 English 混合内容&nbsp;与&#160;实体。</p>
<div class="notranslate">品牌 BrandName 保持不变</div>
<script>var x = "&nbsp;";</script>
<p>Made with premium materials for durability.</p>
<p>前&nbsp;中文&#160;中间&amp;中文&#xA0;结尾。<br/>换行后继续。</p>
<p>最后一句话。</p>
"""


def _par(i: int) -> str:
    """One paragraph sized to be EXACTLY one segment under
    max_segment_tokens=30 (approx 28 tokens) with 3 protected tokens
    (the paragraph index digit, ABC-123, USB-C) so swap corruptions
    are meaningful. Two paragraphs (56 tokens) never merge."""
    return (f"<p>第{i}段" + "中文内容，耐磨耐用。" * 4
            + "型号 ABC-123，支持 USB-C。</p>")


def _assert_mixed_guarantees(out: str) -> None:
    """Structural, entity, identifier, and leak guarantees for MIXED_CHAPTER
    (counts compared against the source; the script string literal counts)."""
    src = MIXED_CHAPTER
    for tag in ("<p>", "<h1>", "<strong>", "<br/>"):
        assert out.count(tag) == src.count(tag), (
            f"{tag}: {out.count(tag)} vs {src.count(tag)}"
        )
    for ent in ("&nbsp;", "&#160;", "&amp;", "&#xA0;"):
        assert out.count(ent) == src.count(ent), (
            f"{ent}: {out.count(ent)} vs {src.count(ent)}"
        )
    assert "ABC-12345" in out
    assert "USB-C" in out
    assert "https://example.com/x?q=1" in out
    assert "BrandName 保持不变" in out  # excluded, byte-identical
    assert 'var x = "&nbsp;";' in out   # script, byte-identical
    assert "Made with premium materials for durability." in out
    assert "__IT" not in out
    assert "\x02" not in out and "\x03" not in out


def _assert_long_guarantees(out: str, html: str) -> None:
    """Structural, entity, identifier, and leak guarantees for the long
    chapter fixture (counts compared against the input)."""
    for tag in ("<p>", "<h1>", "<h2>", "<strong>", "<br/>", "<a "):
        assert out.count(tag) == html.count(tag), (
            f"{tag}: {out.count(tag)} vs {html.count(tag)}"
        )
    for ent in ("&nbsp;", "&#160;", "&amp;", "&#xA0;", "&lt;"):
        assert out.count(ent) == html.count(ent), (
            f"{ent}: {out.count(ent)} vs {html.count(ent)}"
        )
    for ident in ("ABC-123", "USB-C", "Windows 11", "X13", "X1300"):
        assert out.count(ident) == html.count(ident), (
            f"{ident}: {out.count(ident)} vs {html.count(ident)}"
        )
    assert "END-OF-CHAPTER 保持不变" in out
    assert ("<p>Model ABC-123 uses the USB-C interface and works with "
            "Windows 11.</p>") in out
    assert "__IT" not in out
    assert "\x02" not in out and "\x03" not in out


class TestTrueBatching:
    def test_one_batch_contains_multiple_segments(self):
        fake = RecordingFake()
        # 12 paragraphs -> 24 segments (each paragraph splits into a
        # digit part and an identifier part under max_segment_tokens=30);
        # batches of 4 -> 6 batch calls, all full.
        html = "".join(_par(i) for i in range(12))
        cfg = StructuredConfig(max_segment_tokens=30, batch_size=4)
        res = StructuredTranslator(fake, cfg, TranslationConfig()).translate(html)
        assert res.segment_count == 24
        assert fake.call_count == 6
        assert fake.call_sizes == [4, 4, 4, 4, 4, 4]
        assert res.batch_count == 6
        assert sum(fake.call_sizes) == res.segment_count
        # source order preserved
        for i in range(12):
            assert f"EN:第{i}EN:段" in res.translated_html

    def test_batch_source_order_preserved_after_output(self):
        fake = RecordingFake()
        res = StructuredTranslator(
            fake, StructuredConfig(max_segment_tokens=30, batch_size=4),
            TranslationConfig(),
        ).translate(MIXED_CHAPTER)
        out = res.translated_html
        _assert_mixed_guarantees(out)
        # Chinese translated in original slots, English/identifiers exact
        assert "<strong>EN:加厚防水面料</strong>" in out
        assert "型号 ABC-12345" in out
        assert "EN:中文 English EN:混合内容" in out
        # first paragraph survives
        assert out.startswith("<h1>EN:产品介绍</h1>")
        # last paragraph survives
        assert "EN:最后一句话" in out and out.rstrip().endswith("</p>")
        assert res.retry_count == 0 and res.fallback_count == 0

    def test_batch_calls_share_one_model_call_per_batch(self):
        fake = RecordingFake()
        html = "".join(_par(i) for i in range(12))
        res = StructuredTranslator(
            fake, StructuredConfig(max_segment_tokens=30, batch_size=4),
            TranslationConfig(),
        ).translate(html)
        assert res.segment_count == 24
        assert fake.call_count == 6  # 24 segments / batch of 4
        # every batch call carried multiple items
        assert all(s > 1 for s in fake.call_sizes)
        assert max(fake.call_sizes) <= 4
        assert res.batch_count == 6


class TestBatchFailureHandling:
    def test_one_corrupt_item_among_valid_retried_individually(self):
        fake = RecordingFake(corrupt_items={2}, mode="swap")
        # 4 paragraphs -> 8 segments -> 2 batches of 4. Item 2 is the
        # identifier part of paragraph 0 (2 placeholders) -> corrupted.
        html = "".join(_par(i) for i in range(4))
        res = StructuredTranslator(
            fake, StructuredConfig(max_segment_tokens=30, batch_size=4),
            TranslationConfig(),
        ).translate(html)
        # batch 1 (4 items) + 1 individual retry for the corrupt item +
        # batch 2 (4 items)
        assert fake.call_count == 3
        assert fake.call_sizes == [4, 1, 4]
        assert res.retry_count == 1
        assert res.fallback_count == 0
        for i in range(4):
            assert f"EN:第{i}EN:段" in res.translated_html
        assert res.segment_count == 8

    def test_drop_duplicate_reorder_invent_in_one_item(self):
        for mode in ("drop", "duplicate", "swap", "invent"):
            fake = RecordingFake(corrupt_items={2}, mode=mode)
            html = "".join(_par(i) for i in range(4))
            res = StructuredTranslator(
                fake, StructuredConfig(max_segment_tokens=30, batch_size=4),
                TranslationConfig(),
            ).translate(html)
            # corrupt item recovered individually (retry); others from batch
            assert res.retry_count == 1, f"mode={mode}: {res.retry_count}"
            assert res.fallback_count == 0, f"mode={mode}"
            assert "__IT" not in res.translated_html
            for i in range(4):
                assert f"EN:第{i}EN:段" in res.translated_html

    def test_fail_one_retry_succeed_another(self):
        # Item 2 (identifier part of paragraph 0) corrupt on batch (2) and
        # on its retry (5) -> split fallback; item 4 (identifier part of
        # paragraph 1) corrupt on batch (4) only -> retry (6) succeeds.
        fake = RecordingFake(corrupt_items={2, 4, 5}, mode="swap")
        html = "".join(_par(i) for i in range(4))
        res = StructuredTranslator(
            fake, StructuredConfig(max_segment_tokens=30, batch_size=4),
            TranslationConfig(),
        ).translate(html)
        assert res.retry_count >= 1
        assert res.fallback_count >= 1
        for i in range(4):
            assert f"EN:第{i}EN:段" in res.translated_html

    def test_force_split_fallback(self):
        # Item 2 corrupt on batch (2) AND on its retry (5) -> fallback
        fake = RecordingFake(corrupt_items={2, 5}, mode="swap")
        html = "".join(_par(i) for i in range(4))
        res = StructuredTranslator(
            fake, StructuredConfig(max_segment_tokens=30, batch_size=4),
            TranslationConfig(),
        ).translate(html)
        assert res.fallback_count >= 1
        for i in range(4):
            assert f"EN:第{i}EN:段" in res.translated_html

    def test_whole_batch_crash_retried_individually(self):
        # 4 paragraphs -> 8 segments -> 2 batches. The first batch call
        # (4 items) crashes; every item retried as its own single call;
        # the second batch proceeds normally.
        fake = RecordingFake(crash_batches={1})
        html = "".join(_par(i) for i in range(4))
        res = StructuredTranslator(
            fake, StructuredConfig(max_segment_tokens=30, batch_size=4),
            TranslationConfig(),
        ).translate(html)
        assert fake.call_count == 6  # 1 crashed batch + 4 singles + batch 2
        assert fake.call_sizes == [4, 1, 1, 1, 1, 4]
        assert res.segment_count == 8
        for i in range(4):
            assert f"EN:第{i}EN:段" in res.translated_html
        # crash is recovered, not reported as a retry of a validated attempt
        assert res.fallback_count == 0

    def test_persistent_crash_fails_closed(self):
        # Every call crashes -> fail closed, no partial HTML.
        fake = RecordingFake(crash_batches={1, 2, 3, 4, 5, 6, 7, 8})
        html = _par(0)
        with pytest.raises(StructuredTranslationError):
            StructuredTranslator(
                fake, StructuredConfig(), TranslationConfig()
            ).translate(html)


class TestConfigValidation:
    def test_batch_size_must_be_positive(self):
        with pytest.raises(ValueError, match="batch_size"):
            StructuredConfig(batch_size=0)

    def test_batch_size_one_is_sequential_path(self):
        fake = RecordingFake()
        # 4 paragraphs -> 8 segments; batch_size=1 -> one call per segment
        html = "".join(_par(i) for i in range(4))
        res = StructuredTranslator(
            fake, StructuredConfig(max_segment_tokens=30, batch_size=1),
            TranslationConfig(),
        ).translate(html)
        assert fake.call_sizes == [1] * 8
        assert res.batch_count == 8
        assert res.segment_count == 8


def _build_long_chapter() -> str:
    parts = ["<h1>充电器产品说明书</h1>"]
    base = (
        "这款便携式充电器采用加厚防火外壳，内部具备过流保护和短路保护功能。"
        "产品坚固耐用，适合日常通勤和旅行使用。"
    )
    for i in range(60):
        if i % 3 == 0:
            parts.append(f"<h2>第{i + 1}节</h2>")
        if i == 5:
            parts.append(f"<p><strong>加厚防水面料</strong>，{base}</p>")
        elif i == 7:
            parts.append(
                f"<p>{base} 型号 ABC-123，支持 USB-C 接口，兼容 Windows 11。</p>"
            )
        elif i == 13:
            parts.append(
                "<p>型号 X13&nbsp;与 X1300&#160;兼容，功耗 &lt; 5W&amp;稳定，"
                "支持 USB-C。<br/>更多说明&#xA0;见文档。</p>"
            )
        elif i == 17:
            parts.append(
                '<p>更多信息请访问 <a href="/manual?p=2&amp;v=3">在线文档</a>'
                " 或联系支持。<br>联系电话见附录。</p>"
            )
        else:
            parts.append(f"<p>{base}</p>")
    parts.append("<div class='notranslate'>END-OF-CHAPTER 保持不变</div>")
    parts.append(
        "<p>Model ABC-123 uses the USB-C interface and works with Windows 11.</p>"
    )
    return "\n".join(parts)


class TestLongChapterBatchingBenchmark:
    def test_batched_long_chapter_reduces_model_calls(self):
        fake = RecordingFake()
        html = _build_long_chapter()
        assert len(html) > 4000
        assert html.count("<p>") >= 50
        cfg = StructuredConfig(max_segment_tokens=30, batch_size=4)
        start = time.monotonic()
        res = StructuredTranslator(
            fake, cfg, TranslationConfig(), document_id="bench"
        ).translate(html)
        elapsed = time.monotonic() - start
        out = res.translated_html

        # ---- exactness guarantees ----
        _assert_long_guarantees(out, html)
        # first/middle/last: h1 processed through the model in its slot
        assert "EN:充电器产品说明书" in out
        assert out.find("X1300") > out.find("X13")
        assert out.find("END-OF-CHAPTER") > out.find("EN:充电器")
        # no truncation: every segment present, unique, ordered
        ids = [s["segment_id"] for s in res.segments]
        assert len(ids) == len(set(ids))
        seqs = [s["sequence_index"] for s in res.segments]
        assert sorted(seqs) == list(range(len(seqs)))
        # every Chinese run was processed through the model in its slot
        # (fake marks each run with 'EN:' before the Chinese text)
        text_only = re.sub(r"<[^>]+>", "", out.replace("END-OF-CHAPTER 保持不变", ""))
        unmarked = re.sub(r"EN:[\u4e00-\u9fff]+", "", text_only)
        leftover = re.findall(r"[\u4e00-\u9fff]+", unmarked)
        assert not leftover, f"unprocessed CJK: {leftover[:5]}"

        # ---- call-count benchmark ----
        segment_count = res.segment_count
        batch_calls = fake.call_count
        max_batch = max(fake.call_sizes)
        sequential_baseline = segment_count  # one model call per segment
        assert segment_count > 10
        assert batch_calls < sequential_baseline, (
            f"batching did not reduce calls: {batch_calls} vs baseline "
            f"{sequential_baseline}"
        )
        assert max_batch <= 4
        assert res.batch_count == batch_calls
        assert res.retry_count == 0 and res.fallback_count == 0
        print(
            f"\n=== BENCHMARK === segments={segment_count} "
            f"batch_calls={batch_calls} sequential_baseline={sequential_baseline} "
            f"call_reduction={sequential_baseline - batch_calls} "
            f"max_batch_size={max_batch} batch_count={res.batch_count} "
            f"retries={res.retry_count} fallbacks={res.fallback_count} "
            f"elapsed={elapsed:.2f}s (fake) ==="
        )


class TestBatchCardinality:
    """Batch result-count invariants: zero/fewer/extra/None/malformed
    results are never silently zipped away — every affected segment is
    recovered individually, and unrecoverable cases fail closed."""

    HTML4 = "".join(_par(i) for i in range(4))  # 8 segments

    def _run(self, behaviors):
        fake = RecordingFake(count_behaviors=behaviors)
        res = StructuredTranslator(
            fake, StructuredConfig(max_segment_tokens=30, batch_size=4),
            TranslationConfig(),
        ).translate(self.HTML4)
        return fake, res

    def test_zero_outputs_recovered_individually(self):
        fake, res = self._run({1: "zero"})
        assert res.segment_count == 8
        for i in range(4):
            assert f"EN:第{i}EN:段" in res.translated_html
        assert "__IT" not in res.translated_html

    def test_fewer_outputs_recovered_individually(self):
        fake, res = self._run({1: "short"})
        assert res.segment_count == 8
        for i in range(4):
            assert f"EN:第{i}EN:段" in res.translated_html

    def test_extra_outputs_recovered_individually(self):
        fake, res = self._run({1: "extra"})
        assert res.segment_count == 8
        for i in range(4):
            assert f"EN:第{i}EN:段" in res.translated_html

    def test_none_output_recovered_individually(self):
        fake, res = self._run({1: "none"})
        assert res.segment_count == 8
        for i in range(4):
            assert f"EN:第{i}EN:段" in res.translated_html

    def test_malformed_output_recovered_individually(self):
        fake, res = self._run({1: "malformed"})
        assert res.segment_count == 8
        for i in range(4):
            assert f"EN:第{i}EN:段" in res.translated_html

    def test_persistent_cardinality_failure_fails_closed(self):
        # Every call returns zero results -> recovery cannot prove each
        # segment translated exactly once -> fail closed, no partial HTML.
        fake = RecordingFake(
            count_behaviors={i: "zero" for i in range(1, 60)}
        )
        with pytest.raises(StructuredTranslationError, match="results|count"):
            StructuredTranslator(
                fake, StructuredConfig(max_segment_tokens=30, batch_size=4),
                TranslationConfig(),
            ).translate(self.HTML4)


class TestBudgetGrouping:
    """Short/mid/long segments must not all run with the largest target
    budget: quantized buckets keep short segments cheap, and metrics
    distinguish requested budgets from the generation budget."""

    def test_short_and_long_segments_use_different_budgets(self):
        fake = RecordingFake()
        short = "<p>短文本内容。</p>"                                     # 3 tok -> 64
        mid = "<p>" + "中文内容，耐磨耐用。" * 6 + "中段结束。</p>"        # 31 tok -> 77 -> 128
        long_p = "<p>" + "中文内容，耐磨耐用。" * 40 + "长段结束。</p>"    # 200 tok -> 400
        html = short + mid + long_p + short + mid + long_p
        res = StructuredTranslator(
            fake, StructuredConfig(max_segment_tokens=450, batch_size=4),
            TranslationConfig(),
        ).translate(html)

        # 6 blocks -> 6 segments -> chunk1 [s,m,l,s], chunk2 [m,l].
        # Buckets: 64 -> 64, 77 -> 128, 400 -> 400 -> 5 batch calls.
        assert fake.budgets == [64, 128, 400, 128, 400]
        assert res.batch_count == 5
        for texts, budget in zip(fake.call_texts, fake.budgets):
            joined = "".join(texts)
            if "短文本内容" in joined:
                assert budget == 64, f"short segment ran with budget {budget}"
            if "中段结束" in joined:
                assert budget == 128, f"mid segment ran with budget {budget}"
            if "长段结束" in joined:
                assert budget == 400

        # Metrics: requested vs generation budgets are distinct and truthful
        # (bucket 128 raises the 82 requests; 64/400 stay exact).
        assert res.sum_requested_target_tokens == 64 * 2 + 82 * 2 + 400 * 2
        assert res.batch_generation_budget == 64 * 2 + 128 * 2 + 400 * 2
        assert res.sum_requested_target_tokens < res.batch_generation_budget
        assert len(res.batch_metrics) == 5
        for m in res.batch_metrics:
            assert set(m) >= {"batch_index", "items", "max_target_budget",
                              "per_segment_budgets", "per_segment_buckets",
                              "source_tokens", "elapsed_seconds"}
            assert m["max_target_budget"] == max(m["per_segment_buckets"])
            assert all(b <= m["max_target_budget"]
                       for b in m["per_segment_budgets"])
        # every segment present, in order, exact content
        assert res.translated_html.count("EN:短文本内容") == 2
        assert res.translated_html.count("EN:中文内容") == 2 * 40 + 2 * 6
        assert "__IT" not in res.translated_html


class TestDirectApiBatchedParity:
    def test_direct_vs_api_batched_html_identical(self):
        from fastapi.testclient import TestClient

        from translation_server.app import create_app
        from translation_server.runtime import TranslationRuntime, TranslationServerConfig

        fake = RecordingFake()
        html = MIXED_CHAPTER

        # Direct module
        direct = StructuredTranslator(
            fake, StructuredConfig(batch_size=4), TranslationConfig(),
            document_id="parity",
        ).translate(html).translated_html
        direct_calls = fake.call_count

        # Live HTTP API through the same shared path
        cfg = TranslationServerConfig()
        cfg.structured.batch_size = 4
        runtime = TranslationRuntime(cfg)
        runtime._translator = RecordingFake()
        app = create_app(runtime)
        client = TestClient(app)
        resp = client.post(
            "/translate", json={"text": html, "format": "html"}
        )
        assert resp.status_code == 200, resp.text
        api_out = resp.json()["translation"]
        assert api_out == direct
        assert runtime._translator.call_count == direct_calls
        assert runtime._translator.call_sizes == fake.call_sizes
