"""Production slot-order regression tests.

Strict contract: the complete protected-placeholder sequence in model
output must equal the source placeholder order EXACTLY — reordering within a
tag interval, entity swaps, English/model-number swaps, deletions,
duplications, invented placeholder-like tokens (default or retry prefixes),
and fake markup are all rejected or recovered with exact source-order
restoration. Every corruption must end in either exact recovery or a
fail-closed StructuredTranslationError — never partial HTML.
"""

from __future__ import annotations

import re

import pytest

from image_translation.translation.base import Translator
from image_translation.translation.config import StructuredConfig, TranslationConfig
from image_translation.translation.exceptions import StructuredTranslationError
from image_translation.translation.models import TranslationResult
from image_translation.translation.structured_translation import StructuredTranslator


# Matches every placeholder token the project can produce (default + retry
# prefixes): __IT<prefix>_<KIND><4 digits>_
_TOKEN_RE = r"__IT[A-Z0-9]*_[A-Z]\d{4}_"

# The exact mandated case: mixed Chinese/English/model numbers + entities.
SLOT_HTML = (
    "<p>中文 X13 与 English A 中文 X1300 与 English B。</p>"
    "<p>前&nbsp;中文&#160;中间&amp;中文&#xA0;结尾。</p>"
    "<p>中文 100W 与 English D 中文 5V 与 English E。</p>"
)

SLOT_ASSERTS = [
    ("X13", "English A"),
    ("English A", "X1300"),
    ("X1300", "English B"),
]
ENTITY_ORDER = ["&nbsp;", "&#160;", "&amp;", "&#xA0;"]
MEASUREMENT_ASSERTS = [
    ("100W", "English D"),
    ("English D", "5V"),
    ("5V", "English E"),
]


def _assert_source_order(out: str) -> None:
    """All protected content in exact source order."""
    for before, after in SLOT_ASSERTS:
        assert out.find(before) < out.find(after), (
            f"{before!r} must appear before {after!r} in {out!r}"
        )
    # The real &nbsp; entity is never preceded by &amp; (escaped model
    # &nbsp; renders as &amp;nbsp; and must not be confused with it), and
    # the real &amp; entity is never followed by 'nbsp;'.
    nbsp = re.search(r"(?<!&amp;)&nbsp;", out)
    amp = re.search(r"&amp;(?!nbsp;)", out)
    assert nbsp is not None, f"real &nbsp; entity missing in {out!r}"
    assert amp is not None, f"real &amp; entity missing in {out!r}"
    positions = [nbsp.start(), out.find("&#160;"), amp.start(), out.find("&#xA0;")]
    assert all(p >= 0 for p in positions), f"entity missing in {out!r}"
    assert positions == sorted(positions), (
        f"entities out of source order in {out!r}"
    )


class SlotCorruptingFake(Translator):
    """Wraps CJK with 'EN:' then applies one corruption mode to the output.

    Corruption is applied deterministically to EVERY model call so retries
    cannot silently pass; recovery must come from the split fallback or the
    request must fail closed.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.call_count = 0
        self.corruptions_applied = 0
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "slot-fake"

    @property
    def runtime_info(self):
        return None


    def measure_source_tokens(self, text: str, source_lang: str = "zh") -> int:
        """Token count used by HTML segmentation (no model call)."""
        return max(1, (len(text) + 1) // 2)
    def translate_text(
        self,
        text,
        source_lang="zh",
        target_lang="en",
        max_new_tokens=None,
        style=None,
    ):
        return self.translate_batch_texts(
            [text], source_lang, target_lang, max_new_tokens, style=style
        )[0]

    def translate_batch_texts(
        self,
        texts,
        source_lang="zh",
        target_lang="en",
        max_new_tokens=None,
        style=None,
    ):
        out = []
        for t in texts:
            self.call_count += 1
            self.calls.append(t)
            translated = re.sub(r"[\u4e00-\u9fff]+", lambda m: "EN:" + m.group(0), t)
            translated = self._corrupt(translated)
            out.append(TranslationResult(
                source_text=t, translated_text=translated,
                model_name="fake", device="cpu",
            ))
        return out

    # ------------------------------------------------------------------

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

        mode = self.mode

        if mode == "crash":
            raise RuntimeError("gpu exploded")

        if mode == "invent_default":
            self.corruptions_applied += 1
            return text + "__ITRANSLATE_X9999_"

        if mode == "invent_retry":
            self.corruptions_applied += 1
            return text + "__ITABCD_X9999_"

        if mode == "fake_markup":
            # Emit raw <br> and &nbsp; as ordinary text after CJK.
            self.corruptions_applied += 1
            return re.sub(
                r"EN:[\u4e00-\u9fff]+",
                lambda m: m.group(0) + "<br>&nbsp;",
                text,
            )

        if mode == "malformed":
            # Strip every token and inject control characters. Runs without
            # tokens (the split-fallback per-run sources) pass through
            # untouched, so recovery is lossless via the fallback.
            if tokens:
                self.corruptions_applied += 1
                return "".join(body) + "\x02\x03"
            return text

        if not tokens:
            return text

        if mode == "swap_non_tag":
            if len(tokens) >= 2:
                self.corruptions_applied += 1
                tokens[0], tokens[1] = tokens[1], tokens[0]
            return rebuild(tokens)

        if mode == "swap_eng_model":
            # English/entity runs are E-kind; identifier/measurement spans
            # are S-kind. Swapping an S and an E token reorders protected
            # content (e.g. 100W vs English D) around Chinese.
            s_idx = next((i for i, tok in enumerate(tokens)
                          if re.search(r"_S\d{4}_$", tok)), None)
            e_idx = next((i for i, tok in enumerate(tokens)
                          if re.search(r"_E\d{4}_$", tok)), None)
            if s_idx is not None and e_idx is not None:
                self.corruptions_applied += 1
                tokens[s_idx], tokens[e_idx] = tokens[e_idx], tokens[s_idx]
            return rebuild(tokens)

        if mode == "swap_entities":
            # Entity runs are E-kind; only segments WITHOUT identifier/
            # measurement spans (no S tokens) are targeted so the swap hits
            # the pure-entity paragraph (&nbsp;/&#160;) exactly.
            e_idxs = [i for i, tok in enumerate(tokens)
                      if re.search(r"_E\d{4}_$", tok)]
            if len(e_idxs) >= 2 and not any(
                re.search(r"_S\d{4}_$|_M\d{4}_$", tok) for tok in tokens
            ):
                self.corruptions_applied += 1
                tokens[e_idxs[0]], tokens[e_idxs[1]] = (
                    tokens[e_idxs[1]], tokens[e_idxs[0]]
                )
            return rebuild(tokens)

        if mode == "delete":
            self.corruptions_applied += 1
            return rebuild(tokens[:-1])

        if mode == "duplicate":
            self.corruptions_applied += 1
            return rebuild(tokens + [tokens[0]])

        return text


def _run(mode: str, html: str = SLOT_HTML):
    fake = SlotCorruptingFake(mode)
    st = StructuredTranslator(fake, StructuredConfig(), TranslationConfig())
    return fake, st.translate(html).translated_html


class TestReorderedProtectedPlaceholdersRejected:
    def test_swap_two_non_tag_placeholders_recovered_in_source_order(self):
        fake, out = _run("swap_non_tag")
        assert fake.corruptions_applied >= 1
        _assert_source_order(out)
        # Chinese translated in the original slots
        assert "EN:中文" in out

    def test_swap_english_and_model_number_recovered_in_source_order(self):
        fake, out = _run("swap_eng_model")
        assert fake.corruptions_applied >= 1
        _assert_source_order(out)
        # measurements (S-kind spans) and English spans (E-kind) stay in order
        for before, after in MEASUREMENT_ASSERTS:
            assert out.find(before) < out.find(after), (
                f"{before!r} must appear before {after!r} in {out!r}"
            )

    def test_swap_entity_placeholders_recovered_in_source_order(self):
        fake, out = _run("swap_entities")
        assert fake.corruptions_applied >= 1, "entity swap never fired"
        _assert_source_order(out)
        assert out.count("&nbsp;") == 1
        assert out.count("&#160;") == 1
        assert out.count("&amp;") == 1
        assert out.count("&#xA0;") == 1

    def test_no_placeholder_or_control_leak(self):
        fake, out = _run("swap_non_tag")
        assert "__IT" not in out
        assert "\x02" not in out and "\x03" not in out


class TestMissingDuplicatePlaceholders:
    def test_deleted_placeholder_recovered_in_source_order(self):
        fake, out = _run("delete")
        assert fake.corruptions_applied >= 1
        _assert_source_order(out)

    def test_duplicated_placeholder_recovered_in_source_order(self):
        fake, out = _run("duplicate")
        assert fake.corruptions_applied >= 1
        _assert_source_order(out)


class TestInventedPlaceholdersRejected:
    def test_invented_default_prefix_fails_closed(self):
        fake = SlotCorruptingFake("invent_default")
        st = StructuredTranslator(fake, StructuredConfig(), TranslationConfig())
        with pytest.raises(StructuredTranslationError):
            st.translate(SLOT_HTML)
        assert fake.corruptions_applied >= 1

    def test_invented_retry_prefix_fails_closed(self):
        fake = SlotCorruptingFake("invent_retry")
        st = StructuredTranslator(fake, StructuredConfig(), TranslationConfig())
        with pytest.raises(StructuredTranslationError):
            st.translate(SLOT_HTML)
        assert fake.corruptions_applied >= 1


class TestFakeMarkupAndMalformed:
    def test_fake_br_and_nbsp_escaped_never_become_markup(self):
        fake, out = _run("fake_markup")
        assert fake.corruptions_applied >= 1
        _assert_source_order(out)
        # model-emitted <br>/&nbsp; became escaped TEXT, never tags/entities
        assert "&lt;br&gt;" in out
        assert "&amp;nbsp;" in out
        assert len(re.findall(r"<br(?!/)>", out)) == 0
        assert "__IT" not in out

    def test_malformed_output_recovered_losslessly(self):
        """Segment-level malformed output (tokens stripped, control chars
        injected) fails strict validation and is recovered exactly by the
        split fallback — never partial HTML, no control chars."""
        fake, out = _run("malformed")
        assert fake.corruptions_applied >= 1
        _assert_source_order(out)
        for before, after in MEASUREMENT_ASSERTS:
            assert out.find(before) < out.find(after)
        assert "\x02" not in out and "\x03" not in out
        assert "__IT" not in out

    def test_crashing_model_fails_closed(self):
        fake = SlotCorruptingFake("crash")
        st = StructuredTranslator(fake, StructuredConfig(), TranslationConfig())
        with pytest.raises(StructuredTranslationError):
            st.translate(SLOT_HTML)


class TestUnitValidateOutput:
    """Direct ProtectionMap-level checks of the strict sequence contract."""

    def test_within_interval_reorder_rejected(self):
        from image_translation.translation.html_protection import ProtectionMap

        pm = ProtectionMap()
        t1 = pm.reserve("X13", kind="english")
        t2 = pm.reserve("English A", kind="english")
        t3 = pm.reserve("X1300", kind="model")
        t4 = pm.reserve("English B", kind="english")
        # source order: t1 t2 t3 t4
        check = pm.validate_output(
            f"a {t1} b {t3} c {t2} d {t4} e",  # t3/t2 swapped WITHIN the p
            expected_order=[t1, t2, t3, t4],
        )
        assert check["ok"] is False
        assert check["order_ok"] is False
        assert any("sequence changed" in i for i in check["issues"])

    def test_entity_swap_rejected(self):
        from image_translation.translation.html_protection import ProtectionMap

        pm = ProtectionMap()
        t1 = pm.reserve("&nbsp;", kind="entity")
        t2 = pm.reserve("&#160;", kind="entity")
        t3 = pm.reserve("&amp;", kind="entity")
        t4 = pm.reserve("&#xA0;", kind="entity")
        check = pm.validate_output(
            f"x {t1} y {t2} z {t4} w {t3} v",  # last two swapped
            expected_order=[t1, t2, t3, t4],
        )
        assert check["ok"] is False
        assert check["order_ok"] is False

    def test_exact_order_passes(self):
        from image_translation.translation.html_protection import ProtectionMap

        pm = ProtectionMap()
        t1 = pm.reserve("X13", kind="english")
        t2 = pm.reserve("X1300", kind="model")
        check = pm.validate_output(
            f"a {t1} b {t2} c", expected_order=[t1, t2]
        )
        assert check["ok"] is True
        assert check["order_ok"] is True

    def test_unknown_default_prefix_token_rejected(self):
        from image_translation.translation.html_protection import ProtectionMap

        pm = ProtectionMap()
        t1 = pm.reserve("X13", kind="english")
        check = pm.validate_output(
            f"a {t1} b __ITRANSLATE_X9999_ c", expected_order=[t1]
        )
        assert check["ok"] is False
        assert any("unknown placeholder invented" in i for i in check["issues"])

    def test_unknown_retry_prefix_token_rejected(self):
        from image_translation.translation.html_protection import ProtectionMap

        pm = ProtectionMap()
        t1 = pm.reserve("X13", kind="english")
        check = pm.validate_output(
            f"a {t1} b __ITABCD_X9999_ c", expected_order=[t1]
        )
        assert check["ok"] is False
        assert any("unknown placeholder invented" in i for i in check["issues"])
