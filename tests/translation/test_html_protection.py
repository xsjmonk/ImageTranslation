"""Tests for placeholder protection and validation."""

from __future__ import annotations

import pytest

from image_translation.translation.html_protection import (
    DEFAULT_PREFIX,
    ProtectionMap,
    assert_prefix_safe,
)
from image_translation.translation.exceptions import StructuredTranslationError


class TestProtectionMap:
    def test_reserve_and_tokens(self):
        pm = ProtectionMap()
        t1 = pm.reserve("<strong>", kind="tag_start")
        t2 = pm.reserve("https://x.com", kind="span")
        assert t1.startswith(DEFAULT_PREFIX)
        assert t2.startswith(DEFAULT_PREFIX)
        assert t1 != t2
        assert pm.tokens == [t1, t2]

    def test_restore(self):
        pm = ProtectionMap()
        t1 = pm.reserve("<strong>", kind="tag_start")
        out = f"abc {t1} def"
        assert pm.restore(out) == "abc <strong> def"

    def test_validate_ok(self):
        pm = ProtectionMap()
        t1 = pm.reserve("<strong>", kind="tag_start")
        t2 = pm.reserve("</strong>", kind="tag_end")
        check = pm.validate_output(f"a {t1} b {t2} c")
        assert check["ok"] is True
        assert check["order_ok"] is True

    def test_validate_missing(self):
        pm = ProtectionMap()
        t1 = pm.reserve("<strong>", kind="tag_start")
        check = pm.validate_output("a b c")
        assert check["ok"] is False
        assert any("missing" in i for i in check["issues"])

    def test_validate_duplicate(self):
        pm = ProtectionMap()
        t1 = pm.reserve("<strong>", kind="tag_start")
        check = pm.validate_output(f"{t1} {t1}")
        assert check["ok"] is False
        assert any("duplicated" in i for i in check["issues"])

    def test_validate_tag_order(self):
        pm = ProtectionMap()
        t1 = pm.reserve("<strong>", kind="tag_start")
        t2 = pm.reserve("</strong>", kind="tag_end")
        check = pm.validate_output(f"a {t2} b {t1} c")
        assert check["ok"] is False
        assert check["order_ok"] is False

    def test_restore_split_basic(self):
        pm = ProtectionMap()
        t1 = pm.reserve("<strong>", kind="tag_start")
        t2 = pm.reserve("</strong>", kind="tag_end")
        pieces, tags = pm.restore_split(f"x {t1} y {t2} z")
        assert pieces == ["x ", " y ", " z"]
        assert [s.content for s in tags] == ["<strong>", "</strong>"]

    def test_restore_split_span_inline(self):
        """restore_split splits at EVERY placeholder (spans included) and
        returns the protected sequence in first-occurrence order."""
        pm = ProtectionMap()
        span = pm.reserve("https://x.com", kind="span")
        tag = pm.reserve("<b>", kind="tag_start")
        pieces, seq = pm.restore_split(f"a {span} b {tag} c")
        assert pieces == ["a ", " b ", " c"]
        assert [s.content for s in seq] == ["https://x.com", "<b>"]


class TestPrefixSafety:
    def test_prefix_collision_raises(self):
        with pytest.raises(StructuredTranslationError, match="collides"):
            assert_prefix_safe(f"text {DEFAULT_PREFIX} more")

    def test_clean_text_passes(self):
        assert_prefix_safe("plain 中文 text")  # no raise
