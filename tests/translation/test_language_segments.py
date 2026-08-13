"""Tests for language classification and protected identifiers."""

from __future__ import annotations

import pytest

from image_translation.translation.html_protection import ProtectionMap
from image_translation.translation.language_segments import (
    classify,
    LanguageKind,
    find_protected_spans,
    protect_identifiers,
)


class TestClassify:
    def test_pure_chinese(self):
        assert classify("加厚防水面料") == LanguageKind.CHINESE

    def test_pure_english(self):
        assert classify("Premium Quality Guarantee") == LanguageKind.ENGLISH

    def test_mixed(self):
        assert classify("适合 daily use") == LanguageKind.MIXED

    def test_empty_is_english(self):
        assert classify("") == LanguageKind.ENGLISH

    def test_digits_only_english(self):
        assert classify("12345") == LanguageKind.ENGLISH


class TestProtectedSpans:
    def test_url(self):
        spans = find_protected_spans("看 https://example.com/x?q=1 这里")
        assert any(kind == "url" for _, _, kind, _ in spans)

    def test_email(self):
        spans = find_protected_spans("联系 abc.def@mail.com 我们")
        assert any(kind == "email" for _, _, kind, _ in spans)

    def test_product_code(self):
        spans = find_protected_spans("型号 ABC-12345 可用")
        assert any(kind == "product_code" for _, _, kind, _ in spans)

    def test_usb_c(self):
        spans = find_protected_spans("支持 USB-C 接口")
        assert any(kind == "interface" for _, _, kind, _ in spans)

    def test_windows_version(self):
        spans = find_protected_spans("兼容 Windows 11 系统")
        assert any(kind == "version" for _, _, kind, _ in spans)

    def test_measurement(self):
        spans = find_protected_spans("重量 1.5kg 功率 100W")
        assert sum(1 for _, _, k, _ in spans if k == "measurement") >= 2

    def test_english_untouched(self):
        """English-only text gets no span extraction (whole span preserved)."""
        assert find_protected_spans("Pure English text here") == []


class TestProtectIdentifiers:
    def test_round_trip(self):
        text = "看 https://example.com 支持 USB-C"
        pm = ProtectionMap()
        protected = protect_identifiers(text, pm)
        assert "https://example.com" not in protected
        assert "USB-C" not in protected
        restored = pm.restore(protected)
        assert restored == text

    def test_placeholders_exact_once(self):
        text = "型号 ABC-123 与 ABC-123 相同"
        pm = ProtectionMap()
        protected = protect_identifiers(text, pm)
        check = pm.validate_output(protected)
        assert check["ok"] is True

    def test_no_spans_returns_same(self):
        pm = ProtectionMap()
        protected = protect_identifiers("纯中文文本", pm)
        assert protected == "纯中文文本"
        assert pm.tokens == []
