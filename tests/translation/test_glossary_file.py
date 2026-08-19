"""Dedicated TSV glossary and plain-text protection tests."""

from __future__ import annotations

import pytest

from image_translation.translation.exceptions import TranslationConfigurationError
from image_translation.translation.glossary import (
    GlossaryTranslator,
    load_glossary_file,
)
from tests.translation.test_one_round_revision import FakeTranslator


def _write(path, text: str, encoding: str = "utf-8"):
    path.write_text(text, encoding=encoding, newline="")
    return path


class TestGlossaryFile:
    def test_loads_utf8_bom_and_preserves_unicode(self, tmp_path):
        path = _write(
            tmp_path / "glossary.tsv",
            "source\ttarget\texact\n蔡司\tZeiss\ttrue\n",
            encoding="utf-8-sig",
        )
        entries = load_glossary_file(path)
        assert entries[0].source == "蔡司"
        assert entries[0].target == "Zeiss"
        assert entries[0].exact is True

    @pytest.mark.parametrize(
        "content, message",
        [
            ("source\ttarget\n蔡司\tZeiss\n", "expected columns"),
            ("source\ttarget\texact\n蔡司\tZeiss\tmaybe\n", "exact"),
            ("source\ttarget\texact\n蔡司\t\ttrue\n", "non-empty"),
            (
                "source\ttarget\texact\n蔡司\tZeiss\ttrue\n蔡司\tZ\ttrue\n",
                "duplicate",
            ),
            (
                "source\ttarget\texact\n充电\tCharge\ttrue\n充电器\tCharger\ttrue\n",
                "overlapping",
            ),
        ],
    )
    def test_invalid_rows_report_path_and_row(self, tmp_path, content, message):
        path = _write(tmp_path / "bad.tsv", content)
        with pytest.raises(TranslationConfigurationError) as exc:
            load_glossary_file(path)
        assert str(path.resolve()) in str(exc.value)
        assert message in str(exc.value)

    def test_required_and_optional_missing_file(self, tmp_path):
        missing = tmp_path / "missing.tsv"
        with pytest.raises(TranslationConfigurationError, match="not found"):
            load_glossary_file(missing, required=True)
        assert load_glossary_file(missing, required=False) == ()


class TestPlainGlossaryTranslator:
    def test_protects_and_restores_repeated_terms(self, tmp_path):
        path = _write(
            tmp_path / "glossary.tsv",
            "source\ttarget\texact\n蔡司\tZeiss\ttrue\n",
        )
        entries = load_glossary_file(path)
        translator = GlossaryTranslator(FakeTranslator(), entries)
        result = translator.translate_text("蔡司眼镜和蔡司镜框")
        assert result.source_text == "蔡司眼镜和蔡司镜框"
        assert result.translated_text.count("Zeiss") == 2
        assert "__IT" not in result.translated_text

    def test_falls_back_when_model_drops_term_placeholder(self, tmp_path):
        path = _write(
            tmp_path / "glossary.tsv",
            "source\ttarget\texact\n蔡司\tZeiss\ttrue\n",
        )
        entries = load_glossary_file(path)
        translator = FakeTranslator(drop_all={1})
        result = GlossaryTranslator(translator, entries).translate_text("蔡司")
        assert result.translated_text == "Zeiss"
