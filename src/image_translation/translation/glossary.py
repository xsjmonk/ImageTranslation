"""UTF-8 TSV glossary loading and plain-text terminology protection."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Sequence

from .base import Translator
from .chapter_chunking import find_glossary_spans
from .config import GlossaryEntry
from .exceptions import TranslationConfigurationError, TranslationQualityError
from .html_protection import ProtectionMap, assert_prefix_safe
from .models import TranslationResult


DEFAULT_GLOSSARY_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "glossary.tsv"
)
_HEADER = ["source", "target", "exact"]


def load_glossary_file(
    path: str | Path,
    *,
    required: bool = True,
) -> tuple[GlossaryEntry, ...]:
    """Load and validate one UTF-8 TSV glossary, including BOM input."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        if required:
            raise TranslationConfigurationError(
                f"glossary file not found: {resolved}"
            )
        return ()

    entries: list[GlossaryEntry] = []
    rows: list[int] = []
    try:
        with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t", strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise TranslationConfigurationError(
                    f"glossary file {resolved}, row 1: missing header"
                ) from exc
            if header != _HEADER:
                raise TranslationConfigurationError(
                    f"glossary file {resolved}, row 1: expected columns "
                    f"{_HEADER}, got {header}"
                )

            for row in reader:
                row_number = reader.line_num
                if len(row) != 3:
                    raise TranslationConfigurationError(
                        f"glossary file {resolved}, row {row_number}: "
                        "expected exactly 3 tab-delimited columns"
                    )
                source, target, exact_raw = row
                if not source.strip() or not target.strip():
                    raise TranslationConfigurationError(
                        f"glossary file {resolved}, row {row_number}: "
                        "source and target must be non-empty"
                    )
                if exact_raw.lower() not in {"true", "false"}:
                    raise TranslationConfigurationError(
                        f"glossary file {resolved}, row {row_number}: "
                        "exact must be true or false"
                    )
                try:
                    entry = GlossaryEntry(
                        source=source,
                        target=target,
                        exact=exact_raw.lower() == "true",
                    )
                except ValueError as exc:
                    raise TranslationConfigurationError(
                        f"glossary file {resolved}, row {row_number}: {exc}"
                    ) from exc
                if any(existing.source == entry.source for existing in entries):
                    raise TranslationConfigurationError(
                        f"glossary file {resolved}, row {row_number}: "
                        f"duplicate source term {entry.source!r}"
                    )
                entries.append(entry)
                rows.append(row_number)
    except csv.Error as exc:
        raise TranslationConfigurationError(
            f"glossary file {resolved}, row {getattr(reader, 'line_num', '?')}: "
            f"malformed TSV: {exc}"
        ) from exc
    except OSError as exc:
        raise TranslationConfigurationError(
            f"cannot read glossary file {resolved}: {exc}"
        ) from exc

    for left_index, left in enumerate(entries):
        for right_index in range(left_index + 1, len(entries)):
            right = entries[right_index]
            if left.source in right.source or right.source in left.source:
                raise TranslationConfigurationError(
                    f"glossary file {resolved}, rows {rows[left_index]} and "
                    f"{rows[right_index]}: overlapping source terms "
                    f"{left.source!r} and {right.source!r}"
                )
    return tuple(entries)


def _protect_text(
    text: str,
    entries: Sequence[GlossaryEntry],
) -> tuple[str, ProtectionMap, list[str]]:
    pmap = ProtectionMap()
    assert_prefix_safe(text, pmap.prefix)
    spans = find_glossary_spans(text, entries)
    replacements: list[tuple[int, int, str]] = []
    for start, end, entry in spans:
        replacements.append((start, end, pmap.reserve(entry.target, "glossary")))
    protected = text
    for start, end, token in reversed(replacements):
        protected = protected[:start] + token + protected[end:]
    return protected, pmap, [token for _, _, token in replacements]


class GlossaryTranslator(Translator):
    """Plain-text adapter that shares model inference with an inner translator."""

    def __init__(
        self,
        inner: Translator,
        entries: Iterable[GlossaryEntry],
    ) -> None:
        self._inner = inner
        self._entries = tuple(entries)

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def runtime_info(self):
        return self._inner.runtime_info

    def warmup(self) -> None:
        self._inner.warmup()

    def measure_source_tokens(self, text: str, source_lang: str = "zh") -> int:
        protected, _, _ = _protect_text(text, self._entries)
        return self._inner.measure_source_tokens(protected, source_lang)

    def translate_text(
        self,
        text: str,
        source_lang: str = "zh",
        target_lang: str = "en",
    ) -> TranslationResult:
        protected, pmap, order = _protect_text(text, self._entries)
        result = self._inner.translate_text(protected, source_lang, target_lang)
        if isinstance(result.translated_text, str):
            check = pmap.validate_output(
                result.translated_text, expected_order=order
            )
        else:
            check = {"ok": False}
        if check["ok"]:
            translated = pmap.restore(result.translated_text)
        else:
            translated, result = self._split_fallback(
                text, source_lang, target_lang
            )
        return replace(
            result,
            source_text=text,
            translated_text=translated,
            compact_text=translated,
            literal_text=translated,
        )

    def translate_batch_texts(
        self,
        texts: Sequence[str],
        source_lang: str = "zh",
        target_lang: str = "en",
        max_new_tokens: int | None = None,
    ) -> List[TranslationResult]:
        protected_data = [
            _protect_text(text, self._entries) for text in texts
        ]
        results = self._inner.translate_batch_texts(
            [item[0] for item in protected_data],
            source_lang=source_lang,
            target_lang=target_lang,
            max_new_tokens=max_new_tokens,
        )
        if len(results) != len(texts):
            raise TranslationQualityError(
                f"glossary adapter received {len(results)} results for "
                f"{len(texts)} inputs"
            )
        restored: list[TranslationResult] = []
        for index, result in enumerate(results):
            original, _, order = protected_data[index]
            pmap = protected_data[index][1]
            if isinstance(result.translated_text, str):
                check = pmap.validate_output(
                    result.translated_text, expected_order=order
                )
            else:
                check = {"ok": False}
            if check["ok"]:
                translated = pmap.restore(result.translated_text)
            else:
                translated, result = self._split_fallback(
                    texts[index], source_lang, target_lang
                )
            restored.append(
                replace(
                    result,
                    source_text=texts[index],
                    translated_text=translated,
                    compact_text=translated,
                    literal_text=translated,
                )
            )
        return restored

    def _split_fallback(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> tuple[str, TranslationResult]:
        """Translate non-glossary runs independently and insert fixed terms.

        This fallback is used only when the model cannot round-trip
        placeholders. Glossary terms never enter inference in this path, so a
        weak model cannot mutate or drop the configured terminology.
        """
        spans = find_glossary_spans(text, self._entries)
        if not spans:
            raise TranslationQualityError(
                "translation failed glossary placeholder validation"
            )
        pieces: list[str] = []
        cursor = 0
        first_result: TranslationResult | None = None
        for start, end, entry in spans:
            if start > cursor:
                result = self._inner.translate_text(
                    text[cursor:start], source_lang, target_lang
                )
                if not isinstance(result.translated_text, str):
                    raise TranslationQualityError(
                        "glossary fallback returned non-text output"
                    )
                first_result = first_result or result
                pieces.append(result.translated_text)
            pieces.append(entry.target)
            cursor = end
        if cursor < len(text):
            result = self._inner.translate_text(
                text[cursor:], source_lang, target_lang
            )
            if not isinstance(result.translated_text, str):
                raise TranslationQualityError(
                    "glossary fallback returned non-text output"
                )
            first_result = first_result or result
            pieces.append(result.translated_text)
        if first_result is None:
            info = self._inner.runtime_info
            first_result = TranslationResult(
                source_text=text,
                model_name=self._inner.name,
                device=getattr(info, "device", ""),
                source_language=source_lang,
                target_language=target_lang,
            )
        translated = "".join(pieces).strip()
        if not translated:
            raise TranslationQualityError(
                "glossary fallback produced empty translation"
            )
        return translated, first_result
