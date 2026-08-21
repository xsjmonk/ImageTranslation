from __future__ import annotations

from typing import List, Optional

import pytest

from image_translation.translation.base import Translator
from image_translation.translation.config import TranslationStyle
from image_translation.translation.models import TranslationResult


class ContractOnlyFakeTranslator(Translator):
    """Does not load any model; purely validates style-aware signatures."""

    @property
    def name(self) -> str:
        return "contract-only-fake"

    @property
    def runtime_info(self):
        return None

    def measure_source_tokens(self, text: str, source_lang: str = "zh") -> int:
        return max(1, len(text))

    def translate_text(
        self,
        text: str,
        source_lang: str = "zh",
        target_lang: str = "en",
        style: TranslationStyle | str | None = None,
    ) -> TranslationResult:
        return TranslationResult(
            source_text=text,
            translated_text=f"EN:{text}",
            source_language=source_lang,
            target_language=target_lang,
            model_name=self.name,
            device="cpu",
            style=getattr(style, "value", style) or None,
        )

    def translate_batch_texts(
        self,
        texts,
        source_lang: str = "zh",
        target_lang: str = "en",
        max_new_tokens: Optional[int] = None,
        style: TranslationStyle | str | None = None,
    ) -> List[TranslationResult]:
        assert max_new_tokens is None or isinstance(max_new_tokens, int)
        assert style in (None, TranslationStyle.SENTENCE, TranslationStyle.PHRASE)
        return [
            self.translate_text(
                t,
                source_lang=source_lang,
                target_lang=target_lang,
                style=style,
            )
            for t in texts
        ]


@pytest.mark.parametrize("style", [TranslationStyle.SENTENCE, TranslationStyle.PHRASE])
def test_translator_style_contract_accepts_style_and_batch_budget(style: TranslationStyle) -> None:
    t = ContractOnlyFakeTranslator()

    r1 = t.translate_text("你好", style=style)
    assert r1.translated_text.startswith("EN:")
    assert r1.source_text == "你好"

    rs = t.translate_batch_texts(
        ["你好", "世界"],
        max_new_tokens=64,
        style=style,
    )
    assert len(rs) == 2
    assert all(x.translated_text.startswith("EN:") for x in rs)

