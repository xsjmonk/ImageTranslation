from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from image_translation.translation import TranslationStyle
from image_translation.translation.models import TranslationResult, TranslationRuntimeInfo
from image_translation.translation.base import Translator
from translation_server.app import create_app
from translation_server.config import TranslationServerConfig
from translation_server.runtime import TranslationRuntime


class RecordingStyleTranslator(Translator):
    def __init__(self) -> None:
        self.single_styles = []
        self.batch_styles = []

    @property
    def name(self) -> str:
        return "recording-style"

    @property
    def runtime_info(self) -> TranslationRuntimeInfo:
        return TranslationRuntimeInfo(model_name=self.name, device="cpu", ready=True)

    def measure_source_tokens(self, text: str, source_lang: str = "zh") -> int:
        return max(1, len(text))

    def translate_text(
        self, text: str, source_lang="zh", target_lang="en", style=None
    ) -> TranslationResult:
        self.single_styles.append(style)
        return TranslationResult(
            source_text=text,
            translated_text=f"EN:{text}",
            source_language=source_lang,
            target_language=target_lang,
            style=getattr(style, "value", style) or "sentence",
        )

    def translate_batch_texts(
        self,
        texts,
        source_lang="zh",
        target_lang="en",
        max_new_tokens=None,
        style=None,
    ):
        self.batch_styles.extend([style] * len(texts))
        return [
            self.translate_text(text, source_lang, target_lang, style)
            for text in texts
        ]


def _client(default_style: TranslationStyle):
    config = TranslationServerConfig()
    config.runtime.warmup_on_start = False
    config.translation.default_style = default_style
    fake = RecordingStyleTranslator()
    runtime = TranslationRuntime(config)
    runtime._translator = fake
    return TestClient(create_app(runtime)), fake, config


@pytest.mark.parametrize(
    ("default_style", "requested_style", "expected"),
    [
        (TranslationStyle.SENTENCE, None, TranslationStyle.SENTENCE),
        (TranslationStyle.SENTENCE, "phrase", TranslationStyle.PHRASE),
        (TranslationStyle.PHRASE, None, TranslationStyle.PHRASE),
        (TranslationStyle.PHRASE, "sentence", TranslationStyle.SENTENCE),
    ],
)
@pytest.mark.parametrize("format_name", ["plain", "html"])
def test_api_style_matrix_records_normalized_style(
    default_style, requested_style, expected, format_name
):
    client, fake, config = _client(default_style)
    payload = {"text": "你好" if format_name == "plain" else "<p>你好</p>", "format": format_name}
    if requested_style is not None:
        payload["style"] = requested_style

    response = client.post("/translate", json=payload)

    assert response.status_code == 200, response.text
    assert set(response.json()) == {"translation"}
    recorded = fake.single_styles + fake.batch_styles
    assert recorded
    assert all(style == expected for style in recorded)
    assert all(style is not None for style in recorded)
    assert config.translation.default_style == default_style


@pytest.mark.parametrize("value", ["", "invalid"])
def test_api_rejects_invalid_style(value):
    client, _, _ = _client(TranslationStyle.SENTENCE)
    response = client.post("/translate", json={"text": "你好", "style": value})
    assert response.status_code in {400, 422}
    assert "error" in response.json()
