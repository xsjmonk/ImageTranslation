"""HTTP client translator — calls the standalone translation server API."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import List, Sequence

from .base import Translator
from .config import TranslationStyle
from .models import TranslationResult, TranslationRuntimeInfo

logger = logging.getLogger(__name__)


class ServerTranslator(Translator):
    """Translate text via POST /translate on a running translation server."""

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8091",
        source_language: str = "zh",
        target_language: str = "en",
        style: TranslationStyle | str = TranslationStyle.PHRASE,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._base_url = server_url.rstrip("/")
        self._source_language = source_language
        self._target_language = target_language
        self._style = TranslationStyle(style) if isinstance(style, str) else style
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "server"

    @property
    def runtime_info(self) -> TranslationRuntimeInfo:
        return TranslationRuntimeInfo(
            backend="server",
            model_name=self._base_url,
            device="remote",
            precision="unknown",
            dtype="unknown",
            source_language=self._source_language,
            target_language=self._target_language,
        )

    def translate_text(
        self,
        text: str,
        source_lang: str = "zh",
        target_lang: str = "en",
        style: TranslationStyle | str | None = None,
    ) -> TranslationResult:
        results = self.translate_batch_texts(
            [text],
            source_lang=source_lang,
            target_lang=target_lang,
            style=style,
        )
        return results[0]

    def translate_batch_texts(
        self,
        texts: Sequence[str],
        source_lang: str = "zh",
        target_lang: str = "en",
        max_new_tokens: int | None = None,
        style: TranslationStyle | str | None = None,
    ) -> List[TranslationResult]:
        resolved_style = style or self._style
        results: List[TranslationResult] = []
        for text in texts:
            payload = {
                "text": text,
                "format": "plain",
                "source_language": source_lang,
                "target_language": target_lang,
                "style": (
                    resolved_style.value
                    if isinstance(resolved_style, TranslationStyle)
                    else str(resolved_style)
                ),
            }
            translated = self._post_translate(payload)
            results.append(
                TranslationResult(
                    source_text=text,
                    translated_text=translated,
                    compact_text=translated,
                    literal_text=translated,
                    source_language=source_lang,
                    target_language=target_lang,
                )
            )
        return results

    def _post_translate(self, payload: dict) -> str:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/translate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout_seconds
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Translation server HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Translation server unreachable at {self._base_url}: {exc.reason}"
            ) from exc

        translation = data.get("translation")
        if not isinstance(translation, str) or not translation.strip():
            raise RuntimeError("Translation server returned empty translation")
        return translation.strip()
