"""Server runtime — builds the FastAPI app, wires the translator, starts Uvicorn."""

from __future__ import annotations

import logging
import hashlib
from copy import deepcopy
import threading
from collections import deque
from pathlib import Path
from typing import Optional

from image_translation.translation import (
    StructuredTranslationResult,
    StructuredTranslator,
    Translator,
    create_translator,
)

from .config import TranslationServerConfig, load_server_config

logger = logging.getLogger(__name__)

DIAGNOSTIC_MAX_ENTRIES = 8
DIAGNOSTIC_MAX_SEGMENTS = 128
DIAGNOSTIC_MAX_TEXT = 256
DIAGNOSTIC_MAX_ITEMS = 32


class TranslationRuntime:
    """Holds the loaded translator and server config for the FastAPI app lifetime."""

    def __init__(self, config: TranslationServerConfig) -> None:
        self.config = config
        self._translator: Optional[Translator] = None
        self._structured_invocations = 0
        self._structured_invocations_lock = threading.Lock()
        # Bounded, summary-only diagnostics; full chapters never live here.
        self._structured_diagnostics: deque[dict] = deque(
            maxlen=DIAGNOSTIC_MAX_ENTRIES
        )
        self._plain_invocations = 0

    @property
    def translator(self) -> Translator:
        if self._translator is None:
            self._translator = create_translator(self.config.translation)
        return self._translator

    def warmup(self) -> None:
        """Trigger model loading."""
        self.translator.warmup()  # type: ignore[union-attr]

    def translate_plain(
        self, text: str, source_language: str, target_language: str
    ):
        """Run plain translation through the runtime-owned service path."""
        with self._structured_invocations_lock:
            self._plain_invocations += 1
        return self.translator.translate_text(text, source_language, target_language)

    @property
    def plain_invocation_count(self) -> int:
        with self._structured_invocations_lock:
            return self._plain_invocations

    def translate_structured(
        self,
        text: str,
        source_language: str,
        target_language: str,
        document_id: str,
    ) -> StructuredTranslationResult:
        """Translate HTML through the runtime-owned shared service path."""
        with self._structured_invocations_lock:
            self._structured_invocations += 1
        result = StructuredTranslator(
            self.translator,
            self.config.structured,
            self.config.translation,
            document_id=document_id,
        ).translate(text, source_language, target_language)
        with self._structured_invocations_lock:
            self._structured_diagnostics.append(
                {
                    "invocation": self._structured_invocations,
                    "document_id": document_id,
                    "duration_seconds": result.duration_seconds,
                    "segment_count": result.segment_count,
                    "retry_count": result.retry_count,
                    "fallback_count": result.fallback_count,
                    "output_fingerprint": hashlib.sha256(
                        result.translated_html.encode("utf-8")
                    ).hexdigest(),
                    "segments": [
                        self._bound_segment(segment)
                        for segment in result.segments[:DIAGNOSTIC_MAX_SEGMENTS]
                    ],
                }
            )
        return result

    @property
    def structured_invocation_count(self) -> int:
        with self._structured_invocations_lock:
            return self._structured_invocations

    @property
    def structured_diagnostics(self) -> list[dict]:
        with self._structured_invocations_lock:
            return deepcopy(list(self._structured_diagnostics))

    @staticmethod
    def _bound_segment(segment: dict) -> dict:
        """Keep only bounded plan/output evidence for diagnostics."""
        bounded = {}
        for key, value in segment.items():
            if isinstance(value, str):
                if key in {"source_text", "block_text", "translated_text"}:
                    bounded[f"{key}_fingerprint"] = hashlib.sha256(
                        value.encode("utf-8")
                    ).hexdigest()
                bounded[key] = value[:DIAGNOSTIC_MAX_TEXT]
            elif isinstance(value, list):
                bounded[key] = [
                    item if not isinstance(item, str) else item[:DIAGNOSTIC_MAX_TEXT]
                    for item in value[:DIAGNOSTIC_MAX_ITEMS]
                ]
            elif isinstance(value, dict):
                bounded[key] = {
                    str(k): (v[:DIAGNOSTIC_MAX_TEXT] if isinstance(v, str) else v)
                    for k, v in list(value.items())[:DIAGNOSTIC_MAX_ITEMS]
                }
            else:
                bounded[key] = value
        return bounded

    def cache_diagnostics(self) -> dict:
        """Config-derived cache diagnostics for startup logging.

        Read-only (no snapshot resolution, no model load, no network):
        reports the configured cache root, effective offline flag, and
        model revision exactly as the translator will honor them.
        """
        t = self.config.translation
        return {
            "model": t.model_name,
            "model_family": t.model_family,
            "revision": t.model_revision,
            "cache_dir": t.model_cache_dir or "",
            "offline": t.local_files_only or not t.allow_model_download,
            "source_language": t.source_language,
            "target_language": t.target_language,
            "precision": t.precision,
            "device": t.effective_device(),
        }


def build_runtime(config_path: Optional[Path] = None) -> TranslationRuntime:
    """Load config and create a TranslationRuntime.

    This is the entry point for both the server and smoke-test scripts.
    """
    config = load_server_config(config_path)
    return TranslationRuntime(config)
