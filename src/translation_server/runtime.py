"""Server runtime — builds the FastAPI app, wires the translator, starts Uvicorn."""

from __future__ import annotations

import logging
import hashlib
import json
from copy import deepcopy
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from image_translation.translation import (
    StructuredTranslationResult,
    StructuredTranslator,
    Translator,
    create_translator,
)
from image_translation.translation.config import (
    TranslationStyle,
    resolve_translation_style,
)

from .config import TranslationServerConfig, load_server_config

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DiagnosticBudget:
    max_entries: int = 8
    max_segments: int = 128
    max_items: int = 32
    max_keys: int = 32
    max_string_bytes: int = 256
    max_entry_bytes: int = 64 * 1024
    max_total_bytes: int = 8 * 64 * 1024


DIAGNOSTIC_BUDGET = DiagnosticBudget()
DIAGNOSTIC_MAX_ENTRIES = DIAGNOSTIC_BUDGET.max_entries
DIAGNOSTIC_MAX_TEXT = DIAGNOSTIC_BUDGET.max_string_bytes


class TranslationRuntime:
    """Holds the loaded translator and server config for the FastAPI app lifetime."""

    def __init__(self, config: TranslationServerConfig) -> None:
        self.config = config
        self._translator: Optional[Translator] = None
        self._structured_invocations = 0
        self._structured_invocations_lock = threading.Lock()
        # Bounded, summary-only diagnostics; full chapters never live here.
        self._structured_diagnostics: deque[dict] = deque(
            maxlen=DIAGNOSTIC_BUDGET.max_entries
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
        self, text: str, source_language: str, target_language: str,
        style: TranslationStyle | str | None = None,
    ):
        """Run plain translation through the runtime-owned service path."""
        with self._structured_invocations_lock:
            self._plain_invocations += 1
        resolved_style = resolve_translation_style(
            style, self.config.translation.default_style
        )
        return self.translator.translate_text(
            text, source_language, target_language, style=resolved_style
        )

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
        style: TranslationStyle | str | None = None,
    ) -> StructuredTranslationResult:
        """Translate HTML through the runtime-owned shared service path."""
        with self._structured_invocations_lock:
            self._structured_invocations += 1
        resolved_style = resolve_translation_style(
            style, self.config.translation.default_style
        )
        result = StructuredTranslator(
            self.translator,
            self.config.structured,
            self.config.translation,
            document_id=document_id,
        ).translate(
            text, source_language, target_language,
            resolved_style,
        )
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
                        for segment in result.segments[:DIAGNOSTIC_BUDGET.max_segments]
                    ],
                }
            )
            self._structured_diagnostics[-1] = self._fit_diagnostic(
                self._structured_diagnostics[-1]
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
        """Recursively sanitize a segment to JSON-safe bounded data."""
        return TranslationRuntime._sanitize_diagnostic(segment)

    @staticmethod
    def _sanitize_diagnostic(value, *, depth: int = 0):
        budget = DIAGNOSTIC_BUDGET
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            raw = value.encode("utf-8")
            if len(raw) <= budget.max_string_bytes:
                return value
            marker = "…".encode("utf-8")
            return (
                raw[: max(0, budget.max_string_bytes - len(marker))]
                .decode("utf-8", "ignore")
                + "…"
            )
        if depth > 12:
            return {"__truncated__": True, "reason": "max_depth"}
        if isinstance(value, dict):
            result = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= budget.max_keys:
                    result["__truncated_keys__"] = True
                    break
                result[str(key)] = TranslationRuntime._sanitize_diagnostic(
                    item, depth=depth + 1
                )
            return result
        if isinstance(value, (list, tuple)):
            result = [
                TranslationRuntime._sanitize_diagnostic(item, depth=depth + 1)
                for item in value[: budget.max_items]
            ]
            if len(value) > budget.max_items:
                result.append({"__truncated_items__": True})
            return result
        return {"__unsupported__": type(value).__name__[:64]}

    @staticmethod
    def _fit_diagnostic(value: dict) -> dict:
        """Enforce serialized byte budget, failing closed to a summary."""
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        if len(encoded) <= DIAGNOSTIC_BUDGET.max_entry_bytes:
            return value
        # Preserve a bounded plan sample for tests/debugging before falling
        # back to a summary-only record.
        if isinstance(value.get("segments"), list):
            candidate = dict(value)
            segments = value["segments"]
            for count in (64, 32, 16, 8, 4, 2, 1):
                candidate["segments"] = segments[:count]
                candidate["segments_truncated"] = len(segments) > count
                if len(
                    json.dumps(
                        candidate, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                ) <= DIAGNOSTIC_BUDGET.max_entry_bytes:
                    return candidate
        summary = {
            "invocation": value.get("invocation"),
            "duration_seconds": value.get("duration_seconds"),
            "segment_count": value.get("segment_count"),
            "retry_count": value.get("retry_count"),
            "fallback_count": value.get("fallback_count"),
            "output_fingerprint": value.get("output_fingerprint"),
            "__truncated__": True,
            "serialized_bytes": len(encoded),
        }
        return summary

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
