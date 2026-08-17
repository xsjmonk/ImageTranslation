"""Server runtime — builds the FastAPI app, wires the translator, starts Uvicorn."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from image_translation.translation import Translator, create_translator

from .config import TranslationServerConfig, load_server_config

logger = logging.getLogger(__name__)


class TranslationRuntime:
    """Holds the loaded translator and server config for the FastAPI app lifetime."""

    def __init__(self, config: TranslationServerConfig) -> None:
        self.config = config
        self._translator: Optional[Translator] = None

    @property
    def translator(self) -> Translator:
        if self._translator is None:
            self._translator = create_translator(self.config.translation)
        return self._translator

    def warmup(self) -> None:
        """Trigger model loading."""
        self.translator.warmup()  # type: ignore[union-attr]

    def cache_diagnostics(self) -> dict:
        """Config-derived cache diagnostics for startup logging.

        Read-only (no snapshot resolution, no model load, no network):
        reports the configured cache root, effective offline flag, and
        model revision exactly as the translator will honor them.
        """
        t = self.config.translation
        return {
            "model": t.model_name,
            "revision": t.model_revision,
            "cache_dir": t.model_cache_dir or "",
            "offline": t.local_files_only or not t.allow_model_download,
        }


def build_runtime(config_path: Optional[Path] = None) -> TranslationRuntime:
    """Load config and create a TranslationRuntime.

    This is the entry point for both the server and smoke-test scripts.
    """
    config = load_server_config(config_path)
    return TranslationRuntime(config)
