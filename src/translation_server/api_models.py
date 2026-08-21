"""API request/response models — thin FastAPI schemas for the HTTP boundary.

Backward compatibility (documented):
- `format` is optional and defaults to "plain"; existing callers that send
  only {"text": ...} are unaffected.
- format="plain" uses the current plain-text path.
- format="html" uses the new structured (HTML-aware) path.
- No auto-detection of HTML from "<"/">" characters.
- Content validation (emptiness, length) is performed by the shared
  translation layer using configured limits.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    """POST /translate request body."""
    text: str = Field(..., description="Text or HTML chapter to translate")
    format: Literal["plain", "html"] = Field(
        "plain", description="plain (default, backward compatible) or html"
    )
    source_language: Optional[str] = Field(
        None, description="Optional source language code; defaults to server configuration"
    )
    target_language: Optional[str] = Field(
        None, description="Optional target language code; defaults to server configuration"
    )
    style: Optional[Literal["sentence", "phrase"]] = Field(
        None, description="Translation style; defaults to server configuration"
    )


class TranslateResponse(BaseModel):
    """POST /translate response body."""
    translation: str = Field(..., description="Translated text or HTML")


class HealthResponse(BaseModel):
    """GET /health response (backward compatible; extra fields allowed)."""
    status: str = "ok"          # ok | starting
    model: str = ""
    model_family: str = ""
    model_revision: str = ""
    source_language: str = ""
    target_language: str = ""
    device: str = ""
    precision: str = ""
    dtype: str = ""
    ready: bool = False
    cache_dir: str = ""
    snapshot_path: str = ""
    cache_status: str = ""      # cache_hit | download | none
    local_files_only: bool = False
    offline: bool = False        # effective offline (local_files_only OR
                                 # downloads disabled)


class CacheResponse(BaseModel):
    """GET /cache response — model-cache state (thin HTTP facade over the
    shared translator's cache capabilities; no domain logic)."""
    model: str = ""
    revision: str = ""
    cache_dir: str = ""
    snapshot_path: str = ""
    cache_status: str = ""       # cache_hit | download | error | none
    local_files_only: bool = False
    offline: bool = False
    ready: bool = False


class ErrorResponse(BaseModel):
    """Standard safe JSON error envelope (no internals/tracebacks)."""
    error: str
    correlation_id: str = ""
    code: str = "translation_error"
