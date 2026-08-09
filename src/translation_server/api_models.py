"""API request/response models — thin FastAPI schemas for the HTTP boundary.

The request model only validates the basic type. Actual content validation
(empty/whitespace-only/length) is done by the shared translator using the
configured maximum, so there is a single source of truth.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    """POST /translate request body.

    Only type validation here; content validation happens in the shared
    translator (stripped emptiness, configured max length).
    """
    text: str = Field(..., description="Text to translate (zh → en)")


class TranslateResponse(BaseModel):
    """POST /translate response body."""
    translation: str = Field(..., description="Translated English text")


class HealthResponse(BaseModel):
    """GET /health response."""
    status: str = "ok"          # ok | starting
    model: str = ""
    device: str = ""
    ready: bool = False


class ErrorResponse(BaseModel):
    """Standard safe JSON error envelope (no internals/tracebacks)."""
    error: str
