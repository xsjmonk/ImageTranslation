"""API request/response models — thin FastAPI schemas for the HTTP boundary."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    """POST /translate request body."""
    text: str = Field(..., min_length=1, max_length=4000, description="Text to translate (zh → en)")


class TranslateResponse(BaseModel):
    """POST /translate response body."""
    translation: str = Field(..., description="Translated English text")


class HealthResponse(BaseModel):
    """GET /health response."""
    status: str = "ok"
    model: str = ""
    device: str = ""
    ready: bool = False


class ErrorResponse(BaseModel):
    """Standard error envelope."""
    error: str
    detail: str = ""
