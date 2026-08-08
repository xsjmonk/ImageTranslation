"""FastAPI application — thin HTTP layer over the shared translator."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, status

from .api_models import ErrorResponse, HealthResponse, TranslateRequest, TranslateResponse
from .runtime import TranslationRuntime

logger = logging.getLogger(__name__)


def create_app(runtime: TranslationRuntime) -> FastAPI:
    """Build the FastAPI application wired to a TranslationRuntime.

    Args:
        runtime: Pre-configured runtime with translator.

    Returns:
        A FastAPI application instance.
    """
    app = FastAPI(
        title="Translation Server",
        description="Local GPU translation service (M2M100 zh→en)",
        version="0.1.0",
    )

    # Warm up the translator on startup (lifespan)
    @app.on_event("startup")
    async def _startup() -> None:
        _warmup()

    def _warmup() -> None:
        logger.info("Starting translation server...")
        try:
            runtime.warmup()
            info = runtime.translator.runtime_info
            logger.info("[INFO] Translation device: %s", info.device)
            logger.info("[INFO] Model ready: %s", info.model_name)
        except Exception as e:
            logger.error("Failed to load translation model: %s", e)
            raise

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        info = runtime.translator.runtime_info
        return HealthResponse(
            status="ok",
            model=info.model_name,
            device=info.device,
            ready=info.ready,
        )

    @app.post("/translate", response_model=TranslateResponse)
    async def translate(req: TranslateRequest) -> TranslateResponse:
        try:
            translator = runtime.translator
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Translator unavailable: {e}",
            )

        try:
            result = translator.translate_text(req.text)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except Exception as e:
            logger.exception("Translation failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Translation failed: {e}",
            )

        return TranslateResponse(translation=result.translated_text)

    # Error handlers
    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request, exc: HTTPException):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=exc.detail, detail="").model_dump(),
        )

    return app
