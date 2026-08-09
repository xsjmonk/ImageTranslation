"""FastAPI application — thin HTTP layer over the shared translator.

- Long-running GPU inference runs in the threadpool (run_in_threadpool),
  never directly on the asyncio event loop.
- Error mapping:
    TranslationInputError        -> 400
    TranslationDeviceError /
    TranslationModelLoadError /
    translator unavailable       -> 503
    unexpected failure           -> 500 (safe envelope, logged server-side)
- Startup warmup (if enabled) runs in the threadpool via lifespan.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from image_translation.translation.exceptions import (
    TranslationDeviceError,
    TranslationError,
    TranslationInputError,
    TranslationModelLoadError,
)

from .api_models import ErrorResponse, HealthResponse, TranslateRequest, TranslateResponse
from .runtime import TranslationRuntime

logger = logging.getLogger(__name__)


def create_app(runtime: TranslationRuntime) -> FastAPI:
    """Build the FastAPI application wired to a TranslationRuntime."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: optionally warm up the model (blocking; run in threadpool)
        if runtime.config.runtime.warmup_on_start:
            logger.info("Starting translation server (warmup_on_start = true)...")
            try:
                await run_in_threadpool(runtime.warmup)
                info = runtime.translator.runtime_info
                logger.info("[INFO] Translation device: %s", info.device)
                logger.info("[INFO] Model ready: %s", info.model_name)
            except Exception:
                logger.exception("Model warmup failed; failing startup")
                raise
        else:
            logger.info(
                "Starting translation server (warmup_on_start = false; "
                "model loads lazily on first /translate)"
            )
        yield
        logger.info("Translation server shutdown complete.")

    app = FastAPI(
        title="Translation Server",
        description="Local GPU translation service (M2M100 zh→en)",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        info = runtime.translator.runtime_info
        return HealthResponse(
            status="ok" if info.ready else "starting",
            model=info.model_name,
            device=info.device,
            ready=info.ready,
        )

    @app.post("/translate", response_model=TranslateResponse)
    async def translate(req: TranslateRequest) -> TranslateResponse:
        # Acquire the translator (may fail: model load / CUDA unavailable -> 503)
        try:
            translator = runtime.translator
        except TranslationModelLoadError as e:
            logger.exception("Translator model unavailable")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Translation service unavailable",
            ) from e
        except TranslationDeviceError as e:
            logger.exception("Translator device unavailable")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Translation service unavailable",
            ) from e
        except Exception as e:
            logger.exception("Failed to create translator")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Translation service unavailable",
            ) from e

        # Blocking GPU inference must not block the event loop
        try:
            result = await run_in_threadpool(
                translator.translate_text, req.text
            )
        except TranslationInputError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        except (TranslationDeviceError, TranslationModelLoadError) as e:
            logger.exception("Translator unavailable during request")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Translation service unavailable",
            ) from e
        except Exception as e:
            logger.exception("Unexpected translation failure")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Translation failed",
            ) from e

        return TranslateResponse(translation=result.translated_text)

    # ------------------------------------------------------------------
    # Error envelope: JSON, no tracebacks/internals exposed
    # ------------------------------------------------------------------

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=exc.detail).model_dump(),
        )

    return app
