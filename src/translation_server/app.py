"""FastAPI application — thin HTTP layer over the shared translator.

- Long-running GPU inference runs in the threadpool (run_in_threadpool),
  never directly on the asyncio event loop.
- Bounded concurrency: an asyncio.Semaphore limits concurrent translations
  (GPU has one model; default concurrency = 1).
- format="plain" (default) keeps the existing text path; format="html" uses
  the structured translation layer. No HTML auto-detection.
- Error mapping:
    TranslationInputError        -> 400
    TranslationDeviceError /
    TranslationModelLoadError /
    translator unavailable       -> 503
    StructuredTranslationError   -> 422 (input) / 500 (processing)
    unexpected failure           -> 500 (safe envelope + correlation ID)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from image_translation.translation.exceptions import (
    StructuredTranslationError,
    TranslationDeviceError,
    TranslationError,
    TranslationInputError,
    TranslationModelLoadError,
    TranslationQualityError,
)
from image_translation.translation.structured_translation import StructuredTranslator

from .api_models import (
    CacheResponse,
    ErrorResponse,
    HealthResponse,
    TranslateRequest,
    TranslateResponse,
)
from .runtime import TranslationRuntime

logger = logging.getLogger(__name__)


def create_app(runtime: TranslationRuntime) -> FastAPI:
    """Build the FastAPI application wired to a TranslationRuntime."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
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
        description="Local GPU translation service (M2M100 zh→en, plain + HTML)",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Bounded concurrency for GPU translation (default 1)
    concurrency = max(1, runtime.config.structured.concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        info = runtime.translator.runtime_info
        return HealthResponse(
            status="ok" if info.ready else "starting",
            model=info.model_name,
            model_revision=info.model_revision,
            device=info.device,
            precision=info.precision,
            ready=info.ready,
            cache_dir=info.cache_dir,
            snapshot_path=info.snapshot_path,
            cache_status=info.cache_status,
            local_files_only=info.local_files_only,
            offline=info.offline,
        )

    @app.get("/cache", response_model=CacheResponse)
    async def cache() -> CacheResponse:
        """Model-cache state.

        Thin HTTP facade: reuses the translator's runtime_info and, when
        the model is not loaded yet, the optional check_cache() capability
        (a live, model-free resolution). No domain logic lives here.
        """
        translator = runtime.translator
        info = translator.runtime_info
        if not info.ready:
            check = getattr(translator, "check_cache", None)
            if check is not None:
                from image_translation.translation.exceptions import (
                    TranslationModelLoadError,
                )
                try:
                    resolved = check()
                except TranslationModelLoadError:
                    logger.warning("GET /cache: cache resolution failed "
                                   "(model not loaded)")
                    return CacheResponse(
                        model=info.model_name,
                        revision=info.model_revision,
                        cache_status="error",
                        local_files_only=info.local_files_only,
                        offline=info.offline,
                        ready=False,
                    )
                return CacheResponse(
                    model=resolved.model_name,
                    revision=resolved.revision,
                    cache_dir=resolved.cache_dir,
                    snapshot_path=resolved.snapshot_path,
                    cache_status=resolved.cache_status,
                    local_files_only=info.local_files_only,
                    offline=resolved.offline,
                    ready=False,
                )
        return CacheResponse(
            model=info.model_name,
            revision=info.model_revision,
            cache_dir=info.cache_dir,
            snapshot_path=info.snapshot_path,
            cache_status=info.cache_status,
            local_files_only=info.local_files_only,
            offline=info.offline,
            ready=info.ready,
        )

    @app.post("/translate", response_model=TranslateResponse)
    async def translate(req: TranslateRequest) -> TranslateResponse:
        correlation_id = uuid.uuid4().hex[:12]

        # The semaphore bounds ALL GPU work, including lazy model loading:
        # the translator is acquired while holding it.
        async with semaphore:
            try:
                translator = runtime.translator
            except Exception as e:
                logger.exception("[%s] translator unavailable", correlation_id)
                raise _http(503, "Translation service unavailable", correlation_id) from e

            source_lang = req.source_language or runtime.config.translation.source_language
            target_lang = req.target_language or runtime.config.translation.target_language

            if req.format == "html":
                return await _translate_html(
                    runtime, translator, req, source_lang, target_lang, correlation_id
                )
            return await _translate_plain(
                runtime.quality_translator,
                req,
                source_lang,
                target_lang,
                correlation_id,
            )

    # ------------------------------------------------------------------
    # Error envelope: JSON, no tracebacks/internals exposed
    # ------------------------------------------------------------------

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail if isinstance(exc.detail, dict)
            else ErrorResponse(error=str(exc.detail)).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="Invalid translation request",
                correlation_id=uuid.uuid4().hex[:12],
                code="invalid_request",
            ).model_dump(),
        )

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _http(
    code: int,
    message: str,
    correlation_id: str = "",
    error_code: str = "translation_error",
) -> HTTPException:
    return HTTPException(
        status_code=code,
        detail=ErrorResponse(
            error=message,
            correlation_id=correlation_id,
            code=error_code,
        ).model_dump(),
    )


async def _translate_plain(
    translator, req: TranslateRequest, source_lang: str, target_lang: str, correlation_id: str
) -> TranslateResponse:
    try:
        result = await run_in_threadpool(
            translator.translate_text, req.text, source_lang, target_lang
        )
    except TranslationInputError as e:
        raise _http(400, str(e), correlation_id, "invalid_input") from e
    except TranslationQualityError as e:
        raise _http(422, str(e), correlation_id, "translation_quality") from e
    except (TranslationDeviceError, TranslationModelLoadError) as e:
        logger.exception("[%s] translator unavailable during request", correlation_id)
        raise _http(503, "Translation service unavailable", correlation_id,
                    "model_unavailable") from e
    except Exception as e:
        logger.exception("[%s] unexpected plain translation failure", correlation_id)
        raise _http(500, "Translation failed", correlation_id,
                    "translation_failed") from e
    return TranslateResponse(translation=result.translated_text)


async def _translate_html(
    runtime, translator, req: TranslateRequest, source_lang: str, target_lang: str,
    correlation_id: str,
) -> TranslateResponse:
    cfg = replace(runtime.config.structured, glossary=runtime.glossary)
    if not cfg.enabled:
        raise _http(
            400,
            "HTML translation is disabled on this server",
            correlation_id,
            "invalid_format",
        )

    st = StructuredTranslator(
        translator,
        cfg,
        runtime.config.translation,
        document_id=correlation_id,
    )
    try:
        result = await run_in_threadpool(
            st.translate, req.text, source_lang, target_lang
        )
    except StructuredTranslationError as e:
        message = str(e)
        code = 422 if ("max_chapter_characters" in message or "must be a string" in message) else 500
        logger.warning("[%s] structured translation failed: %s", correlation_id, message)
        raise _http(code, message, correlation_id, "structured_translation") from e
    except (TranslationInputError, ValueError) as e:
        raise _http(400, str(e), correlation_id, "invalid_input") from e
    except TranslationQualityError as e:
        raise _http(422, str(e), correlation_id, "translation_quality") from e
    except (TranslationDeviceError, TranslationModelLoadError) as e:
        logger.exception("[%s] translator unavailable during structured request", correlation_id)
        raise _http(503, "Translation service unavailable", correlation_id,
                    "model_unavailable") from e
    except Exception as e:
        logger.exception("[%s] unexpected structured translation failure", correlation_id)
        raise _http(500, "Translation failed", correlation_id,
                    "translation_failed") from e

    logger.info(
        "[%s] html ok segments=%d source_tokens=%d retries=%d fallbacks=%d %.2fs",
        correlation_id,
        result.segment_count,
        result.total_source_tokens,
        result.retry_count,
        result.fallback_count,
        result.duration_seconds,
    )
    return TranslateResponse(translation=result.translated_html)
