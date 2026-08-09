"""M2M100 418M GPU translator — lazy-loaded, CUDA-required by default."""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Sequence

from .base import Translator
from .config import TranslationConfig
from .exceptions import (
    TranslationConfigurationError,
    TranslationDeviceError,
    TranslationModelLoadError,
)
from .models import TranslationResult, TranslationRuntimeInfo
from .text_utils import preprocess

logger = logging.getLogger(__name__)


class M2M100Translator(Translator):
    """M2M100 418M neural machine translation engine.

    Uses facebook/m2m100_418M via Hugging Face Transformers on NVIDIA GPU.
    Model is lazy-loaded on first translate call.
    Thread-safe: one operation (set language → tokenize → generate → decode)
    is atomic with respect to tokenizer/model state.
    """

    def __init__(self, config: TranslationConfig) -> None:
        self._config = config
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None
        self._device_str: str = ""
        self._precision_str: str = ""
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"m2m100@{self._config.model_name}"

    @property
    def runtime_info(self) -> TranslationRuntimeInfo:
        gpu_name = ""
        cuda_ok = False
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
            if cuda_ok and self._config.cuda_device < torch.cuda.device_count():
                gpu_name = torch.cuda.get_device_name(self._config.cuda_device)
        except ImportError:
            pass

        return TranslationRuntimeInfo(
            model_name=self._config.model_name,
            device=self._device_str,
            precision=self._precision_str,
            cuda_available=cuda_ok,
            gpu_name=gpu_name,
            ready=self._model is not None,
        )

    def translate_text(
        self, text: str, source_lang: str = "zh", target_lang: str = "en"
    ) -> TranslationResult:
        """Translate a single string."""
        self._ensure_loaded()
        cleaned = preprocess(text, max_characters=self._config.max_input_characters)
        return self._translate_impl([cleaned], source_lang, target_lang)[0]

    def translate_batch_texts(
        self, texts: Sequence[str], source_lang: str = "zh", target_lang: str = "en"
    ) -> List[TranslationResult]:
        """Translate multiple strings with real GPU batching.

        Each chunk is tokenized together, one generate() call per chunk,
        then batch-decoded. Input ordering is preserved.
        """
        if not texts:
            return []
        self._ensure_loaded()

        cleaned = [
            preprocess(t, max_characters=self._config.max_input_characters)
            for t in texts
        ]

        results: List[TranslationResult] = []
        batch_size = self._config.batch_size
        for i in range(0, len(cleaned), batch_size):
            chunk = cleaned[i : i + batch_size]
            results.extend(self._translate_impl(chunk, source_lang, target_lang))
        return results

    def warmup(self) -> None:
        """Explicitly trigger model loading. Idempotent."""
        self._ensure_loaded()

    # ------------------------------------------------------------------
    # Internal: lazy loading + GPU setup
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return  # Double-check
            self._load_model()

    def _resolve_device(self) -> str:
        """Resolve the effective device string from config + runtime state.

        Returns:
            "cuda:N" when CUDA is used, "cpu" when CPU is used.

        Raises:
            TranslationDeviceError: If CUDA is required but unavailable,
                or the configured device index is out of range.
        """
        import torch

        requested = self._config.device
        cuda_ok = torch.cuda.is_available()

        if requested == "cpu":
            return "cpu"

        # requested == "cuda"
        if not cuda_ok:
            if self._config.allow_cpu_fallback:
                logger.warning(
                    "[WARN] CUDA not available, falling back to CPU "
                    "(allow_cpu_fallback = true)"
                )
                return "cpu"
            raise TranslationDeviceError(
                "CUDA is required for translation but no CUDA-capable "
                "PyTorch device is available."
            )

        index = self._config.cuda_device
        count = torch.cuda.device_count()
        if not (0 <= index < count):
            raise TranslationDeviceError(
                f"Invalid CUDA device index {index}: "
                f"device_count() = {count}."
            )
        return f"cuda:{index}"

    def _load_model(self) -> None:
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        device_str = self._resolve_device()

        if device_str.startswith("cuda"):
            gpu_name = torch.cuda.get_device_name(self._config.cuda_device)
            logger.info("[INFO] Translation device: %s", device_str)
            logger.info("[INFO] GPU: %s", gpu_name)
        else:
            logger.info("[INFO] Translation device: %s", device_str)

        # --- Precision validation ---
        if self._config.precision == "float16" and device_str == "cpu":
            raise TranslationConfigurationError(
                "precision='float16' is not supported on CPU; "
                "use 'auto' or 'float32'."
            )

        # --- Load tokenizer + model ---
        logger.info("[INFO] Loading translation model: %s", self._config.model_name)
        cache_kwargs = {}
        if self._config.model_cache_dir:
            cache_kwargs["cache_dir"] = self._config.model_cache_dir

        try:
            tokenizer = M2M100Tokenizer.from_pretrained(
                self._config.model_name, **cache_kwargs
            )
        except Exception as e:
            raise TranslationModelLoadError(
                f"Failed to load tokenizer for {self._config.model_name}: {e}"
            ) from e

        tokenizer.src_lang = self._config.source_language

        try:
            model = M2M100ForConditionalGeneration.from_pretrained(
                self._config.model_name, **cache_kwargs
            )
        except Exception as e:
            raise TranslationModelLoadError(
                f"Failed to load model {self._config.model_name}: {e}"
            ) from e

        # --- Precision ---
        precision = self._resolve_precision(model, device_str)
        self._precision_str = precision

        # --- Move to device ---
        try:
            model.to(device_str)
        except Exception as e:
            raise TranslationModelLoadError(
                f"Failed to move model to {device_str}: {e}"
            ) from e
        model.eval()

        self._model = model
        self._tokenizer = tokenizer
        self._device_str = device_str

        logger.info("[INFO] Model ready.")

    def _resolve_precision(self, model, device_str: str) -> str:
        """Apply precision strategy and return the effective precision name."""
        precision = self._config.precision
        if precision == "auto":
            try:
                import torch
                if device_str.startswith("cuda") and torch.cuda.is_available():
                    cap = torch.cuda.get_device_capability(self._config.cuda_device)
                    if cap[0] >= 7:  # Volta or newer supports fp16 well
                        model.half()
                        return "float16"
            except Exception:
                pass
            return "float32"
        elif precision == "float16":
            model.half()
            return "float16"
        else:
            return "float32"

    # ------------------------------------------------------------------
    # Internal: batched inference (official M2M100 generation pattern)
    # ------------------------------------------------------------------

    def _translate_impl(
        self, texts: Sequence[str], source_lang: str, target_lang: str
    ) -> List[TranslationResult]:
        """Translate a chunk of texts in one GPU batch.

        Uses the officially documented M2M100 pattern:
        tokenizer.src_lang = source; tokenize; model.generate with
        forced_bos_token_id = tokenizer.get_lang_id(target); batch_decode.
        """
        import torch

        tokenizer = self._tokenizer
        model = self._model
        device_str = self._device_str
        gen_cfg = self._config.generation

        target_lang_id = tokenizer.get_lang_id(target_lang)

        with self._lock, torch.inference_mode():
            # Atomic: set language + tokenize + generate + decode
            tokenizer.src_lang = source_lang

            encoded = tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            # Move every tensor to the device
            encoded = {
                key: value.to(device_str)
                for key, value in encoded.items()
            }

            generated = model.generate(
                **encoded,
                forced_bos_token_id=target_lang_id,
                max_new_tokens=gen_cfg.max_new_tokens,
                num_beams=gen_cfg.num_beams,
                length_penalty=gen_cfg.length_penalty,
                early_stopping=gen_cfg.early_stopping if gen_cfg.num_beams > 1 else False,
                no_repeat_ngram_size=3,
            )

            decoded = tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )

        results: List[TranslationResult] = []
        for text, translated in zip(texts, decoded):
            translated = translated.strip()
            results.append(
                TranslationResult(
                    source_text=text,
                    translated_text=translated,
                    source_language=source_lang,
                    target_language=target_lang,
                    model_name=self._config.model_name,
                    device=device_str,
                    compact_text=translated,
                    literal_text=translated,
                )
            )
        return results
