"""M2M100 418M GPU translator — lazy-loaded, CUDA-required by default."""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Sequence

from .base import Translator
from .config import TranslationConfig
from .models import TranslationResult, TranslationRuntimeInfo
from .text_utils import preprocess

logger = logging.getLogger(__name__)


class TranslationError(Exception):
    """Base exception for translation failures."""


class TranslationConfigurationError(TranslationError):
    """Invalid configuration."""


class TranslationDeviceError(TranslationError):
    """CUDA/device unavailable."""


class TranslationInputError(TranslationError):
    """Invalid input text."""


class TranslationModelLoadError(TranslationError):
    """Failed to load model/tokenizer."""


class M2M100Translator(Translator):
    """M2M100 418M neural machine translation engine.

    Uses facebook/m2m100_418M via Hugging Face Transformers on NVIDIA GPU.
    Model is lazy-loaded on first translate_text() call.
    Thread-safe: serializes GPU inference with a lock.
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
            if cuda_ok:
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
        return self._translate_impl(cleaned, source_lang, target_lang)

    def translate_batch_texts(
        self, texts: Sequence[str], source_lang: str = "zh", target_lang: str = "en"
    ) -> List[TranslationResult]:
        """Translate multiple strings."""
        if not texts:
            return []
        self._ensure_loaded()

        cleaned = [
            preprocess(t, max_characters=self._config.max_input_characters)
            for t in texts
        ]

        # Process in configurable batch sizes
        results: List[TranslationResult] = []
        batch_size = self._config.batch_size
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i : i + batch_size]
            for txt in batch:
                results.append(self._translate_impl(txt, source_lang, target_lang))

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

    def _load_model(self) -> None:
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        # --- CUDA check ---
        cuda_ok = torch.cuda.is_available()
        if self._config.device == "cuda":
            if not cuda_ok:
                if self._config.allow_cpu_fallback:
                    logger.warning("CUDA not available, falling back to CPU")
                else:
                    raise TranslationDeviceError(
                        "CUDA is required for translation but no CUDA-capable "
                        "PyTorch device is available."
                    )

        device_str = self._config.effective_device()
        if self._config.device == "cuda" and cuda_ok:
            gpu_name = torch.cuda.get_device_name(self._config.cuda_device)
            logger.info("[INFO] Translation device: %s", device_str)
            logger.info("[INFO] GPU: %s", gpu_name)
        else:
            logger.info("[INFO] Translation device: %s", device_str)

        # --- Load tokenizer ---
        logger.info("[INFO] Loading translation model: %s", self._config.model_name)
        cache_kwargs = {}
        if self._config.model_cache_dir:
            cache_kwargs["cache_dir"] = self._config.model_cache_dir

        tokenizer = M2M100Tokenizer.from_pretrained(
            self._config.model_name, **cache_kwargs
        )
        tokenizer.src_lang = self._config.source_language

        # --- Load model ---
        model = M2M100ForConditionalGeneration.from_pretrained(
            self._config.model_name, **cache_kwargs
        )

        # --- Precision ---
        precision = self._resolve_precision(model, device_str)
        self._precision_str = precision

        # --- Move to device ---
        model.to(device_str)
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
                    # Check if GPU supports fp16
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
    # Internal: inference
    # ------------------------------------------------------------------

    def _translate_impl(
        self, text: str, source_lang: str, target_lang: str
    ) -> TranslationResult:
        import torch

        tokenizer = self._tokenizer
        model = self._model
        device_str = self._device_str

        # Set source language for tokenizer
        tokenizer.src_lang = source_lang

        # Tokenize
        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = encoded.input_ids.to(device_str)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device_str)

        # Forced BOS token for target language
        target_lang_id = tokenizer.get_lang_id(target_lang)

        gen_cfg = self._config.generation

        # Build decoder input with target language prefix for reliable generation
        decoder_input_ids = torch.tensor(
            [[target_lang_id]], device=device_str, dtype=torch.long
        )

        with self._lock, torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=gen_cfg.max_new_tokens,
                num_beams=gen_cfg.num_beams,
                length_penalty=gen_cfg.length_penalty,
                early_stopping=gen_cfg.early_stopping if gen_cfg.num_beams > 1 else False,
                no_repeat_ngram_size=3,
            )

        # Decode: with decoder_input_ids, generated contains only decoder output.
        # Skip the decoder_input_ids prefix to get just the new tokens.
        output_ids = generated[0][decoder_input_ids.shape[1]:]
        translated = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return TranslationResult(
            source_text=text,
            translated_text=translated,
            source_language=source_lang,
            target_language=target_lang,
            model_name=self._config.model_name,
            device=device_str,
            compact_text=translated,
            literal_text=translated,
        )
