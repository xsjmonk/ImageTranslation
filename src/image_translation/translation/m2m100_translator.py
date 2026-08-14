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
    TranslationError,
    TranslationInputError,
    TranslationModelLoadError,
)
from .models import ResolvedModel, TranslationResult, TranslationRuntimeInfo
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
        self._snapshot_path: str = ""
        self._cache_status: str = ""
        self._cache_dir: str = ""
        self._resolved: Optional[ResolvedModel] = None
        # One computed effective-offline flag for resolution, tokenizer,
        # model, and HTML token measurement.
        self._offline = config.local_files_only or not config.allow_model_download
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
            model_revision=self._config.model_revision,
            device=self._device_str,
            precision=self._precision_str,
            cuda_available=cuda_ok,
            gpu_name=gpu_name,
            ready=self._model is not None,
            cache_dir=self._cache_dir,
            snapshot_path=self._snapshot_path,
            cache_status=self._cache_status,
            local_files_only=self._config.local_files_only,
            offline=self._offline,
        )

    def translate_text(
        self, text: str, source_lang: str = "zh", target_lang: str = "en"
    ) -> TranslationResult:
        """Translate a single string."""
        self._ensure_loaded()
        cleaned = preprocess(text, max_characters=self._config.max_input_characters)
        return self._translate_impl([cleaned], source_lang, target_lang)[0]

    def translate_batch_texts(
        self,
        texts: Sequence[str],
        source_lang: str = "zh",
        target_lang: str = "en",
        max_new_tokens: int | None = None,
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
            results.extend(
                self._translate_impl(
                    chunk, source_lang, target_lang, max_new_tokens=max_new_tokens
                )
            )
        # Final accumulated-count invariant: the plain batch path must never
        # silently return fewer (or more) translations than requested.
        if len(results) != len(cleaned):
            raise TranslationError(
                f"translate_batch_texts accumulated {len(results)} results "
                f"for {len(cleaned)} inputs"
            )
        return results

    def warmup(self) -> None:
        """Explicitly trigger model loading. Idempotent."""
        self._ensure_loaded()

    def check_cache(self) -> ResolvedModel:
        """Validate the configured model cache WITHOUT loading the model.

        Reuses the authoritative snapshot resolution and completeness
        check. Side-effect free apart from the documented cache-directory
        creation; nothing is retained and no GPU/tokenizer/model is loaded.

        Returns:
            The ResolvedModel (snapshot path, revision, cache status,
            offline flag).

        Raises:
            TranslationModelLoadError: offline cache miss, download
                failure, or incomplete snapshot.
        """
        return self._resolve_model_snapshot()

    def measure_source_tokens(self, text: str, source_lang: str = "zh") -> int:
        """Measure source tokens WITHOUT truncation using the EXACT
        tokenizer loaded for inference.

        - lazy-loads the model/tokenizer exactly once (same authoritative
          resolution used by inference);
        - sets the source language consistently with inference;
        - tokenizes with ``truncation=False``;
        - never calls ``from_pretrained`` a second time and never uses a
          second cache or remote model identifier.
        """
        self._ensure_loaded()
        with self._lock:
            self._tokenizer.src_lang = source_lang
            return len(
                self._tokenizer(text, truncation=False)["input_ids"]
            )

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

    def _resolve_model_snapshot(self) -> tuple:
        """Authoritative model resolution used by tokenizer AND model loading.

        Returns (snapshot_path, cache_status) with cache_status one of
        "cache_hit" | "download".

        - Resolves the configured cache root to an absolute path (the
          configured location is authoritative; the implementation never
          silently falls back to the HF default cache when set).
        - First tries Hugging Face local-only snapshot resolution; success
          = cache hit (reused, no network).
        - On a miss: if downloads are allowed, downloads into the
          configured root; if offline (local_files_only or downloads
          disabled), fails with an actionable cache-missing error and no
          network access.
        - Verifies the resolved snapshot contains the files M2M100 requires
          before it can be reported ready.

        Raises:
            TranslationModelLoadError: offline cache miss, download
                failure, unusable cache path, or incomplete snapshot.
        """
        import os
        from pathlib import Path

        from huggingface_hub import snapshot_download

        cfg = self._config
        offline = self._offline

        cache_root: Optional[str] = None
        if cfg.model_cache_dir:
            cache_root = os.path.expandvars(cfg.model_cache_dir)
            root = Path(cache_root).expanduser().resolve()
            cache_root = str(root)
            if offline and not root.is_dir():
                raise TranslationModelLoadError(
                    f"offline mode: configured model cache does not exist: "
                    f"{cache_root}; pre-download the model (see README) or "
                    f"fix model_cache_dir"
                )
            try:
                root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise TranslationModelLoadError(
                    f"cannot create configured model cache {cache_root}: {exc}"
                ) from exc

        logger.info(
            "[INFO] Model cache: %s (model=%s revision=%s offline=%s)",
            cache_root or "HF default", cfg.model_name,
            cfg.model_revision, offline,
        )

        def _snapshot(local_only: bool) -> str:
            kwargs = {}
            if cache_root:
                kwargs["cache_dir"] = cache_root
            return snapshot_download(
                repo_id=cfg.model_name,
                revision=cfg.model_revision,
                local_files_only=local_only,
                **kwargs,
            )

        # 1) Cache-hit probe: local resolution only, never a network call.
        try:
            snapshot_path = _snapshot(local_only=True)
            cache_status = "cache_hit"
            logger.info("[INFO] Model cache HIT (reused): %s", snapshot_path)
        except Exception as exc:
            if offline:
                raise TranslationModelLoadError(
                    f"offline model cache miss: {cfg.model_name} revision "
                    f"{cfg.model_revision} not found in cache "
                    f"{cache_root or 'HF default'}; pre-download the model "
                    f"or set allow_model_download=true"
                ) from exc
            logger.info(
                "[INFO] Model cache MISS; downloading %s revision %s "
                "into %s ...",
                cfg.model_name, cfg.model_revision,
                cache_root or "HF default",
            )
            try:
                snapshot_path = _snapshot(local_only=False)
            except Exception as exc2:
                raise TranslationModelLoadError(
                    f"model download failed for {cfg.model_name} revision "
                    f"{cfg.model_revision} into {cache_root or 'HF default'}: "
                    f"{exc2}"
                ) from exc2
            cache_status = "download"
            logger.info("[INFO] Model download COMPLETE: %s", snapshot_path)

        self._verify_snapshot(snapshot_path)
        return ResolvedModel(
            snapshot_path=snapshot_path,
            model_name=cfg.model_name,
            revision=cfg.model_revision,
            cache_dir=cache_root or "",
            cache_status=cache_status,
            offline=offline,
        )

    @staticmethod
    def _verify_snapshot(snapshot_path: str) -> None:
        """Fail before ready if the resolved snapshot misses required files."""
        from pathlib import Path

        root = Path(snapshot_path)
        required = [
            ("config.json", ["config.json"]),
            ("model weights", ["model.safetensors", "pytorch_model.bin"]),
            ("tokenizer files", ["tokenizer.json", "sentencepiece.bpe.model"]),
        ]
        missing = [
            label for label, candidates in required
            if not any((root / c).is_file() for c in candidates)
        ]
        if missing:
            raise TranslationModelLoadError(
                f"incomplete model snapshot at {snapshot_path}: missing "
                f"{', '.join(missing)}; re-download the model or repair "
                f"the cache"
            )

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

        # --- Authoritative model resolution (cache policy) ---
        logger.info("[INFO] Loading translation model: %s", self._config.model_name)
        resolved = self._resolve_model_snapshot()
        snapshot_path = resolved.snapshot_path
        # Both the tokenizer and the model load from the SAME resolved
        # snapshot for the SAME revision; no ambiguous remote identifier.
        load_kwargs = {"local_files_only": self._offline}

        try:
            tokenizer = M2M100Tokenizer.from_pretrained(
                snapshot_path, **load_kwargs
            )
        except Exception as e:
            raise TranslationModelLoadError(
                f"Failed to load tokenizer from {snapshot_path}: {e}"
            ) from e

        tokenizer.src_lang = self._config.source_language

        try:
            model = M2M100ForConditionalGeneration.from_pretrained(
                snapshot_path, **load_kwargs
            )
        except Exception as e:
            raise TranslationModelLoadError(
                f"Failed to load model from {snapshot_path}: {e}"
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
        self._snapshot_path = snapshot_path
        self._cache_status = resolved.cache_status
        self._cache_dir = self._config.model_cache_dir or ""
        self._resolved = resolved

        logger.info("[INFO] Model ready (snapshot=%s, %s).",
                    snapshot_path, resolved.cache_status)

    def _resolve_precision(self, model, device_str: str) -> str:
        """Apply precision strategy and return the effective precision name.

        'auto' always resolves to FP32: quality is the priority while the
        FP32 baseline is being established. Lower precision (float16) is
        only applied when explicitly requested.
        """
        precision = self._config.precision
        if precision == "float16":
            model.half()
            return "float16"
        return "float32"

    # ------------------------------------------------------------------
    # Internal: batched inference (official M2M100 generation pattern)
    # ------------------------------------------------------------------

    def _translate_impl(
        self,
        texts: Sequence[str],
        source_lang: str,
        target_lang: str,
        max_new_tokens: int | None = None,
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
        target_budget = max_new_tokens if max_new_tokens is not None else gen_cfg.max_new_tokens

        target_lang_id = tokenizer.get_lang_id(target_lang)

        with self._lock, torch.inference_mode():
            # Atomic: set language + tokenize + generate + decode
            tokenizer.src_lang = source_lang

            encoded = tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

            # --- Explicit over-budget rejection: no silent truncation ---
            # The structured path validates budgets before generation; this
            # is the final hard guard (model context window minus target
            # budget). Plain text beyond the model window fails explicitly.
            actual_tokens = encoded["input_ids"].shape[1]
            ceiling = model.config.max_position_embeddings - target_budget - 8
            if actual_tokens > ceiling:
                raise TranslationInputError(
                    f"input exceeds model token budget: measured {actual_tokens} "
                    f"source tokens, ceiling {ceiling} "
                    f"(max_position_embeddings={model.config.max_position_embeddings}, "
                    f"target_budget={target_budget}); text was NOT truncated"
                )

            # --- Diagnostics (debug): tokenizer output before moving ---
            logger.debug(
                "DIAG source_lang=%s target_lang=%s forced_bos_token_id=%s "
                "input_ids=%s attention_mask=%s token_count=%s decoded_source=%r",
                source_lang,
                target_lang,
                target_lang_id,
                encoded["input_ids"].tolist(),
                encoded.get("attention_mask").tolist()
                if encoded.get("attention_mask") is not None else None,
                actual_tokens,
                tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=True),
            )

            # Move every tensor to the device
            encoded = {
                key: value.to(device_str)
                for key, value in encoded.items()
            }

            # --- Diagnostics (debug): dtype/device before generation ---
            logger.debug(
                "DIAG model_dtype=%s input_dtype=%s device=%s precision=%s "
                "num_beams=%s max_new_tokens=%s length_penalty=%s early_stopping=%s "
                "no_repeat_ngram_size=unset",
                next(model.parameters()).dtype,
                encoded["input_ids"].dtype,
                device_str,
                self._precision_str,
                gen_cfg.num_beams,
                gen_cfg.max_new_tokens,
                gen_cfg.length_penalty,
                gen_cfg.early_stopping,
            )

            generated = model.generate(
                **encoded,
                forced_bos_token_id=target_lang_id,
                max_new_tokens=target_budget,
                num_beams=gen_cfg.num_beams,
                length_penalty=gen_cfg.length_penalty,
                early_stopping=gen_cfg.early_stopping if gen_cfg.num_beams > 1 else False,
            )

            # --- Diagnostics (debug): raw generated token IDs before decode ---
            logger.debug(
                "DIAG generated_ids=%s",
                [g.tolist() for g in generated],
            )

            decoded = tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )

        # Decode cardinality invariant: batch_decode must return exactly one
        # output per input — never silently zip away a mismatch.
        if len(decoded) != len(texts):
            raise TranslationError(
                f"model returned {len(decoded)} decoded outputs for "
                f"{len(texts)} inputs"
            )

        results: List[TranslationResult] = []
        for index, text in enumerate(texts):
            translated = decoded[index]
            if not isinstance(translated, str):
                raise TranslationError(
                    f"model returned non-string decoded output at item "
                    f"{index}: {type(translated).__name__}"
                )
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
