"""Hy-MT2 causal-LM translation backend — additive, config-selected."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from typing import List, Optional, Sequence

import torch

from .base import Translator
from .config import TranslationConfig, TranslationStyle, resolve_translation_style
from .exceptions import (
    TranslationConfigurationError,
    TranslationDeviceError,
    TranslationError,
    TranslationInputError,
    TranslationModelLoadError,
)
from .language_names import resolve_language_name
from .model_snapshot import resolve_model_snapshot, verify_causal_lm_snapshot
from .models import ResolvedModel, TranslationResult, TranslationRuntimeInfo
from .text_utils import preprocess

logger = logging.getLogger(__name__)


class HyMt2Translator(Translator):
    """Tencent Hy-MT2 causal language model backend.

    Uses instruction-style prompts and chat templates when available. Does not
    fall back to the seq2seq backend on load or inference failure.
    """

    BACKEND = "hymt2"

    def __init__(self, config: TranslationConfig) -> None:
        if config.backend != self.BACKEND:
            raise TranslationConfigurationError(
                f"HyMt2Translator requires backend='hymt2', got {config.backend!r}"
            )
        self._config = config
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None
        self._device_str: str = ""
        self._precision_str: str = ""
        self._snapshot_path: str = ""
        self._cache_status: str = ""
        self._cache_dir: str = ""
        self._resolved: Optional[ResolvedModel] = None
        self._offline = config.local_files_only or not config.allow_model_download
        self._lock = threading.Lock()
        self._encode_used_chat_template = False

    @property
    def name(self) -> str:
        return f"hymt2@{self._config.model_name}"

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

        tokenizer_class = ""
        if self._tokenizer is not None:
            tokenizer_class = type(self._tokenizer).__name__

        return TranslationRuntimeInfo(
            backend=self.BACKEND,
            model_name=self._config.model_name,
            model_family=self._config.model_family,
            model_revision=self._config.model_revision,
            source_language=self._config.source_language,
            target_language=self._config.target_language,
            device=self._device_str,
            precision=self._precision_str,
            dtype=self._precision_str,
            cuda_available=cuda_ok,
            gpu_name=gpu_name,
            ready=self._model is not None,
            cache_dir=self._cache_dir,
            snapshot_path=self._snapshot_path,
            cache_status=self._cache_status,
            local_files_only=self._config.local_files_only,
            offline=self._offline,
            tokenizer_class=tokenizer_class,
            max_new_tokens=self._config.generation.max_new_tokens,
            max_input_tokens=self._config.max_input_tokens,
        )

    def translate_text(
        self,
        text: str,
        source_lang: str = "zh",
        target_lang: str = "en",
        style: TranslationStyle | str | None = None,
    ) -> TranslationResult:
        style = resolve_translation_style(style, self._config.default_style)
        self._ensure_loaded()
        cleaned = preprocess(text, max_characters=self._config.max_input_characters)
        return self._translate_impl(
            [cleaned], source_lang, target_lang, style=style
        )[0]

    def translate_batch_texts(
        self,
        texts: Sequence[str],
        source_lang: str = "zh",
        target_lang: str = "en",
        max_new_tokens: int | None = None,
        style: TranslationStyle | str | None = None,
    ) -> List[TranslationResult]:
        if not texts:
            return []
        style = resolve_translation_style(style, self._config.default_style)
        self._ensure_loaded()
        cleaned = [
            preprocess(text, max_characters=self._config.max_input_characters)
            for text in texts
        ]
        results: List[TranslationResult] = []
        batch_size = self._config.batch_size
        for start in range(0, len(cleaned), batch_size):
            chunk = cleaned[start : start + batch_size]
            results.extend(
                self._translate_impl(
                    chunk,
                    source_lang,
                    target_lang,
                    max_new_tokens=max_new_tokens,
                    style=style,
                )
            )
        if len(results) != len(cleaned):
            raise TranslationError(
                f"translate_batch_texts accumulated {len(results)} results "
                f"for {len(cleaned)} inputs"
            )
        return results

    def warmup(self) -> None:
        self._ensure_loaded()

    def check_cache(self) -> ResolvedModel:
        return self._resolve_model_snapshot()

    def measure_source_tokens(self, text: str, source_lang: str = "zh") -> int:
        self._ensure_loaded()
        prompt = self._build_prompt(text, source_lang, "en")
        encoded = self._encode_prompts([prompt])
        return int(encoded["input_ids"].shape[1])

    def _build_prompt(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        target_name = resolve_language_name(
            target_lang, self._config.target_language
        )
        return (
            f"Translate the following text into {target_name}. "
            "Note that you should only output the translated result "
            "without any additional explanation:\n\n"
            f"{text}"
        )

    def _tokenizer_has_chat_template(self, tokenizer: object) -> bool:
        template = getattr(tokenizer, "chat_template", None)
        if template:
            return True
        getter = getattr(tokenizer, "get_chat_template", None)
        if callable(getter):
            try:
                return bool(getter())
            except Exception:
                return False
        return False

    def _normalize_encoded_batch(
        self,
        encoded: object,
        *,
        expected_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        if isinstance(encoded, torch.Tensor):
            input_ids = encoded
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

        if not isinstance(encoded, Mapping):
            raise TranslationModelLoadError(
                "Hy-MT2 chat template encoding returned unexpected type: "
                f"{type(encoded)!r}"
            )

        normalized = dict(encoded)
        input_ids = normalized.get("input_ids")
        if input_ids is None:
            raise TranslationModelLoadError(
                "Hy-MT2 chat template encoding missing input_ids"
            )
        if int(input_ids.shape[0]) != expected_batch_size:
            raise TranslationModelLoadError(
                "Hy-MT2 chat template batch size mismatch: "
                f"expected {expected_batch_size}, got {input_ids.shape[0]}"
            )
        if len(input_ids.shape) != 2:
            raise TranslationModelLoadError(
                "Hy-MT2 chat template input_ids must be a 2-D batch tensor"
            )
        return normalized

    def _encode_prompts(self, prompts: Sequence[str]) -> dict[str, torch.Tensor]:
        tokenizer = self._tokenizer
        assert tokenizer is not None

        if self._tokenizer_has_chat_template(tokenizer):
            if not hasattr(tokenizer, "apply_chat_template"):
                raise TranslationModelLoadError(
                    "Hy-MT2 tokenizer advertises a chat template but "
                    "apply_chat_template is unavailable"
                )
            messages_batch = [
                [{"role": "user", "content": prompt}] for prompt in prompts
            ]
            try:
                encoded = tokenizer.apply_chat_template(
                    messages_batch,
                    return_tensors="pt",
                    return_dict=True,
                    padding=True,
                    add_generation_prompt=True,
                    truncation=False,
                )
            except Exception as exc:
                raise TranslationModelLoadError(
                    f"Failed to apply Hy-MT2 chat template: {exc}"
                ) from exc
            self._encode_used_chat_template = True
            return self._normalize_encoded_batch(
                encoded,
                expected_batch_size=len(prompts),
            )

        # Compatibility path for tokenizers without a chat template (test doubles).
        self._encode_used_chat_template = False
        encoded = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        return self._normalize_encoded_batch(
            encoded,
            expected_batch_size=len(prompts),
        )

    def _resolve_device(self) -> str:
        import torch

        requested = self._config.device
        cuda_ok = torch.cuda.is_available()
        if requested == "cpu":
            return "cpu"
        if not cuda_ok:
            if self._config.allow_cpu_fallback:
                logger.warning(
                    "[WARN] CUDA not available, falling back to CPU "
                    "(allow_cpu_fallback = true)"
                )
                return "cpu"
            raise TranslationDeviceError(
                "CUDA is required for Hy-MT2 translation but no CUDA-capable "
                "PyTorch device is available."
            )
        index = self._config.cuda_device
        count = torch.cuda.device_count()
        if not (0 <= index < count):
            raise TranslationDeviceError(
                f"Invalid CUDA device index {index}: device_count() = {count}."
            )
        return f"cuda:{index}"

    def _resolve_model_snapshot(self) -> ResolvedModel:
        cfg = self._config
        return resolve_model_snapshot(
            model_name=cfg.model_name,
            model_revision=cfg.model_revision,
            model_cache_dir=cfg.model_cache_dir,
            offline=self._offline,
            model_family=cfg.model_family,
            verify_snapshot=verify_causal_lm_snapshot,
        )

    def _resolve_dtype(self, device_str: str):
        import torch

        precision = self._config.precision
        if precision == "float16":
            if device_str == "cpu":
                raise TranslationConfigurationError(
                    "precision='float16' is not supported on CPU; "
                    "use 'auto' or 'float32'."
                )
            return torch.float16
        if precision == "float32":
            return torch.float32
        return torch.float16 if device_str.startswith("cuda") else torch.float32

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            self._load_model()

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device_str = self._resolve_device()
        if device_str.startswith("cuda"):
            logger.info(
                "[INFO] Hy-MT2 device: %s (%s)",
                device_str,
                torch.cuda.get_device_name(self._config.cuda_device),
            )
        else:
            logger.info("[INFO] Hy-MT2 device: %s", device_str)

        resolved = self._resolve_model_snapshot()
        snapshot_path = resolved.snapshot_path
        dtype = self._resolve_dtype(device_str)
        self._precision_str = str(dtype).replace("torch.", "")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot_path,
                local_files_only=self._offline,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise TranslationModelLoadError(
                f"Failed to load Hy-MT2 tokenizer from {snapshot_path}: {exc}"
            ) from exc

        try:
            model = AutoModelForCausalLM.from_pretrained(
                snapshot_path,
                local_files_only=self._offline,
                dtype=dtype,
                trust_remote_code=True,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                snapshot_path,
                local_files_only=self._offline,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise TranslationModelLoadError(
                f"Failed to load Hy-MT2 model from {snapshot_path}: {exc}"
            ) from exc

        try:
            model.to(device_str)
        except Exception as exc:
            raise TranslationModelLoadError(
                f"Failed to move Hy-MT2 model to {device_str}: {exc}"
            ) from exc

        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._device_str = device_str
        self._snapshot_path = snapshot_path
        self._cache_status = resolved.cache_status
        self._cache_dir = self._config.model_cache_dir or ""
        self._resolved = resolved
        logger.info(
            "[INFO] Hy-MT2 model ready (snapshot=%s, %s).",
            snapshot_path,
            resolved.cache_status,
        )

    def _translate_impl(
        self,
        texts: Sequence[str],
        source_lang: str,
        target_lang: str,
        max_new_tokens: int | None = None,
        style: TranslationStyle = TranslationStyle.SENTENCE,
    ) -> List[TranslationResult]:
        import torch

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        prompts = [
            self._build_prompt(text, source_lang, target_lang) for text in texts
        ]
        gen_cfg = self._config.generation
        with self._lock, torch.inference_mode():
            encoded = self._encode_prompts(prompts)
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            input_width = int(input_ids.shape[1])
            prompt_lengths = (
                attention_mask.sum(dim=1).tolist()
                if attention_mask is not None
                else [input_width] * input_ids.shape[0]
            )
            actual_tokens = int(max(prompt_lengths))
            source_token_estimate = max(
                int(length) for length in prompt_lengths
            )
            policy = gen_cfg.resolve_style(style, source_token_estimate)
            target_budget = (
                min(policy.max_new_tokens, max(policy.min_new_tokens, max_new_tokens))
                if max_new_tokens is not None
                else min(
                    policy.max_new_tokens,
                    max(
                        policy.min_new_tokens,
                        gen_cfg.target_budget(source_token_estimate),
                    ),
                )
            )
            ceiling = self._config.max_input_tokens
            if actual_tokens > ceiling:
                raise TranslationInputError(
                    f"input exceeds Hy-MT2 token budget: measured {actual_tokens} "
                    f"source tokens, ceiling {ceiling}; text was NOT truncated"
                )

            device = next(model.parameters()).device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            generation_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": target_budget,
                "num_beams": policy.num_beams,
                "do_sample": policy.do_sample,
            }
            if attention_mask is not None:
                generation_kwargs["attention_mask"] = attention_mask
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id
            if policy.do_sample:
                generation_kwargs["temperature"] = gen_cfg.temperature
            if not policy.do_sample:
                torch.manual_seed(0)
                if self._device_str.startswith("cuda"):
                    torch.cuda.manual_seed_all(0)

            started = time.perf_counter()
            generated = model.generate(**generation_kwargs)

            results: List[TranslationResult] = []
            for index, source_text in enumerate(texts):
                new_tokens = generated[index, input_width:]
                translated = tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                ).strip()
                results.append(
                    TranslationResult(
                        source_text=source_text,
                        translated_text=translated,
                        source_language=source_lang,
                        target_language=target_lang,
                        style=policy.style.value,
                        model_name=self._config.model_name,
                        device=self._device_str,
                        compact_text=translated,
                        literal_text=translated,
                    )
                )
            logger.debug(
                "Hy-MT2 translated %d item(s) in %.3fs",
                len(results),
                time.perf_counter() - started,
            )
            return results
