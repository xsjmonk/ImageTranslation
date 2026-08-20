"""Translation configuration — reusable, independent of FastAPI/ImageTranslation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import ceil
from typing import Optional


@dataclass
class GenerationConfig:
    """Deterministic sequence-to-sequence generation policy."""
    max_new_tokens: int = 512
    min_new_tokens: int = 1
    target_token_multiplier: float = 2.5
    short_text_max_new_tokens: int = 64
    num_beams: int = 4
    do_sample: bool = False
    no_repeat_ngram_size: int | None = None
    length_penalty: float = 1.0
    early_stopping: bool = True
    repetition_check: bool = True
    max_repeated_token_run: int = 3
    max_repeated_ngram_ratio: float = 0.35
    retry_on_degenerate_output: bool = True
    retry_num_beams: int = 1
    retry_max_new_tokens: int = 64

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("generation.max_new_tokens must be >= 1")
        if self.min_new_tokens < 1:
            raise ValueError("generation.min_new_tokens must be >= 1")
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError(
                "generation.min_new_tokens must not exceed max_new_tokens"
            )
        if self.target_token_multiplier <= 0:
            raise ValueError("generation.target_token_multiplier must be > 0")
        if self.short_text_max_new_tokens < self.min_new_tokens:
            raise ValueError(
                "generation.short_text_max_new_tokens must be >= min_new_tokens"
            )
        if self.num_beams < 1 or self.retry_num_beams < 1:
            raise ValueError("generation beam counts must be >= 1")
        if self.no_repeat_ngram_size is not None and self.no_repeat_ngram_size < 2:
            raise ValueError("generation.no_repeat_ngram_size must be null or >= 2")
        if self.max_repeated_token_run < 2:
            raise ValueError("generation.max_repeated_token_run must be >= 2")
        if not 0 <= self.max_repeated_ngram_ratio <= 1:
            raise ValueError(
                "generation.max_repeated_ngram_ratio must be between 0 and 1"
            )
        if self.retry_max_new_tokens < self.min_new_tokens:
            raise ValueError(
                "generation.retry_max_new_tokens must be >= min_new_tokens"
            )

    def target_budget(self, source_tokens: int, explicit: int | None = None) -> int:
        """Return a validated target budget from measured source tokens."""
        if source_tokens < 1:
            raise ValueError("source_tokens must be >= 1")
        if explicit is not None:
            if explicit < self.min_new_tokens:
                raise ValueError(
                    "explicit target budget must be >= generation.min_new_tokens"
                )
            return min(self.max_new_tokens, explicit)
        budget = max(
            self.min_new_tokens,
            ceil(source_tokens * self.target_token_multiplier),
        )
        if source_tokens <= 32:
            budget = min(budget, self.short_text_max_new_tokens)
        return min(self.max_new_tokens, budget)


@dataclass(frozen=True)
class QualityConfig:
    """Input-quality policy shared by direct and HTTP translation paths."""
    unknown_token_policy: str = "warn"  # allow | warn | reject

    def __post_init__(self) -> None:
        if self.unknown_token_policy not in {"allow", "warn", "reject"}:
            raise ValueError(
                "quality.unknown_token_policy must be one of "
                "'allow', 'warn', or 'reject'"
            )
@dataclass
class TranslationConfig:
    """Reusable translation configuration.

    Used by both ImageTranslation pipeline and the standalone translation server.

    ``model_cache_dir`` is an injected, resolved cache root supplied by the
    server composition root. The server owns cache policy; this field exists
    so the shared translator can consume that immutable resolved value.
    Standalone callers may leave it unset, but must not infer a model-specific
    cache location.
    - model_revision: revision resolved consistently for snapshot, tokenizer,
      and model loading (default "main").
    - allow_model_download: permits downloading only when the requested
      revision is not already available in the configured cache.
    - local_files_only: forbids ALL network access; a cache miss is a clear
      error. Contradictory with allow_model_download=true (rejected at
      validation time).
    """
    model_name: str = "facebook/nllb-200-distilled-600M"
    model_family: str = "nllb"
    model_revision: str = "main"
    source_language: str = "zho_Hans"
    target_language: str = "eng_Latn"
    device: str = "cuda"
    cuda_device: int = 0
    allow_cpu_fallback: bool = False
    precision: str = "auto"  # auto | float16 | float32
    batch_size: int = 8
    max_input_characters: int = 4000
    max_input_tokens: int = 1024
    commercial_use: bool = False
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    model_cache_dir: Optional[str] = None
    allow_model_download: bool = True
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if self.precision not in ("auto", "float16", "float32"):
            raise ValueError(
                f"precision must be 'auto', 'float16', or 'float32', got '{self.precision}'"
            )
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got '{self.device}'")
        if self.cuda_device < 0:
            raise ValueError("cuda_device must be >= 0")
        if self.max_input_characters < 1:
            raise ValueError("max_input_characters must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_input_tokens < 1:
            raise ValueError("max_input_tokens must be >= 1")
        if self.model_family not in {"nllb", "helsinki"}:
            raise ValueError(
                "model_family must be 'nllb' or 'helsinki'"
            )
        if self.commercial_use and self.model_family == "nllb":
            raise ValueError(
                "NLLB is CC-BY-NC-4.0 and cannot be used for commercial_use; "
                "configure model_family='helsinki' explicitly"
            )
        if not self.model_name or not isinstance(self.model_name, str):
            raise ValueError("model_name must be a non-empty string")
        if not self.model_revision or not isinstance(self.model_revision, str):
            raise ValueError("model_revision must be a non-empty string")
        if self.local_files_only and self.allow_model_download:
            raise ValueError(
                "local_files_only=true contradicts allow_model_download=true; "
                "set allow_model_download=false for a fully offline cache"
            )

    def effective_device(self) -> str:
        """Return the device string to use, e.g. 'cuda:0'."""
        if self.device == "cuda":
            return f"cuda:{self.cuda_device}"
        return "cpu"


@dataclass
class StructuredConfig:
    """Structured (HTML-aware) translation configuration.

    Documented operational limits:
    - max_chapter_characters: hard cap per document; larger input is rejected
      with a clear error (never silently truncated).
    - max_segment_tokens: per-segment source budget for the model
      (tokenizer-measured with truncation=False).
    - max_target_tokens: per-segment target budget passed to generation;
      expansion assumption: target ≈ 2.5× source tokens, capped here.
    - context_window_tokens: CONTEXT INJECTION IS NOT IMPLEMENTED for
      the configured model family (no reliable context API). This setting is
      explicitly UNSUPPORTED: it MUST stay 0 and any non-zero value raises
      a configuration error. Segment adjacency (prev/next segment ids) is a
      diagnostics-only record; it is NOT context and is never sent to the
      model.
    - preserve_patterns: configurable regex patterns for project-specific
      model formats/product codes. Matches inside non-Chinese (Latin) spans
      become ``model_number_protected`` runs — preserved exactly, never
      rewritten by the model. Applied on top of the built-in identifier
      rules; invalid patterns raise a configuration error.
    - translatable_attributes: allowlist of human-readable attribute values
      (e.g. alt, title, aria-label) that may be translated. URL/code/style
      attributes are never translated. Empty = no attributes translated.
    - excluded_tags / excluded_classes: subtrees never translated.
    - segment_warning_seconds: warning threshold only — a slow segment logs
      a warning; it does NOT cancel work.
    - max_total_seconds: real deadline enforced BETWEEN segments (an
      in-flight segment is not interruptible; once it returns, work stops
      and the request fails with a clear error).
    - max_retries_per_segment: stricter-placeholder retry count.
    - batch_size: segments grouped into one model batch call (first pass),
      preserving source order; failed batch items are retried individually.
    - concurrency: bounds concurrent translations (GPU holds one model).
    """
    enabled: bool = True
    max_chapter_characters: int = 100_000
    max_segment_tokens: int = 450
    max_target_tokens: int = 400
    context_window_tokens: int = 0
    preserve_patterns: tuple = ()  # tuple[str, ...] — regexes (model formats)
    translatable_attributes: tuple = ()  # e.g. ("alt", "title", "aria-label")
    excluded_tags: tuple = ("script", "style", "code", "pre")
    excluded_classes: tuple = ("notranslate",)
    segment_warning_seconds: float = 60.0
    max_total_seconds: float = 600.0
    max_retries_per_segment: int = 1
    batch_size: int = 4
    concurrency: int = 1

    def __post_init__(self) -> None:
        if self.max_chapter_characters < 1:
            raise ValueError("max_chapter_characters must be >= 1")
        if self.max_segment_tokens < 8:
            raise ValueError("max_segment_tokens must be >= 8")
        if self.max_target_tokens < 8:
            raise ValueError("max_target_tokens must be >= 8")
        if self.context_window_tokens != 0:
            raise ValueError(
                "context_window_tokens must be 0: context injection is NOT "
                "implemented for the configured model family (no reliable "
                "context API); this "
                "setting is explicitly unsupported"
            )
        if self.segment_warning_seconds <= 0:
            raise ValueError("segment_warning_seconds must be > 0")
        if self.max_total_seconds <= 0:
            raise ValueError("max_total_seconds must be > 0")
        if self.max_retries_per_segment < 0:
            raise ValueError("max_retries_per_segment must be >= 0")
        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        # --- preserve_patterns validation ---
        for pat in self.preserve_patterns:
            if not isinstance(pat, str) or not pat.strip():
                raise ValueError(
                    "preserve_patterns entries must be non-empty regex strings"
                )
            try:
                re.compile(pat)
            except re.error as e:
                raise ValueError(
                    f"preserve_patterns entry {pat!r} is not a valid regex: {e}"
                )
