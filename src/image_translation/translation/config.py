"""Translation configuration — reusable, independent of FastAPI/ImageTranslation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationConfig:
    """M2M100 generation parameter defaults."""
    max_new_tokens: int = 256
    num_beams: int = 4
    length_penalty: float = 1.0
    early_stopping: bool = True


@dataclass
class TranslationConfig:
    """Reusable translation configuration.

    Used by both ImageTranslation pipeline and the standalone translation server.
    """
    model_name: str = "facebook/m2m100_418M"
    source_language: str = "zh"
    target_language: str = "en"
    device: str = "cuda"
    cuda_device: int = 0
    allow_cpu_fallback: bool = False
    precision: str = "auto"  # auto | float16 | float32
    batch_size: int = 8
    max_input_characters: int = 4000
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    model_cache_dir: Optional[str] = None

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

    def effective_device(self) -> str:
        """Return the device string to use, e.g. 'cuda:0'."""
        if self.device == "cuda":
            return f"cuda:{self.cuda_device}"
        return "cpu"


@dataclass
class GlossaryEntry:
    """A chapter terminology mapping (terminology memory).

    Attributes:
        source: the source-language term (e.g. a Chinese product name).
        target: the exact target-language term to restore after inference.
        exact: boundary policy:
            - True (default): whole-occurrence match only — the term must
              not be embedded in a latin word (bounded by non-latin
              alphanumerics/underscore or text edges). CJK ideograph
              neighbors are accepted: Chinese text has no spaces, so a term
              like 充电器 naturally sits adjacent to other ideographs
              ('使用充电器。'). This prevents 'cat' matching inside
              'catalog' while keeping the glossary usable for Chinese.
            - False: explicit opt-in that permits matches inside latin
              words (documented trade-off: may split words the user did not
              intend). Use only when the term is a deliberate prefix/suffix.

    Glossary terms are replaced with protected placeholders BEFORE model
    inference and restored to ``target`` afterwards, so the same configured
    term maps to the same target throughout the entire chapter (consistent
    by construction) and the model can never paraphrase it.
    """

    source: str
    target: str
    exact: bool = True

    def __post_init__(self) -> None:
        if not self.source or not self.source.strip():
            raise ValueError("glossary source term must be non-empty")
        if not self.target or not self.target.strip():
            raise ValueError("glossary target term must be non-empty")
        if self.source == self.target:
            raise ValueError(
                f"glossary source and target must differ: {self.source!r}"
            )


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
      M2M100 (its generate() has no reliable context API). This setting is
      explicitly UNSUPPORTED: it MUST stay 0 and any non-zero value raises
      a configuration error. Segment adjacency (prev/next segment ids) is a
      diagnostics-only record; it is NOT context and is never sent to the
      model.
    - glossary: chapter terminology memory. Each entry maps a source term to
      an exact target term; terms are protected before inference and restored
      consistently across every segment. See GlossaryEntry.
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
    - concurrency: bounds concurrent translations (GPU holds one model).
    """
    enabled: bool = True
    max_chapter_characters: int = 100_000
    max_segment_tokens: int = 450
    max_target_tokens: int = 400
    context_window_tokens: int = 0
    glossary: tuple = ()  # tuple[GlossaryEntry, ...] — terminology memory
    preserve_patterns: tuple = ()  # tuple[str, ...] — regexes (model formats)
    translatable_attributes: tuple = ()  # e.g. ("alt", "title", "aria-label")
    excluded_tags: tuple = ("script", "style", "code", "pre")
    excluded_classes: tuple = ("notranslate",)
    segment_warning_seconds: float = 60.0
    max_total_seconds: float = 600.0
    max_retries_per_segment: int = 1
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
                "implemented for M2M100 (no reliable context API); this "
                "setting is explicitly unsupported"
            )
        if self.segment_warning_seconds <= 0:
            raise ValueError("segment_warning_seconds must be > 0")
        if self.max_total_seconds <= 0:
            raise ValueError("max_total_seconds must be > 0")
        if self.max_retries_per_segment < 0:
            raise ValueError("max_retries_per_segment must be >= 0")
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
        # --- glossary validation ---
        entries = list(self.glossary)
        for e in entries:
            if not isinstance(e, GlossaryEntry):
                raise ValueError(
                    "glossary entries must be GlossaryEntry instances"
                )
        sources = [e.source for e in entries]
        if len(sources) != len(set(sources)):
            raise ValueError("glossary contains duplicate source terms")
        # Ambiguity guard: with a shared boundary, one term must not be a
        # substring of another (replacement order would be ambiguous).
        for a in entries:
            for b in entries:
                if a is b:
                    continue
                if a.source in b.source or b.source in a.source:
                    raise ValueError(
                        f"glossary terms overlap: {a.source!r} vs {b.source!r}"
                    )
