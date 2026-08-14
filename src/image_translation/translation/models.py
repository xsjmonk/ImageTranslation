"""Translation domain models — reusable, no framework dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranslationRequest:
    """Input to the translation engine."""
    text: str
    source_language: str = "zh"
    target_language: str = "en"


@dataclass
class TranslationResult:
    """Structured translation output.

    Compatible with the existing ImageTranslation pipeline (compact_text, literal_text).
    """
    source_text: str
    translated_text: str = ""
    source_language: str = "zh"
    target_language: str = "en"
    model_name: str = ""
    device: str = ""
    # Legacy fields for ImageTranslation pipeline compatibility
    compact_text: str = ""
    literal_text: str = ""


@dataclass
class TranslationRuntimeInfo:
    """Metadata about the loaded translation engine."""
    model_name: str = ""
    model_revision: str = ""
    device: str = ""
    precision: str = ""
    cuda_available: bool = False
    gpu_name: str = ""
    ready: bool = False
    # Model cache diagnostics (no secrets)
    cache_dir: str = ""          # configured HF cache root ("" = HF default)
    snapshot_path: str = ""      # resolved snapshot directory actually loaded
    cache_status: str = ""       # "cache_hit" | "download" | "none"
    local_files_only: bool = False
    offline: bool = False        # effective offline (local_files_only OR
                                 # downloads disabled)


@dataclass(frozen=True)
class ResolvedModel:
    """Authoritative model-resolution result, created ONCE during loading
    and retained by the translator. HTML token measurement, tokenizer
    loading, and model loading all use this same resolution — nothing
    rediscovers the model independently."""
    snapshot_path: str
    model_name: str
    revision: str
    cache_dir: str               # "" = HF default cache
    cache_status: str            # "cache_hit" | "download"
    offline: bool                # effective offline flag
