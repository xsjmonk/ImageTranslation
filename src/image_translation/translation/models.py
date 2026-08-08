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
    device: str = ""
    precision: str = ""
    cuda_available: bool = False
    gpu_name: str = ""
    ready: bool = False
