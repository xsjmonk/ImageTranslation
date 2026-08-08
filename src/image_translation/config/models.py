"""Configuration models using pydantic for validation."""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class GeneralConfig(BaseModel):
    continue_on_error: bool = True


class InputConfig(BaseModel):
    recursive: bool = False
    extensions: List[str] = Field(
        default_factory=lambda: [
            ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
        ]
    )

    @field_validator("extensions")
    @classmethod
    def normalize_extensions(cls, v: List[str]) -> List[str]:
        result: List[str] = []
        for ext in v:
            ext = ext.strip()
            if not ext.startswith("."):
                ext = "." + ext
            result.append(ext.lower())
        return result


class OcrConfig(BaseModel):
    enabled: bool = True
    engine: str = "paddleocr"
    source_language: str = "zh"
    min_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    detect_rotation: bool = True

    @field_validator("min_confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"ocr.min_confidence must be between 0 and 1, got {v}")
        return v


class TranslationConfig(BaseModel):
    enabled: bool = True
    source_language: str = "zh-CN"
    target_language: str = "en-US"
    preserve_already_target_language: bool = True
    default_action: str = "translate"
    preserve_terms: List[str] = Field(default_factory=list)
    preserve_patterns: List[str] = Field(default_factory=list)
    # Compiled patterns (populated during validation)
    _compiled_patterns: List[re.Pattern] = []

    @field_validator("preserve_patterns")
    @classmethod
    def validate_patterns(cls, v: List[str]) -> List[str]:
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e
        return v

    def compiled_patterns(self) -> List[re.Pattern]:
        if not self._compiled_patterns:
            self._compiled_patterns = [re.compile(p) for p in self.preserve_patterns]
        return self._compiled_patterns

    @field_validator("default_action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        allowed = {"translate", "preserve", "remove", "review"}
        if v not in allowed:
            raise ValueError(f"translation.default_action must be one of {allowed}, got '{v}'")
        return v


class ImagingConfig(BaseModel):
    enabled: bool = True
    processor: str = "hybrid"
    mask_expansion_pixels: int = Field(default=3, ge=0)
    preserve_original_dimensions: bool = True

    @field_validator("processor")
    @classmethod
    def validate_processor(cls, v: str) -> str:
        allowed = {"hybrid", "opencv", "neural"}
        if v not in allowed:
            raise ValueError(f"imaging.processor must be one of {allowed}, got '{v}'")
        return v


class RevisionConfig(BaseModel):
    enabled: bool = True
    preserve_rotation: bool = True
    use_source_polygon: bool = True
    allow_multiline: bool = True
    minimum_font_size: int = Field(default=12, gt=0)

    @field_validator("minimum_font_size")
    @classmethod
    def font_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"revision.minimum_font_size must be > 0, got {v}")
        return v


class OutputConfig(BaseModel):
    suffix: str = "_processed"
    preserve_filename: bool = True
    overwrite_existing: bool = False
    save_metadata: bool = True
    save_masks: bool = True
    save_cleaned_images: bool = True

    @field_validator("suffix")
    @classmethod
    def suffix_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("output.suffix must not be empty")
        return v


class LoggingConfig(BaseModel):
    level: str = "INFO"

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"logging.level must be one of {allowed}, got '{v}'")
        return upper


class AppConfig(BaseModel):
    """Root configuration model."""
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    imaging: ImagingConfig = Field(default_factory=ImagingConfig)
    revision: RevisionConfig = Field(default_factory=RevisionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
