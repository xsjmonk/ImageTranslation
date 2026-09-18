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
    engine: str = "gpu"
    gpu_config_path: Optional[str] = None
    server_url: str = "http://127.0.0.1:8091"
    server_timeout_seconds: float = Field(default=120.0, gt=0)
    style: str = "phrase"
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

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: str) -> str:
        allowed = {"noop", "gpu", "server"}
        lower = v.lower()
        if lower not in allowed:
            raise ValueError(f"translation.engine must be one of {allowed}, got '{v}'")
        return lower

    @field_validator("style")
    @classmethod
    def validate_style(cls, v: str) -> str:
        allowed = {"sentence", "phrase"}
        lower = v.lower()
        if lower not in allowed:
            raise ValueError(f"translation.style must be one of {allowed}, got '{v}'")
        return lower


class ImagingConfig(BaseModel):
    enabled: bool = True
    processor: str = "hybrid"
    mask_expansion_pixels: int = Field(default=3, ge=0)
    preserve_original_dimensions: bool = True

    @field_validator("processor")
    @classmethod
    def validate_processor(cls, v: str) -> str:
        allowed = {"hybrid", "opencv", "neural", "enhanced_opencv"}
        if v not in allowed:
            raise ValueError(f"imaging.processor must be one of {allowed}, got '{v}'")
        return v


class ClassificationConfig(BaseModel):
    """Conservative rules for logos, trademarks, and product-embedded text."""
    logo_max_chars: int = Field(default=8, ge=1)
    logo_aspect_ratio_min: float = Field(default=2.5, gt=0)
    product_embedded_patterns: List[str] = Field(default_factory=list)
    review_patterns: List[str] = Field(default_factory=list)
    force_translate_patterns: List[str] = Field(default_factory=list)
    identifier_patterns: List[str] = Field(
        default_factory=lambda: [
            r"^https?://",
            r"^[A-Z0-9][A-Z0-9\-_/\.]+$",
            r"^\d+(\.\d+)?\s*(mm|cm|m|kg|g|ml|l|°|℃|%)?$",
        ]
    )
    _compiled_product: List[re.Pattern] = []
    _compiled_review: List[re.Pattern] = []
    _compiled_force: List[re.Pattern] = []
    _compiled_identifier: List[re.Pattern] = []

    @field_validator("product_embedded_patterns", "review_patterns", "force_translate_patterns", "identifier_patterns")
    @classmethod
    def validate_regex_list(cls, v: List[str]) -> List[str]:
        for pattern in v:
            re.compile(pattern)
        return v

    def compiled_product_patterns(self) -> List[re.Pattern]:
        if not self._compiled_product:
            self._compiled_product = [re.compile(p) for p in self.product_embedded_patterns]
        return self._compiled_product

    def compiled_review_patterns(self) -> List[re.Pattern]:
        if not self._compiled_review:
            self._compiled_review = [re.compile(p) for p in self.review_patterns]
        return self._compiled_review

    def compiled_force_patterns(self) -> List[re.Pattern]:
        if not self._compiled_force:
            self._compiled_force = [re.compile(p) for p in self.force_translate_patterns]
        return self._compiled_force

    def compiled_identifier_patterns(self) -> List[re.Pattern]:
        if not self._compiled_identifier:
            self._compiled_identifier = [re.compile(p) for p in self.identifier_patterns]
        return self._compiled_identifier


class QcConfig(BaseModel):
    preserved_region_tolerance: float = Field(default=12.0, ge=0)
    clip_tolerance_pixels: int = Field(default=2, ge=0)
    enable_final_pixel_checks: bool = True
    enable_final_ocr: bool = True
    final_ocr_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    final_ocr_cjk_coverage_threshold: float = Field(default=0.12, ge=0.0, le=1.0)
    final_ocr_overlap_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    rendered_text_min_alpha_coverage: float = Field(default=0.005, ge=0.0, le=1.0)
    inpaint_flatness_threshold: float = Field(default=4.0, ge=0.0)
    inpaint_seam_threshold: float = Field(default=18.0, ge=0.0)
    inpaint_spill_threshold: float = Field(default=8.0, ge=0.0)
    max_glyph_outside_polygon_ratio: float = Field(default=0.05, ge=0.0, le=1.0)


class RevisionConfig(BaseModel):
    enabled: bool = True
    preserve_rotation: bool = True
    use_source_polygon: bool = True
    allow_multiline: bool = True
    minimum_font_size: int = Field(default=12, gt=0)
    maximum_font_size: int = Field(default=48, gt=0)
    line_spacing: float = Field(default=1.2, gt=0)
    clip_tolerance_pixels: int = Field(default=2, ge=0)
    font_path: Optional[str] = None
    bold_font_path: Optional[str] = None

    @field_validator("minimum_font_size")
    @classmethod
    def font_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"revision.minimum_font_size must be > 0, got {v}")
        return v


class OutputConfig(BaseModel):
    suffix: str = "_processed"
    directory: Optional[str] = None
    preserve_filename: bool = True
    preserve_original: bool = True
    overwrite_existing: bool = False
    allow_same_as_input: bool = False
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
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    imaging: ImagingConfig = Field(default_factory=ImagingConfig)
    revision: RevisionConfig = Field(default_factory=RevisionConfig)
    quality_control: QcConfig = Field(default_factory=QcConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
