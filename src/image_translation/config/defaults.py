"""Built-in default configuration values.

These are used when no config.json is present.
"""

from __future__ import annotations

from .models import (
    AppConfig,
    GeneralConfig,
    ImagingConfig,
    InputConfig,
    LoggingConfig,
    OcrConfig,
    OutputConfig,
    RevisionConfig,
    TranslationConfig,
)


def build_default_config() -> AppConfig:
    """Return an AppConfig populated with sensible defaults."""
    return AppConfig(
        general=GeneralConfig(continue_on_error=True),
        input=InputConfig(
            recursive=False,
            extensions=[".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"],
        ),
        ocr=OcrConfig(
            enabled=True,
            engine="paddleocr",
            source_language="zh",
            min_confidence=0.65,
            detect_rotation=True,
        ),
        translation=TranslationConfig(
            enabled=True,
            source_language="zh-CN",
            target_language="en-US",
            preserve_already_target_language=True,
            default_action="translate",
            preserve_terms=[],
            preserve_patterns=[],
        ),
        imaging=ImagingConfig(
            enabled=True,
            processor="hybrid",
            mask_expansion_pixels=3,
            preserve_original_dimensions=True,
        ),
        revision=RevisionConfig(
            enabled=True,
            preserve_rotation=True,
            use_source_polygon=True,
            allow_multiline=True,
            minimum_font_size=12,
        ),
        output=OutputConfig(
            suffix="_processed",
            preserve_filename=True,
            overwrite_existing=False,
            save_metadata=True,
            save_masks=True,
            save_cleaned_images=True,
        ),
        logging=LoggingConfig(level="INFO"),
    )
