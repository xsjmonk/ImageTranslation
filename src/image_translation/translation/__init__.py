"""Translation package — reusable across ImageTranslation and translation_server."""

from .base import Translator
from .chapter_chunking import Segment, collect_blocks, segment_blocks
from .classifier import classify_regions
from .config import (
    TranslationConfig,
    GenerationConfig,
    QualityConfig,
    StructuredConfig,
)
from .exceptions import (
    StructuredTranslationError,
    TranslationConfigurationError,
    TranslationDeviceError,
    TranslationError,
    TranslationInputError,
    TranslationModelLoadError,
    TranslationQualityError,
)
from .factory import create_translator
from .html_document import HTMLDocument
from .html_protection import ProtectionMap
from .language_segments import classify, LanguageKind, protect_identifiers
from .seq2seq_translator import Seq2SeqTranslator
from .model_adapters import ModelFamilyAdapter, create_model_family_adapter
from .models import TranslationRequest, TranslationResult, TranslationRuntimeInfo
from .reconstruction import rebuild_document
from .structured_translation import (
    StructuredTranslationResult,
    StructuredTranslator,
    translate_html,
)
from .text_utils import preprocess
from .translator import NoopTranslator

__all__ = [
    # Base
    "Translator",
    # Config
    "TranslationConfig",
    "GenerationConfig",
    "QualityConfig",
    "StructuredConfig",
    # Models
    "TranslationRequest",
    "TranslationResult",
    "TranslationRuntimeInfo",
    # Factory
    "create_translator",
    # Implementations
    "Seq2SeqTranslator",
    "ModelFamilyAdapter",
    "create_model_family_adapter",
    "NoopTranslator",
    # Exceptions
    "TranslationError",
    "TranslationConfigurationError",
    "TranslationDeviceError",
    "TranslationInputError",
    "TranslationModelLoadError",
    "TranslationQualityError",
    "StructuredTranslationError",
    # Utilities
    "classify_regions",
    "preprocess",
    # Structured (HTML-aware) translation
    "HTMLDocument",
    "ProtectionMap",
    "LanguageKind",
    "classify",
    "protect_identifiers",
    "Segment",
    "collect_blocks",
    "segment_blocks",
    "rebuild_document",
    "StructuredTranslator",
    "StructuredTranslationResult",
    "translate_html",
]
