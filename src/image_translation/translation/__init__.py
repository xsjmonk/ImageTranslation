"""Translation package — reusable across ImageTranslation and translation_server."""

from .base import Translator
from .chapter_chunking import Segment, collect_blocks, segment_blocks
from .classifier import classify_regions
from .config import (
    TranslationConfig,
    GenerationConfig,
    QualityConfig,
    StructuredConfig,
    GlossaryEntry,
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
from .glossary import GlossaryTranslator, load_glossary_file
from .html_document import HTMLDocument
from .html_protection import ProtectionMap
from .language_segments import classify, LanguageKind, protect_identifiers
from .m2m100_translator import M2M100Translator
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
    "GlossaryEntry",
    # Models
    "TranslationRequest",
    "TranslationResult",
    "TranslationRuntimeInfo",
    # Factory
    "create_translator",
    "GlossaryTranslator",
    "load_glossary_file",
    # Implementations
    "M2M100Translator",
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
