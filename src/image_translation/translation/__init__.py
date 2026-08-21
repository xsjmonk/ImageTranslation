"""Translation package — reusable across ImageTranslation and translation_server."""

from .base import Translator
from .chapter_chunking import Segment, collect_blocks, segment_blocks
from .classifier import classify_regions
from .config import (
    TranslationConfig,
    GenerationConfig,
    QualityConfig,
    StructuredConfig,
    TranslationStyle,
    ResolvedGenerationPolicy,
    resolve_translation_style,
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
from .html_document import HTMLDocument, compare_document_structure
from .html_protection import ProtectionMap
from .language_segments import classify, LanguageKind, protect_identifiers
from .seq2seq_translator import Seq2SeqTranslator
from .model_adapters import ModelFamilyAdapter, create_model_family_adapter
from .models import TranslationRequest, TranslationResult, TranslationRuntimeInfo
from .reconstruction import rebuild_document
from .structured_translation import (
    StructuredTranslationResult,
    StructuredTranslator,
    compare_structured_results,
    translate_html,
)
from .text_utils import preprocess
from .phrase_policy import (
    PhraseValidationResult,
    content_token_count,
    validate_phrase_output,
)
from .translator import NoopTranslator

__all__ = [
    # Base
    "Translator",
    # Config
    "TranslationConfig",
    "GenerationConfig",
    "QualityConfig",
    "StructuredConfig",
    "TranslationStyle",
    "ResolvedGenerationPolicy",
    "resolve_translation_style",
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
    "PhraseValidationResult",
    "content_token_count",
    "validate_phrase_output",
    # Structured (HTML-aware) translation
    "HTMLDocument",
    "compare_document_structure",
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
    "compare_structured_results",
    "translate_html",
]
