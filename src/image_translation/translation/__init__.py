"""Translation package — reusable across ImageTranslation and translation_server."""

from .base import Translator
from .classifier import classify_regions
from .config import TranslationConfig, GenerationConfig
from .factory import create_translator
from .m2m100_translator import (
    M2M100Translator,
    TranslationConfigurationError,
    TranslationDeviceError,
    TranslationError,
    TranslationInputError,
    TranslationModelLoadError,
)
from .models import TranslationRequest, TranslationResult, TranslationRuntimeInfo
from .text_utils import preprocess
from .translator import NoopTranslator

__all__ = [
    # Base
    "Translator",
    # Config
    "TranslationConfig",
    "GenerationConfig",
    # Models
    "TranslationRequest",
    "TranslationResult",
    "TranslationRuntimeInfo",
    # Factory
    "create_translator",
    # Implementations
    "M2M100Translator",
    "NoopTranslator",
    # Exceptions
    "TranslationError",
    "TranslationConfigurationError",
    "TranslationDeviceError",
    "TranslationInputError",
    "TranslationModelLoadError",
    # Utilities
    "classify_regions",
    "preprocess",
]
