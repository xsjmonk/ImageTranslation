"""Shared translation exception hierarchy — reusable across all translation clients."""

from __future__ import annotations


class TranslationError(Exception):
    """Base exception for all translation failures."""


class TranslationInputError(TranslationError):
    """Invalid translation input (empty, whitespace, too long, wrong type)."""


class TranslationConfigurationError(TranslationError):
    """Invalid translation/server configuration."""


class TranslationDeviceError(TranslationError):
    """CUDA/device unavailable or invalid device selection."""


class TranslationModelLoadError(TranslationError):
    """Failed to load the translation model or tokenizer."""


class StructuredTranslationError(TranslationError):
    """Structured (HTML) translation failed: parsing, protection, chunking,
    reconstruction validation, or retry-exhausted failures."""


class BatchItemError(StructuredTranslationError):
    """One or more batch result items are invalid (non-string translated
    text, missing attribute, or None).

    Carries the indices of the bad items and the validated outputs of the
    GOOD items so the caller can recover only the affected inputs and keep
    already-successful neighbors without re-sending them.
    """

    def __init__(
        self,
        message: str,
        bad_indices,
        valid_outputs,
    ) -> None:
        super().__init__(message)
        self.bad_indices = list(bad_indices)
        self.valid_outputs = dict(valid_outputs)  # index -> validated string
