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
