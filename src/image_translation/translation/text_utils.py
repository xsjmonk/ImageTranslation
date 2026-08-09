"""Conservative text preprocessing for translation input.

Preprocessing is intentionally minimal to preserve translation fidelity:
- normalize line endings
- strip outer whitespace

NFKC Unicode normalization is intentionally NOT applied: it can rewrite
compatibility characters and alter content that matters for product/OCR text.
"""

from __future__ import annotations

from .exceptions import TranslationInputError


def preprocess(text: str, max_characters: int = 4000) -> str:
    """Validate and minimally normalize text for translation.

    Args:
        text: Raw input text.
        max_characters: Maximum allowed length.

    Returns:
        Cleaned text.

    Raises:
        TranslationInputError: If text is None, empty/whitespace-only,
            not a string, or exceeds max_characters.
    """
    if text is None:
        raise TranslationInputError("Input text must not be None")

    if not isinstance(text, str):
        raise TranslationInputError(
            f"Input must be a string, got {type(text).__name__}"
        )

    # Normalize line breaks only
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse runs of blank lines (keep single newlines)
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")

    # Trim outer whitespace
    cleaned = cleaned.strip()

    if not cleaned:
        raise TranslationInputError("Input text must not be empty or whitespace-only")

    if len(cleaned) > max_characters:
        raise TranslationInputError(
            f"Input text exceeds maximum length of {max_characters} characters "
            f"(got {len(cleaned)})"
        )

    return cleaned
