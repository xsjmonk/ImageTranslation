"""Conservative text preprocessing for translation input."""

from __future__ import annotations

import unicodedata


def preprocess(text: str, max_characters: int = 4000) -> str:
    """Normalize text for translation without altering meaning.

    Allowed:
    - Normalize Unicode whitespace
    - Normalize line breaks
    - Trim outer whitespace

    Not allowed:
    - Rewrite Chinese characters
    - Remove punctuation indiscriminately
    - Convert terminology

    Args:
        text: Raw input text.
        max_characters: Maximum allowed length.

    Returns:
        Cleaned text.

    Raises:
        ValueError: If text is empty, whitespace-only, or exceeds max_characters.
    """
    if text is None:
        raise ValueError("Input text must not be None")

    if not isinstance(text, str):
        raise ValueError(f"Input must be a string, got {type(text).__name__}")

    # Normalize Unicode whitespace (non-breaking spaces etc.)
    cleaned = unicodedata.normalize("NFKC", text)

    # Normalize line breaks
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple consecutive newlines
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")

    # Trim outer whitespace
    cleaned = cleaned.strip()

    if not cleaned:
        raise ValueError("Input text must not be empty or whitespace-only")

    if len(cleaned) > max_characters:
        raise ValueError(
            f"Input text exceeds maximum length of {max_characters} characters "
            f"(got {len(cleaned)})"
        )

    return cleaned
