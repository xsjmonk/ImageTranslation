"""Map configured/API language codes to full language names for Hy-MT2 prompts."""

from __future__ import annotations

_LANGUAGE_NAMES = {
    "zho_hans": "Chinese",
    "zho-hans": "Chinese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh_cn": "Chinese",
    "chinese": "Chinese",
    "eng_latn": "English",
    "eng-latn": "English",
    "en": "English",
    "en-us": "English",
    "en_us": "English",
    "english": "English",
}


def resolve_language_name(language_code: str, configured_default: str = "") -> str:
    """Return a full language name suitable for Hy-MT2 instruction prompts."""
    for candidate in (language_code, configured_default):
        if not candidate:
            continue
        normalized = candidate.strip().lower().replace("_", "-")
        if normalized in _LANGUAGE_NAMES:
            return _LANGUAGE_NAMES[normalized]
        compact = normalized.replace("-", "_")
        if compact in _LANGUAGE_NAMES:
            return _LANGUAGE_NAMES[compact]
        if candidate.strip():
            return candidate.strip()
    raise ValueError(f"unsupported language code for Hy-MT2 prompt: {language_code!r}")
