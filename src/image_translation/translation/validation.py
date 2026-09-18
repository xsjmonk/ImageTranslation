"""Translation output validation for the image localization pipeline."""

from __future__ import annotations

import re
from typing import List

from ..config.models import TranslationConfig
from ..models.text_region import TextAction, TextRegion
from ..quality_control import QcIssue, QcResult, QcSeverity, _BRACKETED_SOURCE_RE, _CJK_RE

_MAX_EXPANSION_RATIO = 8.0


def validate_translations(
    regions: List[TextRegion],
    config: TranslationConfig,
) -> QcResult:
    """Validate translation quality: protected terms, expansion, placeholders."""
    issues: List[QcIssue] = []

    for region in regions:
        if region.action != TextAction.translate:
            continue

        source = region.source_text.strip()
        translation = region.translation or {}
        translated = (
            translation.get("translated_text")
            or translation.get("compact_text")
            or ""
        ).strip()

        if not translated:
            continue

        if _BRACKETED_SOURCE_RE.match(translated):
            issues.append(
                QcIssue(
                    severity=QcSeverity.error,
                    code="noop_translation",
                    message=f"No-op placeholder translation for: {source[:80]}",
                    region_id=region.id,
                )
            )

        if _CJK_RE.search(translated):
            issues.append(
                QcIssue(
                    severity=QcSeverity.error,
                    code="chinese_in_translation",
                    message=f"CJK in translation for: {source[:80]}",
                    region_id=region.id,
                )
            )

        for term in config.preserve_terms:
            if term and term.lower() in source.lower():
                if term.lower() not in translated.lower():
                    issues.append(
                        QcIssue(
                            severity=QcSeverity.error,
                            code="protected_term_lost",
                            message=(
                                f"Protected term '{term}' missing from translation "
                                f"of: {source[:80]}"
                            ),
                            region_id=region.id,
                        )
                    )

        if len(source) > 0:
            ratio = len(translated) / len(source)
            if ratio > _MAX_EXPANSION_RATIO:
                issues.append(
                    QcIssue(
                        severity=QcSeverity.warning,
                        code="translation_expansion",
                        message=(
                            f"Translation expansion ratio {ratio:.1f}x for: {source[:80]}"
                        ),
                        region_id=region.id,
                    )
                )

    passed = not any(i.severity == QcSeverity.error for i in issues)
    return QcResult(passed=passed, issues=issues)
