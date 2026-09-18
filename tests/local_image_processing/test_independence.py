"""Verify LocalImageProcessing does not depend on image_translation."""

from __future__ import annotations

from pathlib import Path


def test_no_image_translation_imports_in_module_sources():
    root = Path(__file__).resolve().parents[2] / "LocalImageProcessing"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "image_translation" in text:
            offenders.append(str(path.relative_to(root)))
    assert offenders == []
