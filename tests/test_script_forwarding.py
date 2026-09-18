"""Verify skill wrapper scripts forward output folder arguments."""

from pathlib import Path


def test_powershell_script_contains_output_forwarding():
    script = Path(
        r"d:\Drop\outlook.com\LocalBox\Code\Skills\ImageTranslation\scripts\translate-image.ps1"
    )
    text = script.read_text(encoding="utf-8")
    assert "OutputFolder" in text
    assert '"-o", $OutputFolder' in text


def test_shell_script_contains_output_forwarding():
    script = Path(
        r"d:\Drop\outlook.com\LocalBox\Code\Skills\ImageTranslation\scripts\translate-image.sh"
    )
    text = script.read_text(encoding="utf-8")
    assert "OUTPUT_FOLDER" in text
    assert '"-o"' in text or "-o" in text
