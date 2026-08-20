"""Deterministic checks for the supported PowerShell entry points."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "script"


def _powershell() -> str:
    for name in ("pwsh", "powershell"):
        result = subprocess.run(
            ["where.exe", name],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return name
    pytest.skip("PowerShell is unavailable")


@pytest.mark.parametrize(
    "script_name",
    ["Start-TranslationServer.ps1", "Initialize-Env.ps1"],
)
def test_power_shell_script_parses(script_name: str):
    shell = _powershell()
    path = SCRIPT_DIR / script_name
    command = (
        "$errors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{path}', [ref]$null, [ref]$errors) | Out-Null; "
        "if ($errors.Count -ne 0) { "
        "$errors | ForEach-Object { $_.ToString() }; exit 1 }"
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_translation_launcher_targets_nllb_and_existing_runtime():
    text = (SCRIPT_DIR / "Start-TranslationServer.ps1").read_text(
        encoding="utf-8"
    )
    lowered = text.lower()
    assert "m2m100" not in lowered
    assert "facebook/nllb-200-distilled-600m" in lowered
    assert "envs\\dp\\python.exe" in lowered
    assert "translation_server" in lowered
    assert "'-c', $configpath" in lowered
    assert "$env:pythonpath = join-path $reporoot 'src'" in lowered


def test_environment_initializer_has_no_runtime_side_effects():
    text = (SCRIPT_DIR / "Initialize-Env.ps1").read_text(
        encoding="utf-8"
    ).lower()
    assert "environment.yml" in text
    assert "env create" in text
    assert "env update" in text
    assert "translation_server" not in text
    assert "snapshot_download" not in text
