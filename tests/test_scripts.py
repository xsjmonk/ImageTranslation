"""Deterministic checks for the supported PowerShell entry points."""

from __future__ import annotations

import json
import subprocess
import sys
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


def test_translation_launcher_uses_python_config_summary():
    text = (SCRIPT_DIR / "Start-TranslationServer.ps1").read_text(
        encoding="utf-8"
    )
    lowered = text.lower()
    assert "m2m100" not in lowered
    assert "get-content -raw -path $configpath" not in lowered
    assert "--print-config-summary" in lowered
    assert "envs\\dp\\python.exe" in lowered
    assert "translation_server" in lowered
    assert "'-c', $configpath" in lowered
    assert "$env:pythonpath = join-path $reporoot 'src'" in lowered
    assert "could not load config summary" in lowered


def test_startup_script_config_summary_matches_loader(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    cfg_path = tmp_path / "commented.jsonc"
    cfg_path.write_text(
        """
        {
          // Hy-MT2 opt-in profile
          "server": { "model_cache_dir": "%s" },
          "translation": {
            "active_model": "hymt2",
            "models": {
              "nllb": {
                "backend": "current",
                "model_name": "facebook/nllb-200-distilled-600M",
                "model_family": "nllb",
                "model_revision": "main"
              },
              "hymt2": {
                "backend": "hymt2",
                "model_name": "tencent/Hy-MT2-1.8B-FP8",
                "model_family": "hymt2",
                "model_revision": "main"
              }
            }
          }
        }
        """ % str(cache).replace("\\", "/"),
        encoding="utf-8",
    )
    env = {"PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "translation_server",
            "--print-config-summary",
            "-c",
            str(cfg_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        env={**dict(**__import__("os").environ), **env},
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["active_model"] == "hymt2"
    assert summary["backend"] == "hymt2"
    assert summary["model"] == "tencent/Hy-MT2-1.8B-FP8"


def test_environment_initializer_has_no_runtime_side_effects():
    text = (SCRIPT_DIR / "Initialize-Env.ps1").read_text(
        encoding="utf-8"
    ).lower()
    assert "environment.yml" in text
    assert "env create" in text
    assert "env update" in text
    assert "translation_server" not in text
    assert "snapshot_download" not in text
