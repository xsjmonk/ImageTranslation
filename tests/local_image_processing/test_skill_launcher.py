"""Skill launcher integration tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SKILL_ROOT = Path(r"D:\Drop\outlook.com\LocalBox\Code\Skills\ImageTranslation")
PYTHON = Path(sys.executable)


def _write_manifest_and_image(tmp_path: Path):
    import cv2
    import numpy as np

    source = tmp_path / "skill.jpg"
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :] = (30, 60, 90)
    cv2.rectangle(image, (20, 20), (100, 50), (255, 255, 255), -1)
    cv2.imwrite(str(source), image)

    output_root = tmp_path / "skill_processed"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "contract_version": "1.0",
                "source_path": str(source),
                "regions": [
                    {
                        "id": "text-1",
                        "polygon": [[10, 10], [110, 10], [110, 60], [10, 60]],
                        "action": "translate",
                        "source_text": "测试",
                        "translated_text": "OK",
                        "style": {"alignment": "center", "color_rgb": [0, 0, 0]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return source, manifest_path, output_root


def test_powershell_script_forwards_manifest_and_output():
    script = SKILL_ROOT / "scripts" / "translate-image.ps1"
    text = script.read_text(encoding="utf-8")
    assert "$Manifest" in text
    assert "OutputFolder" in text
    assert "'-o', $OutputFolder" in text or '"-o", $OutputFolder' in text
    assert "LocalImageProcessing" in text
    assert "IMAGE_TRANSLATION_REPO" in text


def test_shell_script_forwards_output_folder():
    script = SKILL_ROOT / "scripts" / "translate-image.sh"
    text = script.read_text(encoding="utf-8")
    assert "OUTPUT_FOLDER" in text
    assert '"-o"' in text or "-o" in text
    assert "LocalImageProcessing" in text


def test_powershell_launcher_end_to_end(tmp_path: Path):
    source, manifest_path, output_root = _write_manifest_and_image(tmp_path)
    script = SKILL_ROOT / "scripts" / "translate-image.ps1"
    env = os.environ.copy()
    env["IMAGE_TRANSLATION_REPO"] = str(REPO)
    env["PYTHONPATH"] = str(REPO)
    env["TMP"] = str(tmp_path)
    env["TEMP"] = str(tmp_path)
    env["TMPDIR"] = str(tmp_path)

    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Manifest",
            str(manifest_path),
            "-OutputFolder",
            str(output_root),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        cwd="C:/",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_root / "skill.jpg").is_file()
    assert (output_root / "original_skill.jpg").is_file()
    assert (output_root / "metadata" / "skill.json").is_file()
    assert source.read_bytes() == (output_root / "original_skill.jpg").read_bytes()


def test_python_module_from_different_cwd(tmp_path: Path):
    source, manifest_path, output_root = _write_manifest_and_image(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["TMP"] = str(tmp_path)
    env["TEMP"] = str(tmp_path)
    env["TMPDIR"] = str(tmp_path)

    result = subprocess.run(
        [
            str(PYTHON),
            "-m",
            "LocalImageProcessing",
            "process",
            "--manifest",
            str(manifest_path),
            "-o",
            str(output_root),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        cwd="C:/",
    )
    assert result.returncode == 0, result.stderr
    assert (output_root / "skill.jpg").is_file()
