"""Tests for local manifest environment and launcher resolution."""

from __future__ import annotations

from pathlib import Path

from image_translation.local_manifest.env_check import (
    EnvironmentIssue,
    check_local_image_environment,
    format_environment_errors,
)
from image_translation.local_manifest.paths import (
    DEFAULT_ENV_NAME,
    resolve_repo_root,
    resolve_src_root,
)


def test_check_local_image_environment_passes_in_dp():
    issues = check_local_image_environment()
    assert issues == []


def test_format_environment_errors_lists_packages():
    text = format_environment_errors(
        [EnvironmentIssue("opencv", "No module named 'cv2'")]
    )
    assert "opencv" in text
    assert "Initialize-Env.ps1" in text


def test_resolve_repo_root_from_this_checkout():
    root = resolve_repo_root()
    assert (root / "environment.yml").is_file()
    assert (root / "src" / "image_translation").is_dir()


def test_resolve_src_root():
    root = resolve_repo_root()
    src = resolve_src_root(root)
    assert src == root / "src"
    assert (src / "image_translation" / "local_manifest").is_dir()


def test_default_conda_environment_name():
    assert DEFAULT_ENV_NAME == "dp"


def test_process_image_manifest_launcher_exists():
    root = resolve_repo_root()
    launcher = root / "script" / "Process-ImageManifest.ps1"
    assert launcher.is_file()
    text = launcher.read_text(encoding="utf-8")
    assert "IMAGE_TRANSLATION_REPO" in text
    assert "image_translation.local_manifest" in text


def test_skill_launcher_forwards_to_repository():
    skill_launcher = Path(
        r"D:\Drop\outlook.com\LocalBox\Code\Skills\ImageTranslation\scripts\process-image-manifest.ps1"
    )
    assert skill_launcher.is_file()
    text = skill_launcher.read_text(encoding="utf-8")
    assert "Process-ImageManifest.ps1" in text
    assert "IMAGE_TRANSLATION_REPO" in text
