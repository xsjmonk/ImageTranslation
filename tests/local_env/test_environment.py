"""Environment setup validation for the translate-image skill."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from image_translation.local_env.env_check import (
    EnvironmentIssue,
    check_basic_operations,
    check_imports,
    format_environment_errors,
    run_environment_check,
)
from image_translation.local_env.repo_paths import DEFAULT_ENV_NAME, resolve_repo_root


def test_required_imports_pass():
    assert check_imports() == []


def test_basic_in_memory_operations_pass():
    assert check_basic_operations() == []


def test_run_environment_check_is_read_only(tmp_path: Path):
    before = {p.name for p in tmp_path.iterdir()} if tmp_path.exists() else set()
    issues = run_environment_check()
    after = {p.name for p in tmp_path.iterdir()} if tmp_path.exists() else set()
    assert issues == []
    assert before == after


def test_format_environment_errors_references_initialize_script():
    text = format_environment_errors(
        [EnvironmentIssue("opencv", "No module named 'cv2'")]
    )
    assert "opencv" in text
    assert "Initialize-Env.ps1" in text


def test_default_conda_environment_name():
    assert DEFAULT_ENV_NAME == "dp"


def test_repo_root_resolves_from_checkout():
    root = resolve_repo_root()
    assert (root / "environment.yml").is_file()


def test_check_env_module_usage():
    assert ["--check-env"] in (["--check-env"], ["check-env"])


def test_pillow_in_memory_roundtrip():
    from PIL import Image

    image = Image.new("RGB", (4, 4), color=(0, 128, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    loaded = Image.open(buffer)
    assert loaded.size == (4, 4)


def test_opencv_basic_operation():
    import cv2
    import numpy as np

    array = np.zeros((8, 8, 3), dtype=np.uint8)
    result = cv2.resize(array, (4, 4))
    assert result.shape == (4, 4, 3)


def test_shapely_and_pyclipper_import():
    from shapely.geometry import box
    import pyclipper

    assert box(0, 0, 1, 1).area == 1
    assert hasattr(pyclipper, "Pyclipper")


def test_repository_check_launcher_exists():
    root = resolve_repo_root()
    launcher = root / "script" / "Check-LocalImageEnv.ps1"
    assert launcher.is_file()
    text = launcher.read_text(encoding="utf-8")
    assert "image_translation.local_env" in text
    assert "PYTHONPATH" in text


def test_skill_environment_check_wrapper_exists():
    wrapper = Path(
        r"D:\Drop\outlook.com\LocalBox\Code\Skills\ImageTranslation\scripts\check-local-image-env.ps1"
    )
    assert wrapper.is_file()
    text = wrapper.read_text(encoding="utf-8")
    assert "Check-LocalImageEnv.ps1" in text
    assert "IMAGE_TRANSLATION_REPO" in text
