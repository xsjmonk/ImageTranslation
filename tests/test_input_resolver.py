"""Tests for input resolver – folder, single image, errors, extensions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from image_translation.input.input_resolver import (
    InputError,
    SUPPORTED_EXTENSIONS,
    resolve_input,
)
from image_translation.input.models import InputType


def _make_parsed(input_path: str, config_path: str | None = None):
    """Create a mock parsed namespace."""
    ns = argparse.Namespace()
    ns.input_path = Path(input_path)
    ns.config_path = Path(config_path) if config_path else None
    return ns


class TestResolveFolder:
    def test_existing_folder(self, tmp_path: Path):
        folder = tmp_path / "photos"
        folder.mkdir()
        parsed = _make_parsed(str(folder))
        result = resolve_input(parsed)

        assert result.input_type == InputType.FOLDER
        assert result.input_folder == folder.resolve()
        assert result.single_image_path is None
        assert result.output_folder == tmp_path / "photos_processed"

    def test_nonexistent_path(self, tmp_path: Path):
        parsed = _make_parsed(str(tmp_path / "nope"))
        with pytest.raises(InputError, match="does not exist"):
            resolve_input(parsed)


class TestResolveSingleImage:
    def test_jpg_file(self, tmp_path: Path):
        img = tmp_path / "test.jpg"
        img.write_text("fake image")
        parsed = _make_parsed(str(img))
        result = resolve_input(parsed)

        assert result.input_type == InputType.SINGLE_IMAGE
        assert result.single_image_path == img.resolve()
        # output_folder = <parent>_processed/<filename>
        assert result.output_folder.parent.name == tmp_path.name + "_processed"
        assert result.output_folder.name == "test.jpg"

    def test_unsupported_extension(self, tmp_path: Path):
        img = tmp_path / "test.txt"
        img.write_text("not an image")
        parsed = _make_parsed(str(img))
        with pytest.raises(InputError, match="Unsupported file type"):
            resolve_input(parsed)

    def test_case_insensitive_extension(self, tmp_path: Path):
        img = tmp_path / "test.JPG"
        img.write_text("fake")
        parsed = _make_parsed(str(img))
        result = resolve_input(parsed)
        assert result.input_type == InputType.SINGLE_IMAGE

    def test_png_file(self, tmp_path: Path):
        img = tmp_path / "icon.png"
        img.write_text("fake png")
        parsed = _make_parsed(str(img))
        result = resolve_input(parsed)
        assert result.input_type == InputType.SINGLE_IMAGE


class TestOutputDerivation:
    def test_folder_output(self, tmp_path: Path):
        folder = tmp_path / "my_images"
        folder.mkdir()
        parsed = _make_parsed(str(folder))
        result = resolve_input(parsed)
        assert result.output_folder == tmp_path / "my_images_processed"

    def test_single_image_output(self, tmp_path: Path):
        parent = tmp_path / "my_images"
        parent.mkdir()
        img = parent / "01.jpg"
        img.write_text("fake")
        parsed = _make_parsed(str(img))
        result = resolve_input(parsed)
        assert result.output_folder == tmp_path / "my_images_processed" / "01.jpg"
