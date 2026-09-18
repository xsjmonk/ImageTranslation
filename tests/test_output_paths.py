"""Tests for output path resolution."""

from pathlib import Path

import pytest

from image_translation.config.models import AppConfig, OutputConfig
from image_translation.input.models import InputType
from image_translation.utilities.output_paths import (
    OutputPathError,
    localized_output_basename,
    preserved_original_basename,
    resolve_output_file_path,
    resolve_output_root,
    resolve_preserved_original_path,
    validate_input_output_paths,
)


def test_resolve_output_root_folder_suffix(tmp_path: Path):
    folder = tmp_path / "photos"
    folder.mkdir()
    config = AppConfig()
    root = resolve_output_root(folder, InputType.FOLDER, config)
    assert root == tmp_path / "photos_processed"


def test_resolve_output_root_cli_override(tmp_path: Path):
    folder = tmp_path / "photos"
    folder.mkdir()
    custom = tmp_path / "out"
    root = resolve_output_root(folder, InputType.FOLDER, AppConfig(), custom)
    assert root == custom.resolve()


def test_resolve_output_root_config_directory(tmp_path: Path):
    folder = tmp_path / "photos"
    folder.mkdir()
    config = AppConfig(output=OutputConfig(directory="localized"))
    root = resolve_output_root(folder, InputType.FOLDER, config)
    assert root == (folder / "localized").resolve()


def test_preserve_filename_flag(tmp_path: Path):
    source = tmp_path / "a.jpg"
    out = resolve_output_file_path(
        source,
        tmp_path / "out",
        tmp_path,
        OutputConfig(preserve_filename=False),
    )
    assert out.name == "a_en.jpg"


def test_recursive_relative_output_path(tmp_path: Path):
    src_root = tmp_path / "in"
    sub = src_root / "sub"
    sub.mkdir(parents=True)
    source = sub / "a.jpg"
    source.write_text("x")
    out_root = tmp_path / "out"
    path = resolve_output_file_path(
        source, out_root, src_root, AppConfig().output
    )
    assert path == out_root / "sub" / "a.jpg"


def test_preserved_original_basename_rules():
    assert preserved_original_basename("01.jpg") == "original_01.jpg"
    assert preserved_original_basename("original_01.jpg") == "original_01.jpg"


def test_localized_basename_strips_original_prefix():
    assert localized_output_basename("01.jpg") == "01.jpg"
    assert localized_output_basename("original_01.jpg") == "01.jpg"


def test_resolve_preserved_original_path(tmp_path: Path):
    source = tmp_path / "product.png"
    preserved = resolve_preserved_original_path(source, tmp_path / "out", None)
    assert preserved.name == "original_product.png"


def test_validate_blocks_same_path(tmp_path: Path):
    folder = tmp_path / "same"
    folder.mkdir()
    config = AppConfig()
    with pytest.raises(OutputPathError):
        validate_input_output_paths(folder, folder, config)
