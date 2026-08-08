"""Tests for folder enumeration – images, non-images, ordering, recursive."""

from __future__ import annotations

from pathlib import Path

from image_translation.utilities.folders import enumerate_images


EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class TestEnumerateImages:
    def test_ignores_non_images(self, tmp_path: Path):
        (tmp_path / "a.jpg").write_text("")
        (tmp_path / "b.txt").write_text("")
        (tmp_path / "c.png").write_text("")
        (tmp_path / "readme.md").write_text("")

        result = enumerate_images(tmp_path, EXTS)
        names = [p.name for p in result]
        assert names == ["a.jpg", "c.png"]

    def test_deterministic_ordering(self, tmp_path: Path):
        # Create files in reverse order
        for name in ["z.jpg", "a.png", "m.webp"]:
            (tmp_path / name).write_text("")

        result = enumerate_images(tmp_path, EXTS)
        names = [p.name for p in result]
        assert names == ["a.png", "m.webp", "z.jpg"]

    def test_natural_sorting(self, tmp_path: Path):
        for name in ["10.jpg", "2.jpg", "1.jpg", "20.jpg"]:
            (tmp_path / name).write_text("")

        result = enumerate_images(tmp_path, EXTS)
        names = [p.name for p in result]
        assert names == ["1.jpg", "2.jpg", "10.jpg", "20.jpg"]

    def test_recursive_excludes_processed(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        processed = tmp_path / "sub_processed"
        processed.mkdir()

        (tmp_path / "a.jpg").write_text("")
        (sub / "b.jpg").write_text("")
        (processed / "c.jpg").write_text("")

        result = enumerate_images(tmp_path, EXTS, recursive=True)
        names = [p.name for p in result]
        assert "c.jpg" not in names
        assert "a.jpg" in names
        assert "b.jpg" in names

    def test_empty_folder(self, tmp_path: Path):
        result = enumerate_images(tmp_path, EXTS)
        assert result == []
