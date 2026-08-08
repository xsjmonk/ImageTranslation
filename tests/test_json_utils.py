"""Tests for JSON utilities – Chinese text, pretty output."""

from __future__ import annotations

import json
from pathlib import Path

from image_translation.utilities.json_utils import load_json, save_json


class TestJsonUtils:
    def test_chinese_preserved(self, tmp_path: Path):
        data = {"text": "加厚升级", "nested": {"value": "中文测试"}}
        p = tmp_path / "test.json"
        save_json(p, data)

        raw = p.read_text(encoding="utf-8")
        assert "加厚升级" in raw
        assert "中文测试" in raw
        # ensure_ascii=False means Chinese stays as-is
        assert "\\u52a0" not in raw

    def test_pretty_valid_json(self, tmp_path: Path):
        data = {"a": 1, "b": [2, 3]}
        p = tmp_path / "pretty.json"
        save_json(p, data, pretty=True)

        raw = p.read_text(encoding="utf-8")
        # Should be valid JSON
        parsed = json.loads(raw)
        assert parsed == data
        # Should be indented
        assert "\n  " in raw

    def test_load_json(self, tmp_path: Path):
        data = {"x": "y"}
        p = tmp_path / "load.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        loaded = load_json(p)
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path: Path):
        p = tmp_path / "deep" / "nested" / "data.json"
        save_json(p, {"ok": True})
        assert p.exists()
