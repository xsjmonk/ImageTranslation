"""Tests for translation_server configuration loading + validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from translation_server.config import (
    RuntimeConfig,
    ServerConfig,
    TranslationServerConfig,
    load_server_config,
)


def _write_config(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "server.config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


class TestLoadServerConfig:
    def test_explicit_config_loads(self, tmp_path):
        cfg_path = _write_config(
            tmp_path,
            {
                "server": {"host": "127.0.0.1", "port": 9000},
                "runtime": {"warmup_on_start": False},
                "translation": {"model_name": "facebook/m2m100_418M"},
            },
        )
        cfg = load_server_config(cfg_path)
        assert cfg.server.port == 9000
        assert cfg.runtime.warmup_on_start is False
        assert cfg.translation.model_name == "facebook/m2m100_418M"

    def test_missing_explicit_config_fails(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_server_config(tmp_path / "nope.json")

    def test_malformed_json_fails(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{nope", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_server_config(p)

    def test_defaults_when_no_file(self, tmp_path, monkeypatch):
        # Point repo root resolution at an empty temp dir
        import translation_server.config as cfg_mod
        monkeypatch.setattr(
            cfg_mod, "_find_repo_root", lambda: tmp_path
        )
        cfg = load_server_config(None)
        assert cfg.server.port == 8091
        assert cfg.runtime.warmup_on_start is True
        assert cfg.translation.device == "cuda"

    def test_repo_default_config_loaded_when_no_path(self):
        """The repository's translation-server.config.json must load by default."""
        cfg = load_server_config(None)
        assert cfg.server.port == 8091
        assert cfg.runtime.warmup_on_start is True
        assert cfg.translation.device == "cuda"


class TestServerConfigValidation:
    def test_port_too_small(self, tmp_path):
        p = _write_config(tmp_path, {"server": {"port": 0}})
        with pytest.raises(ValueError, match="port"):
            load_server_config(p)

    def test_port_too_large(self, tmp_path):
        p = _write_config(tmp_path, {"server": {"port": 70000}})
        with pytest.raises(ValueError, match="port"):
            load_server_config(p)

    def test_workers_zero(self, tmp_path):
        p = _write_config(tmp_path, {"server": {"workers": 0}})
        with pytest.raises(ValueError, match="workers"):
            load_server_config(p)

    def test_workers_gt_one_with_cuda_rejected(self, tmp_path):
        p = _write_config(
            tmp_path,
            {
                "server": {"workers": 2},
                "translation": {"device": "cuda"},
            },
        )
        with pytest.raises(ValueError, match="workers must be 1"):
            load_server_config(p)

    def test_workers_gt_one_with_cpu_allowed(self, tmp_path):
        p = _write_config(
            tmp_path,
            {
                "server": {"workers": 2},
                "translation": {"device": "cpu"},
            },
        )
        cfg = load_server_config(p)
        assert cfg.server.workers == 2

    def test_invalid_log_level(self, tmp_path):
        p = _write_config(tmp_path, {"server": {"log_level": "verbose"}})
        with pytest.raises(ValueError, match="log_level"):
            load_server_config(p)

    def test_negative_cuda_device(self, tmp_path):
        p = _write_config(tmp_path, {"translation": {"cuda_device": -1}})
        with pytest.raises(ValueError, match="cuda_device"):
            load_server_config(p)

    def test_batch_size_zero(self, tmp_path):
        p = _write_config(tmp_path, {"translation": {"batch_size": 0}})
        with pytest.raises(ValueError, match="batch_size"):
            load_server_config(p)

    def test_max_input_characters_zero(self, tmp_path):
        p = _write_config(tmp_path, {"translation": {"max_input_characters": 0}})
        with pytest.raises(ValueError, match="max_input_characters"):
            load_server_config(p)
