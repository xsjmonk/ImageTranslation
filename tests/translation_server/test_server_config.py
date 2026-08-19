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
                "structured": {"preserve_patterns": ["^[A-Z]{2,}/\\d{4}$"]},
                "translation": {"model_name": "facebook/m2m100_418M"},
            },
        )
        cfg = load_server_config(cfg_path)
        assert cfg.server.port == 9000
        assert cfg.runtime.warmup_on_start is False
        assert cfg.translation.model_name == "facebook/m2m100_418M"
        assert cfg.structured.preserve_patterns == ("^[A-Z]{2,}/\\d{4}$",)

    def test_quality_and_generation_policy_load(self, tmp_path):
        cfg_path = _write_config(
            tmp_path,
            {
                "quality": {"unknown_token_policy": "reject"},
                "translation": {
                    "generation": {
                        "short_text_max_new_tokens": 32,
                        "retry_num_beams": 1,
                    }
                },
            },
        )
        cfg = load_server_config(cfg_path)
        assert cfg.translation.quality.unknown_token_policy == "reject"
        assert cfg.translation.generation.short_text_max_new_tokens == 32
        assert cfg.translation.generation.retry_num_beams == 1

    def test_invalid_unknown_token_policy_rejected(self, tmp_path):
        cfg_path = _write_config(
            tmp_path,
            {"quality": {"unknown_token_policy": "replace"}},
        )
        with pytest.raises(ValueError, match="unknown_token_policy"):
            load_server_config(cfg_path)

    def test_glossary_path_resolves_from_config_directory(self, tmp_path):
        glossary = tmp_path / "terms.tsv"
        glossary.write_text(
            "source\ttarget\texact\n蔡司\tZeiss\ttrue\n",
            encoding="utf-8",
        )
        cfg_path = _write_config(
            tmp_path,
            {
                "quality": {
                    "glossary_file": "terms.tsv",
                    "glossary_required": True,
                }
            },
        )
        cfg = load_server_config(cfg_path)
        assert Path(cfg.translation.quality.glossary_file) == glossary.resolve()

    def test_legacy_json_glossary_rejected(self, tmp_path):
        cfg_path = _write_config(
            tmp_path,
            {"structured": {"glossary": []}},
        )
        with pytest.raises(ValueError, match="quality.glossary_file"):
            load_server_config(cfg_path)

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


class TestModelCacheConfig:
    def test_absolute_cache_dir_preserved_and_normalized(self, tmp_path):
        cfg_path = _write_config(
            tmp_path,
            {"translation": {"model_cache_dir": "D:/Models/ImageTranslation/hf"}},
        )
        cfg = load_server_config(cfg_path)
        assert Path(cfg.translation.model_cache_dir) == \
            Path("D:/Models/ImageTranslation/hf").resolve()

    def test_relative_cache_dir_resolves_against_config_dir(self, tmp_path):
        cfg_path = _write_config(
            tmp_path,
            {"translation": {"model_cache_dir": "models/hf-cache"}},
        )
        cfg = load_server_config(cfg_path)
        assert Path(cfg.translation.model_cache_dir) == \
            (tmp_path / "models" / "hf-cache").resolve()

    def test_env_var_expansion_in_cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("IT_TEST_CACHE", "expanded-cache")
        cfg_path = _write_config(
            tmp_path,
            {"translation": {"model_cache_dir": "${IT_TEST_CACHE}/hf"}},
        )
        cfg = load_server_config(cfg_path)
        assert Path(cfg.translation.model_cache_dir) == \
            (tmp_path / "expanded-cache" / "hf").resolve()

    def test_omitted_cache_dir_defaults_to_none(self, tmp_path):
        cfg_path = _write_config(tmp_path, {"translation": {}})
        cfg = load_server_config(cfg_path)
        assert cfg.translation.model_cache_dir is None
        assert cfg.translation.model_revision == "main"
        assert cfg.translation.allow_model_download is True
        assert cfg.translation.local_files_only is False

    def test_new_fields_parsed(self, tmp_path):
        cfg_path = _write_config(
            tmp_path,
            {
                "translation": {
                    "model_revision": "v1.1",
                    "allow_model_download": False,
                    "local_files_only": True,
                }
            },
        )
        cfg = load_server_config(cfg_path)
        assert cfg.translation.model_revision == "v1.1"
        assert cfg.translation.allow_model_download is False
        assert cfg.translation.local_files_only is True

    def test_contradictory_download_offline_rejected(self, tmp_path):
        cfg_path = _write_config(
            tmp_path,
            {
                "translation": {
                    "allow_model_download": True,
                    "local_files_only": True,
                }
            },
        )
        with pytest.raises(ValueError, match="contradicts"):
            load_server_config(cfg_path)

    def test_cache_dir_pointing_at_file_rejected(self, tmp_path):
        target = tmp_path / "not-a-dir"
        target.write_text("x", encoding="utf-8")
        cfg_path = _write_config(
            tmp_path, {"translation": {"model_cache_dir": str(target)}}
        )
        with pytest.raises(ValueError, match="not a directory"):
            load_server_config(cfg_path)
