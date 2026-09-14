"""Tests for comment-capable translation-server configuration documents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from translation_server.config import load_server_config, load_server_config_from_text
from translation_server.config_document import (
    ConfigDocumentError,
    parse_config_text,
    select_model_profile,
)


def _models_section(active: str = "nllb") -> dict:
    return {
        "active_model": active,
        "models": {
            "nllb": {
                "backend": "current",
                "model_name": "facebook/nllb-200-distilled-600M",
                "model_family": "nllb",
                "model_revision": "main",
            },
            "hymt2": {
                "backend": "hymt2",
                "model_name": "tencent/Hy-MT2-1.8B-FP8",
                "model_family": "hymt2",
                "model_revision": "main",
            },
        },
    }


class TestJson5Parser:
    def test_accepts_line_comments_and_literal_urls(self):
        text = """
        {
          // Network endpoint
          "server": { "host": "127.0.0.1", "port": 8091 },
          "translation": {
            "active_model": "nllb",
            "models": {
              "nllb": {
                "backend": "current",
                "model_name": "facebook/nllb-200-distilled-600M",
                "model_family": "nllb",
                "model_revision": "main"
              }
            }
          },
          "text_example": "https://example.test/a//b /* literal */"
        }
        """
        parsed = parse_config_text(text)
        assert parsed["text_example"] == "https://example.test/a//b /* literal */"
        assert parsed["server"]["host"] == "127.0.0.1"

    def test_block_comments_and_array_element_comments(self):
        text = """
        {
          /* server block */
          "server": { "host": "127.0.0.1", "port": 8091 },
          "structured": {
            "excluded_tags": [
              "script", // inline
              /* block */ "style"
            ]
          },
          "translation": {
            "active_model": "nllb",
            "models": {
              "nllb": {
                "backend": "current",
                "model_name": "facebook/nllb-200-distilled-600M",
                "model_family": "nllb",
                "model_revision": "main"
              }
            }
          }
        }
        """
        parsed = parse_config_text(text)
        assert parsed["structured"]["excluded_tags"] == ["script", "style"]

    def test_escaped_quotes_in_strings(self):
        parsed = parse_config_text(
            r'{"translation":{"active_model":"nllb","models":{"nllb":'
            r'{"backend":"current","model_name":"facebook/nllb-200-distilled-600M",'
            r'"model_family":"nllb","model_revision":"main"}}},'
            r'"note":"say \"hello\" // not a comment"}'
        )
        assert parsed["note"] == 'say "hello" // not a comment'

    def test_malformed_json_reports_location(self):
        with pytest.raises(ConfigDocumentError, match="<config>"):
            parse_config_text("{nope")

    def test_malformed_block_comment(self):
        with pytest.raises(ConfigDocumentError):
            parse_config_text('{"server": {"host": "x" /* unclosed }')


class TestConfigOwnership:
    def test_active_model_selects_profile_identity(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        cfg = load_server_config_from_text(
            json.dumps(
                {
                    "server": {"model_cache_dir": str(cache)},
                    "translation": _models_section("hymt2"),
                }
            ),
            tmp_path / "cfg.json",
        )
        assert cfg.active_model == "hymt2"
        assert cfg.translation.backend == "hymt2"
        assert cfg.translation.model_name == "tencent/Hy-MT2-1.8B-FP8"

    def test_switching_active_model_keeps_shared_settings(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        shared = {
            "source_language": "zho_Hans",
            "target_language": "eng_Latn",
            "device": "cpu",
            "generation": {"num_beams": 2},
            **_models_section("nllb"),
        }
        current = load_server_config_from_text(
            json.dumps(
                {"server": {"model_cache_dir": str(cache)}, "translation": shared}
            ),
            tmp_path / "current.json",
        )
        hymt2 = load_server_config_from_text(
            json.dumps(
                {
                    "server": {"model_cache_dir": str(cache)},
                    "translation": {**shared, "active_model": "hymt2"},
                }
            ),
            tmp_path / "hymt2.json",
        )
        assert current.translation.source_language == hymt2.translation.source_language
        assert current.translation.device == hymt2.translation.device
        assert (
            current.translation.generation.num_beams
            == hymt2.translation.generation.num_beams
        )
        assert current.translation.backend == "current"
        assert current.active_model == "nllb"
        assert hymt2.translation.backend == "hymt2"

    def test_unknown_active_model_fails_early(self):
        with pytest.raises(ValueError, match="active_model 'missing'"):
            load_server_config_from_text(
                json.dumps(
                    {
                        "translation": {
                            "active_model": "missing",
                            "models": _models_section()["models"],
                        }
                    }
                )
            )

    def test_unknown_backend_in_profile_fails(self):
        with pytest.raises(ValueError, match="backend must be one of"):
            load_server_config_from_text(
                json.dumps(
                    {
                        "translation": {
                            "active_model": "nllb",
                            "models": {
                                "nllb": {
                                    "backend": "other",
                                    "model_name": "facebook/nllb-200-distilled-600M",
                                    "model_family": "nllb",
                                    "model_revision": "main",
                                }
                            },
                        }
                    }
                )
            )

    def test_model_profile_cannot_set_cache_root(self):
        with pytest.raises(ValueError, match="model_cache_dir"):
            load_server_config_from_text(
                json.dumps(
                    {
                        "translation": {
                            "active_model": "nllb",
                            "models": {
                                "nllb": {
                                    "backend": "current",
                                    "model_name": "facebook/nllb-200-distilled-600M",
                                    "model_family": "nllb",
                                    "model_revision": "main",
                                    "model_cache_dir": "D:/Caches/translation",
                                }
                            },
                        }
                    }
                )
            )

    def test_conflicting_legacy_and_server_cache_rejected(self, tmp_path):
        cache = tmp_path / "server-cache"
        legacy = tmp_path / "legacy-cache"
        cache.mkdir()
        legacy.mkdir()
        with pytest.raises(ValueError, match="conflicts with server.model_cache_dir"):
            load_server_config(
                _write_json(
                    tmp_path,
                    {
                        "server": {"model_cache_dir": str(cache)},
                        "translation": {"model_cache_dir": str(legacy)},
                    },
                )
            )

    def test_flat_legacy_schema_still_supported(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        cfg = load_server_config(
            _write_json(
                tmp_path,
                {
                    "server": {"model_cache_dir": str(cache)},
                    "translation": {
                        "backend": "hymt2",
                        "model_name": "tencent/Hy-MT2-1.8B-FP8",
                    },
                },
            )
        )
        assert cfg.translation.backend == "hymt2"
        assert cfg.active_model == "hymt2"

    def test_legacy_model_keys_with_models_registry_rejected(self):
        with pytest.raises(ValueError, match="unknown translation key"):
            load_server_config_from_text(
                json.dumps(
                    {
                        "translation": {
                            "backend": "current",
                            **_models_section(),
                        }
                    }
                )
            )

    def test_unknown_top_level_key_rejected(self):
        with pytest.raises(ValueError, match="unknown top-level key"):
            load_server_config_from_text('{"text_example": "x"}')

    def test_quality_owned_at_top_level_only(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        cfg = load_server_config_from_text(
            json.dumps(
                {
                    "server": {"model_cache_dir": str(cache)},
                    "quality": {"unknown_token_policy": "reject"},
                    "translation": _models_section(),
                }
            ),
            tmp_path / "cfg.json",
        )
        assert cfg.translation.quality.unknown_token_policy == "reject"

    def test_comment_preserving_config_loads(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        text = """
        {
          // switch engines by changing active_model only
          "server": { "model_cache_dir": "%s" },
          "translation": {
            "active_model": "nllb",
            "models": {
              "nllb": {
                "backend": "current",
                "model_name": "facebook/nllb-200-distilled-600M",
                "model_family": "nllb",
                "model_revision": "main"
              }
            }
          }
        }
        """ % str(cache).replace("\\", "/")
        cfg = load_server_config_from_text(text, tmp_path / "commented.json")
        assert cfg.active_model == "nllb"
        assert cfg.translation.backend == "current"


def _write_json(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "server.config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestSelectModelProfile:
    def test_defaults_for_legacy_flat_schema(self):
        active, profile = select_model_profile({})
        assert active == "current"
        assert profile["backend"] == "current"
        assert profile["model_name"] == "facebook/nllb-200-distilled-600M"
