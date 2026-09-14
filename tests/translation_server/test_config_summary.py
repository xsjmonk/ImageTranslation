"""Tests for the read-only startup config summary entry point."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from translation_server.config import build_config_summary, load_server_config


ROOT = Path(__file__).resolve().parents[2]


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


class TestConfigSummary:
    def test_repo_default_config_selects_nllb_profile(self):
        cfg = load_server_config(ROOT / "translation-server.config.json")
        assert cfg.active_model == "nllb"
        assert cfg.translation.backend == "current"
        assert cfg.translation.model_name == "facebook/nllb-200-distilled-600M"

    def test_summary_matches_normalized_loader(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        cfg_path = tmp_path / "server.config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "server": {"model_cache_dir": str(cache), "host": "127.0.0.1"},
                    "runtime": {"warmup_on_start": False},
                    "translation": _models_section("hymt2"),
                }
            ),
            encoding="utf-8",
        )
        cfg = load_server_config(cfg_path)
        summary = build_config_summary(cfg)
        assert summary["active_model"] == "hymt2"
        assert summary["backend"] == "hymt2"
        assert summary["model"] == "tencent/Hy-MT2-1.8B-FP8"
        assert summary["cache_dir"] == str(cache.resolve())
        assert summary["warmup_on_start"] is False

    def test_cli_print_config_summary_for_commented_profiles(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()

        def run_summary(active: str) -> dict:
            other = "hymt2" if active == "nllb" else "nllb"
            text = f"""
            {{
              // profile switch
              "server": {{ "model_cache_dir": "{str(cache).replace(chr(92), "/")}" }},
              "translation": {{
                "active_model": "{active}",
                "models": {{
                  "nllb": {{
                    "backend": "current",
                    "model_name": "facebook/nllb-200-distilled-600M",
                    "model_family": "nllb",
                    "model_revision": "main"
                  }},
                  "hymt2": {{
                    "backend": "hymt2",
                    "model_name": "tencent/Hy-MT2-1.8B-FP8",
                    "model_family": "hymt2",
                    "model_revision": "main"
                  }}
                }}
              }}
            }}
            """
            cfg_path = tmp_path / f"{active}.jsonc"
            cfg_path.write_text(text, encoding="utf-8")
            env = {"PYTHONPATH": str(ROOT / "src")}
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "translation_server",
                    "--print-config-summary",
                    "-c",
                    str(cfg_path),
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=ROOT,
                env={**dict(**__import__("os").environ), **env},
            )
            assert result.returncode == 0, result.stderr
            payload = json.loads(result.stdout)
            assert payload["active_model"] == active
            if active == "nllb":
                assert payload["backend"] == "current"
                assert payload["model"] == "facebook/nllb-200-distilled-600M"
            else:
                assert payload["backend"] == "hymt2"
                assert payload["model"] == "tencent/Hy-MT2-1.8B-FP8"
            assert other != active
            return payload

        run_summary("nllb")
        run_summary("hymt2")
