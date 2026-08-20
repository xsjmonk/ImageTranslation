"""Tests for the --check-cache preflight service (cache-aware CLI mode).

The preflight reuses the shared translator's authoritative resolution; it
must never load the tokenizer/model and must never hit the network in
offline mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _write_config(tmp_path: Path, cache_dir: str, offline: bool = False) -> Path:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    p = tmp_path / "server.config.json"
    p.write_text(json.dumps({
        "server": {"host": "127.0.0.1", "port": 18099},
        "server": {
            "model_cache_dir": cache_dir,
        },
        "translation": {
            "model_name": "facebook/nllb-200-distilled-600M",
            "allow_model_download": not offline,
            "local_files_only": offline,
        },
    }), encoding="utf-8")
    return p


class FakeHub:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.calls: list[dict] = []
        self.cached: dict = {}

    def snapshot_download(self, repo_id, revision="main", cache_dir=None,
                          local_files_only=False, **_kw):
        self.calls.append({
            "revision": revision, "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        })
        key = (cache_dir, revision)
        if key in self.cached:
            return self.cached[key]
        if local_files_only:
            raise FileNotFoundError(f"not cached: {key}")
        snap = self.tmp_path / f"dl_{len(self.calls)}"
        snap.mkdir()
        (snap / "config.json").write_text("{}", encoding="utf-8")
        (snap / "model.safetensors").write_bytes(b"w")
        (snap / "tokenizer.json").write_text("{}", encoding="utf-8")
        self.cached[key] = str(snap)
        return str(snap)

    def seed(self, cache_dir, revision="main") -> Path:
        snap = self.tmp_path / f"seed_{len(self.cached)}"
        snap.mkdir()
        (snap / "config.json").write_text("{}", encoding="utf-8")
        (snap / "model.safetensors").write_bytes(b"w")
        (snap / "tokenizer.json").write_text("{}", encoding="utf-8")
        self.cached[(cache_dir, revision)] = str(snap)
        return snap


@pytest.fixture
def hub(monkeypatch, tmp_path):
    fake = FakeHub(tmp_path)
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        fake.snapshot_download)
    return fake


@pytest.fixture
def no_model_load(monkeypatch):
    """from_pretrained must never be called by the preflight."""
    import transformers

    def _fail(*a, **k):
        raise AssertionError("check_cache must not load tokenizer/model")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        classmethod(lambda cls, *a, **k: _fail(*a, **k)))
    monkeypatch.setattr(
        transformers.AutoModelForSeq2SeqLM, "from_pretrained",
        classmethod(lambda cls, *a, **k: _fail(*a, **k)))


class TestCheckCacheService:
    def test_cache_hit_reports_snapshot_without_loading(self, hub, no_model_load, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache = str(cache_dir.resolve())
        snap = hub.seed(cache)
        cfg_path = _write_config(tmp_path, cache)

        from image_translation.translation import create_translator
        from image_translation.translation.config import TranslationConfig
        from translation_server.config import load_server_config

        sc = load_server_config(cfg_path)
        resolved = create_translator(sc.translation).check_cache()
        assert resolved.cache_status == "cache_hit"
        assert resolved.snapshot_path == str(snap)
        assert resolved.offline is False
        assert resolved.revision == "main"
        # only a local-only resolution happened; never a download
        assert [c["local_files_only"] for c in hub.calls] == [True]
        assert len(TranslationConfig().model_name) > 0  # sanity import

    def test_offline_miss_fails_without_network(self, hub, no_model_load, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache = str(cache_dir.resolve())
        cfg_path = _write_config(tmp_path, cache, offline=True)

        from image_translation.translation import create_translator
        from image_translation.translation.exceptions import TranslationModelLoadError
        from translation_server.config import load_server_config

        sc = load_server_config(cfg_path)
        with pytest.raises(TranslationModelLoadError, match="cache miss"):
            create_translator(sc.translation).check_cache()
        # exactly one local-only resolution; no network call
        assert len(hub.calls) == 1
        assert hub.calls[0]["local_files_only"] is True

    def test_incomplete_snapshot_rejected(self, hub, no_model_load, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache = str(cache_dir.resolve())
        snap = hub.tmp_path / "broken"
        snap.mkdir()
        (snap / "config.json").write_text("{}", encoding="utf-8")  # no weights
        hub.cached[(cache, "main")] = str(snap)

        from image_translation.translation import create_translator
        from image_translation.translation.exceptions import TranslationModelLoadError
        from translation_server.config import load_server_config

        sc = load_server_config(_write_config(tmp_path, cache))
        with pytest.raises(TranslationModelLoadError, match="incomplete model snapshot"):
            create_translator(sc.translation).check_cache()


class TestCheckCacheCli:
    def test_cli_check_cache_ok(self, hub, no_model_load, tmp_path, capsys):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache = str(cache_dir.resolve())
        hub.seed(cache)
        cfg_path = _write_config(tmp_path, cache)

        from translation_server.__main__ import main

        code = main(["-c", str(cfg_path), "--check-cache"])
        out = capsys.readouterr().out
        assert code == 0
        assert "status: cache_hit" in out
        assert "snapshot:" in out
        assert "revision: main" in out
        assert "offline: False" in out

    def test_cli_check_cache_offline_miss_exits_1(self, hub, no_model_load, tmp_path, capsys):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache = str(cache_dir.resolve())
        cfg_path = _write_config(tmp_path, cache, offline=True)

        from translation_server.__main__ import main

        code = main(["-c", str(cfg_path), "--check-cache"])
        err = capsys.readouterr().err
        assert code == 1
        assert "cache miss" in err
        # offline: never a network call
        assert all(c["local_files_only"] is True for c in hub.calls)


class TestCacheEndpoint:
    """GET /cache — thin HTTP facade over the translator's cache state."""

    def _client(self, translator):
        from fastapi.testclient import TestClient

        from translation_server.app import create_app
        from translation_server.runtime import TranslationRuntime, TranslationServerConfig

        runtime = TranslationRuntime(TranslationServerConfig())
        runtime._translator = translator
        return TestClient(create_app(runtime))

    def test_ready_reports_runtime_info(self):
        from image_translation.translation.models import TranslationRuntimeInfo
        from image_translation.translation.base import Translator
        from image_translation.translation.models import TranslationResult

        class ReadyFake(Translator):
            @property
            def name(self):
                return "fake"

            @property
            def runtime_info(self):
                return TranslationRuntimeInfo(
                    model_name="facebook/nllb-200-distilled-600M",
                    model_revision="main",
                    device="cuda:0",
                    precision="float32",
                    ready=True,
                    cache_dir="C:/models/hf",
                    snapshot_path="C:/models/hf/snap",
                    cache_status="cache_hit",
                    local_files_only=False,
                    offline=False,
                )

            def translate_text(self, text, source_lang="zh", target_lang="en"):
                return TranslationResult(source_text=text, translated_text=text)

            def translate_batch_texts(self, texts, source_lang="zh", target_lang="en"):
                return [self.translate_text(t) for t in texts]

        resp = self._client(ReadyFake()).get("/cache")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "facebook/nllb-200-distilled-600M"
        assert data["revision"] == "main"
        assert data["cache_dir"] == "C:/models/hf"
        assert data["snapshot_path"] == "C:/models/hf/snap"
        assert data["cache_status"] == "cache_hit"
        assert data["offline"] is False
        assert data["ready"] is True

    def test_not_ready_without_check_capability_falls_back_to_info(self):
        from image_translation.translation.models import TranslationRuntimeInfo
        from image_translation.translation.base import Translator
        from image_translation.translation.models import TranslationResult

        class LazyFake(Translator):
            @property
            def name(self):
                return "fake"

            @property
            def runtime_info(self):
                return TranslationRuntimeInfo(
                    model_name="fake", ready=False, cache_status="none")

            def translate_text(self, text, source_lang="zh", target_lang="en"):
                return TranslationResult(source_text=text, translated_text=text)

            def translate_batch_texts(self, texts, source_lang="zh", target_lang="en"):
                return [self.translate_text(t) for t in texts]

        resp = self._client(LazyFake()).get("/cache")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is False
        assert data["cache_status"] == "none"

    def test_not_ready_with_check_capability_reports_live_resolution(self, hub, no_model_load, tmp_path):
        from image_translation.translation import create_translator
        from image_translation.translation.config import TranslationConfig
        from image_translation.translation.models import TranslationRuntimeInfo
        from image_translation.translation.base import Translator
        from image_translation.translation.models import TranslationResult

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache = str(cache_dir.resolve())
        hub.seed(cache)

        real = create_translator(
            TranslationConfig(device="cpu", model_cache_dir=cache)
        )

        class Wrapper(Translator):
            @property
            def name(self):
                return "wrapped"

            @property
            def runtime_info(self):
                return TranslationRuntimeInfo(
                    model_name="fake", ready=False, cache_status="none")

            def check_cache(self):
                return real.check_cache()

            def translate_text(self, text, source_lang="zh", target_lang="en"):
                return TranslationResult(source_text=text, translated_text=text)

            def translate_batch_texts(self, texts, source_lang="zh", target_lang="en"):
                return [self.translate_text(t) for t in texts]

        resp = self._client(Wrapper()).get("/cache")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cache_status"] == "cache_hit"
        assert data["snapshot_path"]
        assert data["ready"] is False
        # live resolution used only local resolution; no model load
        assert all(c["local_files_only"] is True for c in hub.calls)

    def test_check_failure_reports_error_state(self, tmp_path):
        from image_translation.translation.exceptions import TranslationModelLoadError
        from image_translation.translation.models import TranslationRuntimeInfo
        from image_translation.translation.base import Translator
        from image_translation.translation.models import TranslationResult

        class BrokenCache(Translator):
            @property
            def name(self):
                return "broken"

            @property
            def runtime_info(self):
                return TranslationRuntimeInfo(model_name="fake", ready=False)

            def check_cache(self):
                raise TranslationModelLoadError("cache miss")

            def translate_text(self, text, source_lang="zh", target_lang="en"):
                return TranslationResult(source_text=text, translated_text=text)

            def translate_batch_texts(self, texts, source_lang="zh", target_lang="en"):
                return [self.translate_text(t) for t in texts]

        resp = self._client(BrokenCache()).get("/cache")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cache_status"] == "error"
        assert data["ready"] is False


class TestStartupCacheDiagnostics:
    """Config-derived cache diagnostics logged at startup (no model load)."""

    def test_cache_diagnostics_config_derived(self, tmp_path):
        from translation_server.config import load_server_config
        from translation_server.runtime import TranslationRuntime

        cfg_path = _write_config(tmp_path, str(tmp_path / "cache"))
        runtime = TranslationRuntime(load_server_config(cfg_path))
        diag = runtime.cache_diagnostics()
        assert diag["model"] == "facebook/nllb-200-distilled-600M"
        assert diag["revision"] == "main"
        assert diag["cache_dir"] == str((tmp_path / "cache").resolve())
        assert diag["offline"] is False

    def test_cache_diagnostics_offline(self, tmp_path):
        from translation_server.config import load_server_config
        from translation_server.runtime import TranslationRuntime

        cfg_path = _write_config(tmp_path, str(tmp_path / "cache"), offline=True)
        runtime = TranslationRuntime(load_server_config(cfg_path))
        assert runtime.cache_diagnostics()["offline"] is True

    def test_startup_logs_cache_line(self, hub, no_model_load, tmp_path, caplog):
        """main() logs the cache diagnostics line before any model work."""
        import logging

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cfg_path = _write_config(tmp_path, str(cache_dir.resolve()))

        from translation_server.__main__ import main

        with caplog.at_level(logging.INFO, logger="translation_server"):
            code = main(["-c", str(cfg_path), "--check-cache"])
        assert code == 0
        assert any("Model cache:" in r.message for r in caplog.records)
        assert any("offline=False" in r.message for r in caplog.records)
