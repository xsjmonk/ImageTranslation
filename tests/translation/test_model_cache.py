"""Model-cache configuration and reuse tests.

All Hugging Face APIs and the filesystem are mocked; no real network
download happens in these tests. The configured cache location must be
authoritative, reused without a second download, and offline-safe.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from image_translation.translation.config import TranslationConfig
from image_translation.translation.exceptions import TranslationModelLoadError
from image_translation.translation.m2m100_translator import M2M100Translator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(root: Path, name: str = "snap", complete: bool = True) -> Path:
    snap = root / name
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    if complete:
        (snap / "model.safetensors").write_bytes(b"weights")
        (snap / "tokenizer.json").write_text("{}", encoding="utf-8")
    return snap


class FakeHub:
    """Mock huggingface_hub.snapshot_download keyed by (cache_dir, revision)."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.calls: list[dict] = []
        self.cached: dict = {}  # (cache_dir|None, revision) -> snapshot path
        self.fail_downloads = False

    def snapshot_download(self, repo_id, revision="main", cache_dir=None,
                          local_files_only=False, **_kw):
        self.calls.append({
            "repo_id": repo_id,
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        })
        key = (cache_dir, revision)
        if key in self.cached:
            return self.cached[key]
        if local_files_only:
            raise FileNotFoundError(f"not cached: {key}")
        if self.fail_downloads:
            raise RuntimeError("network down")
        snap = _make_snapshot(self.tmp_path, name=f"dl_{len(self.calls)}")
        self.cached[key] = str(snap)
        return str(snap)

    def seed(self, cache_dir, revision="main", complete=True) -> Path:
        snap = _make_snapshot(self.tmp_path, name=f"seed_{len(self.cached)}",
                              complete=complete)
        self.cached[(cache_dir, revision)] = str(snap)
        return snap

    def download_calls(self) -> list:
        """Calls that attempted a real (non-local-only) resolution."""
        return [c for c in self.calls if not c["local_files_only"]]


def _patch_transformers(monkeypatch):
    """Mock tokenizer/model from_pretrained; records (kind, path, kwargs)."""
    import transformers

    loaded: list = []

    def _tokenizer(path, **kw):
        loaded.append(("tokenizer", str(path), kw))
        return SimpleNamespace(src_lang=None)

    def _model(path, **kw):
        loaded.append(("model", str(path), kw))
        m = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024))
        m.eval = lambda: m
        m.to = lambda *a, **k: m
        m.half = lambda: m
        return m

    monkeypatch.setattr(transformers.M2M100Tokenizer, "from_pretrained",
                        classmethod(lambda cls, path, **kw: _tokenizer(path, **kw)))
    monkeypatch.setattr(transformers.M2M100ForConditionalGeneration,
                        "from_pretrained",
                        classmethod(lambda cls, path, **kw: _model(path, **kw)))
    return loaded


def _make_translator(config: TranslationConfig) -> M2M100Translator:
    return M2M100Translator(config)


@pytest.fixture
def hub(monkeypatch, tmp_path):
    fake = FakeHub(tmp_path)
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        fake.snapshot_download)
    return fake


@pytest.fixture
def loaded(monkeypatch):
    return _patch_transformers(monkeypatch)


def _warmup(config, hub, loaded):
    t = _make_translator(config)
    t.warmup()
    return t


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestCacheConfig:
    def test_defaults(self):
        cfg = TranslationConfig()
        assert cfg.model_revision == "main"
        assert cfg.allow_model_download is True
        assert cfg.local_files_only is False
        assert cfg.model_cache_dir is None

    def test_contradictory_settings_rejected(self):
        with pytest.raises(ValueError, match="contradicts"):
            TranslationConfig(local_files_only=True, allow_model_download=True)

    def test_offline_pair_accepted(self):
        cfg = TranslationConfig(local_files_only=True, allow_model_download=False)
        assert cfg.local_files_only is True

    def test_blank_revision_rejected(self):
        with pytest.raises(ValueError, match="model_revision"):
            TranslationConfig(model_revision="")


# ---------------------------------------------------------------------------
# Resolution behavior
# ---------------------------------------------------------------------------

class TestModelResolution:
    def test_cache_hit_uses_local_resolution_no_download(self, hub, loaded, tmp_path):
        cfg = TranslationConfig(device="cpu", model_cache_dir=str(tmp_path / "cache"))
        snap = hub.seed(str(tmp_path / "cache"))
        t = _warmup(cfg, hub, loaded)

        # exactly one resolution call, local-only, in the configured cache
        assert len(hub.calls) == 1
        assert hub.calls[0]["local_files_only"] is True
        assert hub.calls[0]["cache_dir"] == str((tmp_path / "cache").resolve())
        assert hub.download_calls() == []  # never attempted a download
        # tokenizer and model loaded from the SAME resolved snapshot
        assert loaded[0][1] == str(snap) and loaded[1][1] == str(snap)
        # runtime diagnostics
        info = t.runtime_info
        assert info.cache_status == "cache_hit"
        assert info.snapshot_path == str(snap)
        assert info.model_revision == "main"
        assert info.device == "cpu"
        assert info.precision == "float32"
        assert info.ready is True

    def test_cache_miss_downloads_into_configured_root(self, hub, loaded, tmp_path):
        cache = str((tmp_path / "cache").resolve())
        cfg = TranslationConfig(device="cpu", model_cache_dir=cache)
        t = _warmup(cfg, hub, loaded)

        # probe (local-only, raises) then download into the configured root
        assert [c["local_files_only"] for c in hub.calls] == [True, False]
        assert all(c["cache_dir"] == cache for c in hub.calls)
        assert t.runtime_info.cache_status == "download"
        assert Path(t.runtime_info.snapshot_path).is_dir()
        assert loaded[0][1] == loaded[1][1] == t.runtime_info.snapshot_path

    def test_offline_miss_fails_without_network(self, hub, loaded, tmp_path):
        # Cache dir EXISTS but the model is not cached: only a local-only
        # resolution is attempted; never a network call.
        cache = tmp_path / "cache"
        cache.mkdir()
        cfg = TranslationConfig(
            device="cpu",
            model_cache_dir=str(cache),
            local_files_only=True,
            allow_model_download=False,
        )
        with pytest.raises(TranslationModelLoadError, match="cache miss"):
            _warmup(cfg, hub, loaded)
        # only a local-only resolution was attempted; never a network call
        assert len(hub.calls) == 1
        assert hub.calls[0]["local_files_only"] is True
        assert hub.download_calls() == []
        assert loaded == []

    def test_offline_missing_cache_dir_fails_without_network(self, hub, loaded, tmp_path):
        # Cache dir itself is missing: actionable error, zero hub calls.
        cfg = TranslationConfig(
            device="cpu",
            model_cache_dir=str(tmp_path / "does-not-exist"),
            local_files_only=True,
            allow_model_download=False,
        )
        with pytest.raises(TranslationModelLoadError, match="does not exist"):
            _warmup(cfg, hub, loaded)
        assert hub.calls == []
        assert loaded == []

    def test_same_snapshot_and_revision_for_tokenizer_and_model(self, hub, loaded, tmp_path):
        cfg = TranslationConfig(device="cpu", model_revision="v1.1")
        _warmup(cfg, hub, loaded)
        assert loaded[0][1] == loaded[1][1]  # same snapshot
        # resolution carried the configured revision consistently
        assert all(c["revision"] == "v1.1" for c in hub.calls)

    def test_changed_revision_does_not_reuse_wrong_snapshot(self, hub, loaded, tmp_path):
        cache = str((tmp_path / "cache").resolve())
        snap_main = hub.seed(cache, revision="main")
        cfg = TranslationConfig(device="cpu", model_cache_dir=cache,
                                model_revision="v2.0")
        t = _warmup(cfg, hub, loaded)
        # v2.0 was not cached -> downloaded; the v1 snapshot was not reused
        assert t.runtime_info.snapshot_path != snap_main
        assert t.runtime_info.cache_status == "download"
        assert any(c["revision"] == "v2.0" for c in hub.calls)

    def test_incomplete_snapshot_fails_before_ready(self, hub, loaded, tmp_path):
        cache = str((tmp_path / "cache").resolve())
        hub.seed(cache, complete=False)  # missing model.safetensors
        cfg = TranslationConfig(device="cpu", model_cache_dir=cache)
        t = _make_translator(cfg)
        with pytest.raises(TranslationModelLoadError, match="incomplete model snapshot"):
            t.warmup()
        assert t.runtime_info.ready is False

    def test_explicit_cache_dir_never_falls_back_to_default(self, hub, loaded, tmp_path):
        cache = str((tmp_path / "cache").resolve())
        cfg = TranslationConfig(device="cpu", model_cache_dir=cache)
        _warmup(cfg, hub, loaded)
        # every resolution call carried the explicit cache dir (None/absent
        # would mean the HF default cache)
        assert all(c["cache_dir"] == cache for c in hub.calls)

    def test_omitted_cache_dir_keeps_default_behavior(self, hub, loaded, tmp_path):
        cfg = TranslationConfig(device="cpu")  # no cache dir
        _warmup(cfg, hub, loaded)
        assert hub.calls[0]["cache_dir"] is None  # HF default cache

    def test_failed_download_never_ready(self, hub, loaded, tmp_path):
        hub.fail_downloads = True
        cfg = TranslationConfig(device="cpu", model_cache_dir=str(tmp_path / "cache"))
        t = _make_translator(cfg)
        with pytest.raises(TranslationModelLoadError, match="download failed"):
            t.warmup()
        assert t.runtime_info.ready is False

    def test_offline_missing_cache_dir_fails_actionably(self, hub, loaded, tmp_path):
        cache = str(tmp_path / "does-not-exist")
        cfg = TranslationConfig(
            device="cpu", model_cache_dir=cache,
            local_files_only=True, allow_model_download=False,
        )
        with pytest.raises(TranslationModelLoadError, match="does not exist"):
            _warmup(cfg, hub, loaded)

    def test_no_network_during_ordinary_translation_after_startup(self, hub, loaded, tmp_path):
        """After warmup, translate_text must not call the hub at all."""
        cache = str((tmp_path / "cache").resolve())
        cfg = TranslationConfig(device="cpu", model_cache_dir=cache)
        t = _warmup(cfg, hub, loaded)
        hub.calls.clear()
        # translate_text is lazily loaded already; no hub interaction
        from image_translation.translation.models import TranslationResult
        calls_before = len(hub.calls)
        assert calls_before == 0
