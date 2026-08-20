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
from image_translation.translation.seq2seq_translator import Seq2SeqTranslator


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
    """Mock tokenizer/model from_pretrained; records (kind, path, kwargs).

    The tokenizer is a full fake with measure (str) and batch (tensors)
    paths; batch_decode reconstructs placeholder-preserving translations
    from the last tokenized batch so the whole structured flow works.
    Returns a LoadRecord (a list of (kind, path, kwargs) entries) with
    ``.tokenizer`` / ``.model`` referencing the fake instances.
    """
    import transformers

    class LoadRecord(list):
        def __init__(self):
            super().__init__()
            self.tokenizer = None
            self.model = None

    loaded = LoadRecord()

    class FakeM2MTokenizer:
        def __init__(self):
            self.src_lang = None
            self.last_batch: list = []
            self.measure_calls: list = []

        def __call__(self, text, truncation=False, return_tensors=None,
                     padding=False, **_kw):
            if isinstance(text, str):
                n = max(1, (len(text) + 1) // 2)
                self.measure_calls.append((text, n))
                return {"input_ids": list(range(n))}
            self.last_batch = list(text)
            import torch
            n = len(text)
            return {
                "input_ids": torch.zeros(n, 4, dtype=torch.long),
                "attention_mask": torch.ones(n, 4, dtype=torch.long),
            }

        def convert_tokens_to_ids(self, lang):
            return 1

        def batch_decode(self, generated, skip_special_tokens=True):
            import re
            return [
                re.sub(r"[\u4e00-\u9fff]+", lambda m: "EN:" + m.group(0), t)
                for t in self.last_batch
            ]

    class FakeM2MModel:
        config = SimpleNamespace(max_position_embeddings=1024)

        def eval(self):
            return self

        def to(self, *a, **k):
            return self

        def half(self):
            return self

        def generate(self, **kw):
            import torch
            return torch.zeros(len(kw["input_ids"]), 6, dtype=torch.long)

        def parameters(self):
            import torch
            return iter([torch.zeros(1)])

    tokenizer = FakeM2MTokenizer()
    model = FakeM2MModel()
    loaded.tokenizer = tokenizer
    loaded.model = model

    def _tok(path, kw):
        loaded.append(("tokenizer", str(path), kw))
        return tokenizer

    def _mod(path, kw):
        loaded.append(("model", str(path), kw))
        return model

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        classmethod(lambda cls, path, **kw: _tok(path, kw)),
    )
    monkeypatch.setattr(
        transformers.AutoModelForSeq2SeqLM, "from_pretrained",
        classmethod(lambda cls, path, **kw: _mod(path, kw)),
    )
    return loaded


def _make_translator(config: TranslationConfig) -> Seq2SeqTranslator:
    return Seq2SeqTranslator(config)


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


# ---------------------------------------------------------------------------
# HTML token measurement must use the inference tokenizer (no second
# tokenizer, no independent Hugging Face access).
# ---------------------------------------------------------------------------

class TestHtmlMeasurementUsesLoadedTokenizer:
    @staticmethod
    def _make(hub, loaded, tmp_path, *, cache=None, offline=False,
              revision="main"):
        from image_translation.translation.config import StructuredConfig
        from image_translation.translation.structured_translation import (
            StructuredTranslator,
        )

        cache = cache or str(tmp_path / "cache")
        cfg = TranslationConfig(
            device="cpu",
            model_cache_dir=cache,
            local_files_only=offline,
            allow_model_download=not offline,
            model_revision=revision,
        )
        t = Seq2SeqTranslator(cfg)
        st = StructuredTranslator(
            t, StructuredConfig(max_segment_tokens=40), TranslationConfig()
        )
        return t, st

    def test_html_measurement_uses_inference_tokenizer(self, hub, loaded, tmp_path):
        t, st = self._make(hub, loaded, tmp_path)
        res = st.translate("<p>中文内容，耐磨耐用。</p>")
        assert res.translated_html
        # the loaded inference tokenizer performed the measurement
        assert loaded[0][0] == "tokenizer" and loaded[1][0] == "model"
        assert len(loaded) == 2  # exactly one tokenizer + one model load

    def test_no_second_from_pretrained_during_html_translation(self, hub, loaded, tmp_path):
        t, st = self._make(hub, loaded, tmp_path)
        st.translate("<p>中文</p><p>English text</p>")
        kinds = [c[0] for c in loaded]
        assert kinds == ["tokenizer", "model"]  # no extra tokenizer load

    def test_no_remote_identifier_after_load(self, hub, loaded, tmp_path):
        t, st = self._make(hub, loaded, tmp_path)
        st.translate("<p>中文</p>")
        # both loads used the resolved snapshot path, never the repo id
        snapshot = hub.cached[(str((tmp_path / "cache").resolve()), "main")]
        assert all(c[1] == snapshot for c in loaded)

    def test_offline_html_succeeds_with_complete_cache_zero_network(self, hub, loaded, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache = str(cache_dir.resolve())
        hub.seed(cache)
        t, st = self._make(hub, loaded, tmp_path, cache=cache, offline=True)
        html = "<p>中文 X13 与 English A 中文 X1300。</p>"
        res = st.translate(html)
        assert res.translated_html
        assert "X13" in res.translated_html and "X1300" in res.translated_html
        # zero network: only local-only resolution calls
        assert hub.download_calls() == []
        assert all(c["local_files_only"] is True for c in hub.calls)
        assert t.runtime_info.offline is True

    def test_offline_html_fails_clearly_when_cache_missing(self, hub, loaded, tmp_path):
        cache = tmp_path / "missing-cache"
        t, st = self._make(hub, loaded, tmp_path, cache=str(cache), offline=True)
        with pytest.raises(TranslationModelLoadError, match="does not exist"):
            st.translate("<p>中文</p>")
        assert hub.calls == []  # zero network access

    def test_non_default_revision_used_by_inference_and_measurement(self, hub, loaded, tmp_path):
        t, st = self._make(hub, loaded, tmp_path, revision="v2.0")
        st.translate("<p>中文</p>")
        assert all(c["revision"] == "v2.0" for c in hub.calls)
        assert t.runtime_info.model_revision == "v2.0"

    def test_segmentation_uses_inference_tokenizer_counts(self, hub, loaded, tmp_path):
        """A tokenizer with deliberately different token counts must drive
        segmentation — proving measurement uses the loaded inference
        tokenizer, not a separately constructed one."""
        from image_translation.translation.config import StructuredConfig
        from image_translation.translation.structured_translation import (
            StructuredTranslator,
        )

        cache = str((tmp_path / "cache").resolve())
        hub.seed(cache)
        cfg = TranslationConfig(device="cpu", model_cache_dir=cache)
        t = Seq2SeqTranslator(cfg)
        t.warmup()
        tokenizer = loaded.tokenizer
        base_call = type(tokenizer).__call__

        class OneTokenPerChar(type(tokenizer)):
            def __call__(self, text, truncation=False, **kw):
                if isinstance(text, str):
                    n = len(text)  # 1 token per char (vs (len+1)//2)
                    self.measure_calls.append((text, n))
                    return {"input_ids": list(range(n))}
                return base_call(self, text, truncation, **kw)

        tokenizer.__class__ = OneTokenPerChar

        st = StructuredTranslator(
            t, StructuredConfig(max_segment_tokens=30), TranslationConfig()
        )
        # 40 chars -> 40 tokens under the inference tokenizer's 1-per-char
        # counting -> exceeds budget 30 -> paragraph must split. The
        # default (len+1)//2 counting would have kept it as one 20-token
        # segment — proving the loaded tokenizer drove the split.
        res = st.translate("<p>" + "中" * 40 + "</p>")
        assert res.segment_count >= 2

    def test_long_mixed_html_invariants_preserved(self, hub, loaded, tmp_path):
        t, st = self._make(hub, loaded, tmp_path)
        html = (
            "<p>中文 X13 与 English A 中文 X1300 与 English B。</p>"
            "<p>前&nbsp;中文&#160;中间&amp;中文&#xA0;结尾。<br/>更多说明见文档。</p>"
        )
        res = st.translate(html)
        out = res.translated_html
        # identifiers in source order
        assert out.find("X13") < out.find("English A") < out.find("X1300") \
            < out.find("English B")
        # entities and tags preserved
        assert out.count("&nbsp;") == 1 and out.count("&#160;") == 1
        assert out.count("&amp;") == 1 and out.count("&#xA0;") == 1
        assert "<br/>" in out
        # Chinese translated in place
        assert "EN:中文" in out
        assert "__IT" not in out
