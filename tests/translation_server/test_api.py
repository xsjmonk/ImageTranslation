"""Tests for translation_server FastAPI app — uses mocked translators."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from image_translation.translation.base import Translator
from image_translation.translation.exceptions import (
    TranslationDeviceError,
    TranslationInputError,
    TranslationModelLoadError,
)
from image_translation.translation.models import TranslationResult, TranslationRuntimeInfo
from image_translation.translation.text_utils import preprocess


# ---------------------------------------------------------------------------
# Fake translators for testing
# ---------------------------------------------------------------------------

class FakeTranslator(Translator):
    """Returns canned translations — no GPU needed."""

    def __init__(self, ready: bool = True, max_input_characters: int = 4000) -> None:
        self._ready = ready
        self._max = max_input_characters
        self.styles = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def runtime_info(self) -> TranslationRuntimeInfo:
        return TranslationRuntimeInfo(
            model_name="fake",
            model_revision="main",
            device="cpu",
            precision="float32",
            cuda_available=False,
            ready=self._ready,
            cache_dir="C:/models/hf",
            snapshot_path="C:/models/hf/snap",
            cache_status="cache_hit",
            local_files_only=True,
            offline=True,
        )


    def measure_source_tokens(self, text: str, source_lang: str = "zh") -> int:
        """Token count used by HTML segmentation (no model call)."""
        return max(1, (len(text) + 1) // 2)
    def translate_text(
        self, text: str, source_lang: str = "zh", target_lang: str = "en",
        style=None,
    ) -> TranslationResult:
        cleaned = preprocess(text, max_characters=self._max)
        self.styles.append(style)
        return TranslationResult(
            source_text=cleaned,
            translated_text=f"[EN] {cleaned}",
            source_language=source_lang,
            target_language=target_lang,
            model_name="fake",
            device="cpu",
            style=getattr(style, "value", style) or "sentence",
        )

    def translate_batch_texts(
        self, texts, source_lang: str = "zh", target_lang: str = "en",
        max_new_tokens=None, style=None
    ):
        return [
            self.translate_text(t, source_lang, target_lang, style)
            for t in texts
        ]

    def warmup(self) -> None:
        pass


class SlowTranslator(FakeTranslator):
    """Deliberately blocks for a short time to simulate GPU inference."""

    def __init__(self, delay: float = 0.5) -> None:
        super().__init__()
        self._delay = delay

    def translate_text(self, text, source_lang="zh", target_lang="en", style=None):
        time.sleep(self._delay)
        return super().translate_text(text, source_lang, target_lang, style)


# ---------------------------------------------------------------------------
# App factory helpers
# ---------------------------------------------------------------------------

def _make_client(runtime_translator):
    from translation_server.runtime import TranslationRuntime, TranslationServerConfig
    from translation_server.app import create_app

    config = TranslationServerConfig()
    config.runtime.warmup_on_start = False
    runtime = TranslationRuntime(config)
    runtime._translator = runtime_translator

    app = create_app(runtime)
    return TestClient(app)


@pytest.fixture
def client():
    return _make_client(FakeTranslator())


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["ready"] is True

    def test_health_returns_model_info(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "model" in data
        assert "device" in data

    def test_health_exposes_cache_diagnostics(self, client):
        resp = client.get("/health")
        data = resp.json()
        # model, revision, device, precision, cache root, snapshot path,
        # cache status, and ready state are observable
        assert data["model"] == "fake"
        assert data["model_revision"] == "main"
        assert data["device"] == "cpu"
        assert data["precision"] == "float32"
        assert data["cache_dir"] == "C:/models/hf"
        assert data["snapshot_path"] == "C:/models/hf/snap"
        assert data["cache_status"] == "cache_hit"
        assert data["ready"] is True
        # effective offline mode is accurate (downloads disabled)
        assert data["offline"] is True
        assert data["local_files_only"] is True

    def test_health_not_ready_reports_starting(self):
        c = _make_client(FakeTranslator(ready=False))
        resp = c.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "starting"
        assert data["ready"] is False


# ---------------------------------------------------------------------------
# Translate — JSON contract
# ---------------------------------------------------------------------------

class TestTranslate:
    def test_translate_chinese(self, client):
        resp = client.post(
            "/translate",
            json={"text": "你好"},
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "translation" in data
        assert len(data["translation"]) > 0

    def test_translate_utf8_chinese(self, client):
        resp = client.post("/translate", json={"text": "加厚防水面料"})
        assert resp.status_code == 200
        data = resp.json()
        assert "translation" in data

    def test_response_contains_exactly_translation(self, client):
        resp = client.post("/translate", json={"text": "测试"})
        assert resp.status_code == 200
        assert set(resp.json().keys()) == {"translation"}

    def test_whitespace_only_400(self, client):
        resp = client.post("/translate", json={"text": "   "})
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_missing_text_422(self, client):
        resp = client.post("/translate", json={})
        assert resp.status_code == 422
        assert resp.json()["code"] == "invalid_request"

    def test_too_long_400(self):
        """Length limit comes from configured translator max, not the API model."""
        c = _make_client(FakeTranslator(max_input_characters=10))
        resp = c.post("/translate", json={"text": "x" * 50})
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_malformed_json_422(self, client):
        resp = client.post("/translate", data="{not json", headers={"Content-Type": "application/json"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Error semantics: 503 vs 500
# ---------------------------------------------------------------------------

class TestErrorSemantics:
    def test_device_unavailable_503(self):
        class DeviceFailTranslator(FakeTranslator):
            def translate_text(self, text, source_lang="zh", target_lang="en", style=None):
                raise TranslationDeviceError("CUDA unavailable")

        c = _make_client(DeviceFailTranslator())
        resp = c.post("/translate", json={"text": "你好"})
        assert resp.status_code == 503
        assert "error" in resp.json()
        assert "CUDA" not in resp.json()["error"]  # no internals leaked

    def test_model_load_failure_503(self):
        class ModelFailTranslator(FakeTranslator):
            def translate_text(self, text, source_lang="zh", target_lang="en", style=None):
                raise TranslationModelLoadError("model weights missing")

        c = _make_client(ModelFailTranslator())
        resp = c.post("/translate", json={"text": "你好"})
        assert resp.status_code == 503

    def test_unexpected_error_500(self):
        class CrashTranslator(FakeTranslator):
            def translate_text(self, text, source_lang="zh", target_lang="en", style=None):
                raise RuntimeError("boom: internal detail")

        c = _make_client(CrashTranslator())
        resp = c.post("/translate", json={"text": "你好"})
        assert resp.status_code == 500
        data = resp.json()
        assert "error" in data
        assert "boom" not in data["error"]  # no raw exception leaked


# ---------------------------------------------------------------------------
# Long-running inference must not block the event loop
# ---------------------------------------------------------------------------

class TestLongRunningInference:
    def test_translate_waits_and_returns_200(self):
        c = _make_client(SlowTranslator(delay=0.5))
        start = time.monotonic()
        resp = c.post("/translate", json={"text": "你好"})
        elapsed = time.monotonic() - start

        assert resp.status_code == 200
        assert set(resp.json().keys()) == {"translation"}
        assert elapsed >= 0.4  # it actually waited for inference

    def test_health_responsive_during_translation(self):
        c = _make_client(SlowTranslator(delay=1.0))
        results: dict = {}
        done = threading.Event()

        def do_translate():
            results["translate"] = c.post("/translate", json={"text": "你好"})
            done.set()

        t = threading.Thread(target=do_translate, daemon=True)
        t.start()
        time.sleep(0.2)  # let translation start sleeping

        health = c.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        done.wait(timeout=5)
        assert results["translate"].status_code == 200
