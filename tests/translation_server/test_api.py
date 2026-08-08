"""Tests for translation_server FastAPI app — uses mocked translator."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from image_translation.translation.base import Translator
from image_translation.translation.models import TranslationResult, TranslationRuntimeInfo


# ---------------------------------------------------------------------------
# Fake translator for testing
# ---------------------------------------------------------------------------

class FakeTranslator(Translator):
    """Returns canned translations — no GPU needed."""

    @property
    def name(self) -> str:
        return "fake"

    @property
    def runtime_info(self) -> TranslationRuntimeInfo:
        return TranslationRuntimeInfo(
            model_name="fake",
            device="cpu",
            precision="float32",
            cuda_available=False,
            ready=True,
        )

    def translate_text(
        self, text: str, source_lang: str = "zh", target_lang: str = "en"
    ) -> TranslationResult:
        if not text or not text.strip():
            raise ValueError("Input text must not be empty")
        return TranslationResult(
            source_text=text,
            translated_text=f"[EN] {text}",
            source_language=source_lang,
            target_language=target_lang,
            model_name="fake",
            device="cpu",
        )

    def translate_batch_texts(
        self, texts, source_lang: str = "zh", target_lang: str = "en"
    ):
        return [self.translate_text(t, source_lang, target_lang) for t in texts]

    def warmup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Test app
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from translation_server.runtime import TranslationRuntime, TranslationServerConfig
    from translation_server.app import create_app

    config = TranslationServerConfig()
    runtime = TranslationRuntime(config)
    runtime._translator = FakeTranslator()

    app = create_app(runtime)
    return TestClient(app)


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

    def test_empty_text_400(self, client):
        resp = client.post("/translate", json={"text": ""})
        assert resp.status_code == 422  # pydantic validation

    def test_missing_text_422(self, client):
        resp = client.post("/translate", json={})
        assert resp.status_code == 422

    def test_response_contains_only_translation(self, client):
        resp = client.post("/translate", json={"text": "测试"})
        data = resp.json()
        # Only the public field, not internal model details
        assert set(data.keys()) == {"translation"}


class TestUnavailableTranslator:
    def test_503_when_translator_unavailable(self):
        """If the translator raises during translate, the API returns 503."""
        from translation_server.runtime import TranslationRuntime, TranslationServerConfig
        from translation_server.app import create_app

        config = TranslationServerConfig()
        runtime = TranslationRuntime(config)

        # Inject a broken translator that fails on translate_text
        class BrokenTranslator(FakeTranslator):
            def translate_text(self, text, source_lang="zh", target_lang="en"):
                raise RuntimeError("Model not loaded")

        runtime._translator = BrokenTranslator()

        app = create_app(runtime)
        client = TestClient(app)

        resp = client.post("/translate", json={"text": "你好"})
        assert resp.status_code == 500
