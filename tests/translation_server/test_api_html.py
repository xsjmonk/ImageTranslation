"""API tests for format=html structured translation (fake translator, no GPU)."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from image_translation.translation.base import Translator
from image_translation.translation.models import TranslationResult
from image_translation.translation.structured_translation import StructuredTranslator
from image_translation.translation.config import TranslationConfig


class FakeTranslator(Translator):
    @property
    def name(self) -> str:
        return "fake"

    @property
    def runtime_info(self):
        from image_translation.translation.models import TranslationRuntimeInfo
        return TranslationRuntimeInfo(model_name="fake", device="cpu", ready=True)


    def measure_source_tokens(self, text: str, source_lang: str = "zh") -> int:
        """Token count used by HTML segmentation (no model call)."""
        return max(1, (len(text) + 1) // 2)
    def translate_text(
        self, text, source_lang="zh", target_lang="en",
        max_new_tokens=None, style=None
    ):
        return self.translate_batch_texts(
            [text], source_lang, target_lang, max_new_tokens, style
        )[0]

    def translate_batch_texts(
        self, texts, source_lang="zh", target_lang="en",
        max_new_tokens=None, style=None
    ):
        import re
        return [
            TranslationResult(
                source_text=t,
                translated_text=re.sub(
                    r"[\u4e00-\u9fff]+", lambda m: "EN:" + m.group(0), t
                ),
                model_name="fake", device="cpu",
            )
            for t in texts
        ]


@pytest.fixture
def client():
    from translation_server.runtime import TranslationRuntime, TranslationServerConfig
    from translation_server.app import create_app

    config = TranslationServerConfig()
    config.runtime.warmup_on_start = False
    runtime = TranslationRuntime(config)
    runtime._translator = FakeTranslator()

    app = create_app(runtime)
    return TestClient(app)


MIXED = "<p>这是 <strong>加厚防水面料</strong>，适合 daily use。</p>"


class TestFormatField:
    def test_format_omitted_is_plain(self, client):
        resp = client.post("/translate", json={"text": "你好"})
        assert resp.status_code == 200
        assert set(resp.json().keys()) == {"translation"}

    def test_format_plain_explicit(self, client):
        resp = client.post("/translate", json={"text": "你好", "format": "plain"})
        assert resp.status_code == 200
        assert resp.json()["translation"] == "EN:你好"

    def test_format_html(self, client):
        resp = client.post(
            "/translate", json={"text": MIXED, "format": "html"}
        )
        assert resp.status_code == 200
        out = resp.json()["translation"]
        assert "<strong>" in out and "</strong>" in out
        # Tags preserved at original positions; EN: prefix from the fake
        assert out == "<p>EN:这是 <strong>EN:加厚防水面料</strong>，EN:适合 daily use。</p>"

    def test_plain_does_not_auto_detect_html(self, client):
        """format omitted + '<' chars must NOT trigger HTML handling."""
        resp = client.post("/translate", json={"text": "a < b > c"})
        assert resp.status_code == 200
        assert resp.json()["translation"] == "a < b > c"

    def test_invalid_format_422(self, client):
        resp = client.post("/translate", json={"text": "x", "format": "xml"})
        assert resp.status_code == 422

    def test_optional_language_fields(self, client):
        resp = client.post(
            "/translate",
            json={"text": "你好", "format": "plain",
                  "source_language": "zh", "target_language": "en"},
        )
        assert resp.status_code == 200

    def test_language_fields_reach_translator(self):
        """source/target language request fields propagate to model calls."""
        from translation_server.runtime import TranslationRuntime, TranslationServerConfig
        from translation_server.app import create_app

        config = TranslationServerConfig()
        config.runtime.warmup_on_start = False
        runtime = TranslationRuntime(config)

        class RecordingFake(FakeTranslator):
            def __init__(self):
                super().__init__()
                self.calls = []

            def translate_batch_texts(self, texts, source_lang="zh", target_lang="en", max_new_tokens=None):
                self.calls.append((source_lang, target_lang))
                return super().translate_batch_texts(
                    texts, source_lang, target_lang, max_new_tokens
                )

        fake = RecordingFake()
        runtime._translator = fake
        c = TestClient(create_app(runtime))
        resp = c.post(
            "/translate",
            json={"text": "<p>加厚防水面料</p>", "format": "html",
                  "source_language": "zh", "target_language": "fr"},
        )
        assert resp.status_code == 200
        assert fake.calls, "model was never called"
        assert all(src == "zh" and tgt == "fr" for src, tgt in fake.calls)

    def test_mixed_text_exact_preservation_via_api(self, client):
        """HTML requests: exact English/identifier/tag preservation."""
        resp = client.post(
            "/translate",
            json={
                "text": (
                    "<p>请 click <strong>Continue</strong> 然后继续。</p>"
                    "<p>Use the USB-C cable 连接设备。</p>"
                    "<p>型号 ABC-123，compatible with Windows 11。</p>"
                ),
                "format": "html",
            },
        )
        assert resp.status_code == 200
        out = resp.json()["translation"]
        assert "<strong>Continue</strong>" in out
        assert "Use the USB-C cable" in out
        assert "ABC-123" in out and "Windows 11" in out
        assert "click" in out

    def test_entity_spelling_preserved_via_api(self, client):
        """HTML requests: entity spellings survive exactly (never decoded
        or normalized), and &lt;br&gt; never becomes a real tag."""
        resp = client.post(
            "/translate",
            json={
                "text": (
                    "<p>中文&nbsp;English 与&#160;中文，&#xA0;以及&amp;。</p>"
                    "<p>中文&lt;br&gt;English 与<br/>中文</p>"
                ),
                "format": "html",
            },
        )
        assert resp.status_code == 200
        out = resp.json()["translation"]
        assert "&nbsp;" in out and "&#160;" in out and "&#xA0;" in out
        assert "&amp;" in out
        assert "&lt;br&gt;" in out
        assert "<br/>" in out
        # escaped markup never became a REAL <br> tag (only <br/> survives)
        assert len(re.findall(r"<br(?!/)>", out)) == 0

    def test_error_returns_no_partial_html(self):
        """On failure the error envelope contains no translation field."""
        from translation_server.runtime import TranslationRuntime, TranslationServerConfig
        from translation_server.app import create_app

        config = TranslationServerConfig()
        config.runtime.warmup_on_start = False
        config.structured.max_chapter_characters = 100
        runtime = TranslationRuntime(config)

        class CrashFake(FakeTranslator):
            def translate_batch_texts(self, texts, source_lang="zh", target_lang="en", max_new_tokens=None):
                raise RuntimeError("boom")

        runtime._translator = CrashFake()
        c = TestClient(create_app(runtime))
        resp = c.post("/translate", json={"text": "<p>加厚防水面料</p>", "format": "html"})
        assert resp.status_code == 500
        assert "translation" not in resp.json()
        assert resp.json()["error"]


class TestHtmlErrors:
    def test_chapter_too_large_422(self, client):
        resp = client.post(
            "/translate",
            json={"text": "x" * 300_000, "format": "html"},
        )
        assert resp.status_code == 422
        assert "max_chapter_characters" in resp.json()["error"]

    def test_crash_returns_500_with_correlation_id(self, client):
        # Swap in a translator that crashes inside the structured path
        from translation_server.runtime import TranslationRuntime, TranslationServerConfig
        from translation_server.app import create_app

        config = TranslationServerConfig()
        config.runtime.warmup_on_start = False
        config.structured.max_chapter_characters = 100
        runtime = TranslationRuntime(config)

        class CrashFake(FakeTranslator):
            def translate_batch_texts(self, texts, source_lang="zh", target_lang="en", max_new_tokens=None):
                raise RuntimeError("gpu oom: internal")

        runtime._translator = CrashFake()
        app = create_app(runtime)
        c = TestClient(app)

        resp = c.post("/translate", json={"text": "<p>加厚防水面料</p>", "format": "html"})
        assert resp.status_code == 500
        data = resp.json()
        assert "error" in data
        assert "gpu oom" not in data["error"]  # no internals leaked
        assert "correlation_id" in data and data["correlation_id"]

    def test_excluded_content_never_sent(self, client):
        resp = client.post(
            "/translate",
            json={
                "text": '<script>var a="中文"</script><p>加厚防水面料</p>',
                "format": "html",
            },
        )
        assert resp.status_code == 200
        out = resp.json()["translation"]
        assert 'var a="中文"' in out


class TestConcurrency:
    def test_semaphore_limits_concurrent_gpu_work(self):
        """concurrency=1: concurrent requests must serialize (including the
        lazy translator acquisition inside the semaphore). Real HTTP server
        so every request shares one event loop (asyncio.Semaphore is
        loop-bound; per-thread TestClients would each get their own loop)."""
        import socket
        import threading
        import time
        import uvicorn

        import httpx
        from translation_server.runtime import TranslationRuntime, TranslationServerConfig
        from translation_server.app import create_app

        # Pick a free port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        config = TranslationServerConfig()
        config.runtime.warmup_on_start = False
        config.structured.concurrency = 1
        runtime = TranslationRuntime(config)

        in_flight = 0
        max_in_flight = 0
        lock = threading.Lock()

        class SlowFake(FakeTranslator):
            def translate_batch_texts(self, texts, source_lang="zh", target_lang="en", max_new_tokens=None):
                nonlocal in_flight, max_in_flight
                with lock:
                    in_flight += 1
                    max_in_flight = max(max_in_flight, in_flight)
                time.sleep(0.3)
                try:
                    return super().translate_batch_texts(
                        texts, source_lang, target_lang, max_new_tokens
                    )
                finally:
                    with lock:
                        in_flight -= 1

        runtime._translator = SlowFake()
        app = create_app(runtime)

        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        try:
            # wait for readiness
            url = f"http://127.0.0.1:{port}"
            with httpx.Client(timeout=30) as c:
                for _ in range(100):
                    try:
                        if c.get(f"{url}/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        time.sleep(0.1)
                else:
                    raise AssertionError("server did not become ready")

                results = {}

                def worker(i):
                    resp = c.post(
                        f"{url}/translate",
                        json={"text": "加厚防水面料", "format": "html"},
                    )
                    results[i] = resp.status_code

                threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
                for wt in threads:
                    wt.start()
                for wt in threads:
                    wt.join(timeout=30)

            assert all(code == 200 for code in results.values())
            assert max_in_flight == 1, f"max concurrent GPU calls: {max_in_flight}"
        finally:
            server.should_exit = True
            t.join(timeout=10)
