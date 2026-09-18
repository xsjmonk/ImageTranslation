"""Tests for image-pipeline translator factory."""

from image_translation.config.models import TranslationConfig
from image_translation.translator_factory import create_image_pipeline_translator
from image_translation.translation.translator import NoopTranslator
from image_translation.translation.server_translator import ServerTranslator


def test_factory_returns_noop_when_disabled():
    config = TranslationConfig(enabled=False)
    translator = create_image_pipeline_translator(config)
    assert isinstance(translator, NoopTranslator)


def test_factory_returns_noop_engine():
    config = TranslationConfig(enabled=True, engine="noop")
    translator = create_image_pipeline_translator(config)
    assert isinstance(translator, NoopTranslator)


def test_factory_returns_server_engine():
    config = TranslationConfig(
        enabled=True,
        engine="server",
        server_url="http://127.0.0.1:9000",
    )
    translator = create_image_pipeline_translator(config)
    assert isinstance(translator, ServerTranslator)
