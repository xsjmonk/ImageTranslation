"""Contract validation tests for LocalImageProcessing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from LocalImageProcessing.contract import (
    BatchManifest,
    RegionAction,
    SingleImageManifest,
    load_manifest,
)


def test_translate_region_requires_translated_text():
    with pytest.raises(ValidationError, match="translated_text"):
        SingleImageManifest.model_validate(
            {
                "source_path": "photo.jpg",
                "regions": [
                    {
                        "id": "r1",
                        "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                        "action": "translate",
                    }
                ],
            }
        )


def test_polygon_requires_three_points():
    with pytest.raises(ValidationError, match="polygon"):
        SingleImageManifest.model_validate(
            {
                "source_path": "photo.jpg",
                "regions": [
                    {
                        "id": "r1",
                        "polygon": [[0, 0], [10, 0]],
                        "action": "preserve",
                    }
                ],
            }
        )


def test_load_manifest_single_image_wrapper():
    manifest = load_manifest(
        {
            "contract_version": "1.0",
            "source_path": "photo.png",
            "regions": [],
        }
    )
    assert isinstance(manifest, BatchManifest)
    assert len(manifest.images) == 1
    assert manifest.images[0].source_path.name == "photo.png"


def test_unsupported_contract_version_rejected():
    with pytest.raises(ValidationError, match="contract_version"):
        BatchManifest.model_validate(
            {
                "contract_version": "9.9",
                "images": [{"source_path": "photo.jpg", "regions": []}],
            }
        )


def test_preserve_region_does_not_require_translation():
    manifest = SingleImageManifest.model_validate(
        {
            "source_path": "photo.jpg",
            "regions": [
                {
                    "id": "logo",
                    "polygon": [[0, 0], [20, 0], [20, 20], [0, 20]],
                    "action": RegionAction.preserve.value,
                    "source_text": "BRAND",
                }
            ],
        }
    )
    assert manifest.regions[0].action == RegionAction.preserve
