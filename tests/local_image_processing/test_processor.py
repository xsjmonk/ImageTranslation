"""Synthetic end-to-end processing tests for LocalImageProcessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from LocalImageProcessing.contract import BatchManifest, ImageManifest, RegionAction, RegionManifest, sha256_file
from LocalImageProcessing.processor import ProcessingOptions, process_batch
from image_translation.utilities import files


def _write_test_image(path: Path, width: int = 120, height: int = 80) -> None:
    import cv2

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (40, 80, 120)
    cv2.rectangle(image, (20, 20), (100, 50), (255, 255, 255), -1)
    cv2.putText(image, "TEST", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(str(path), image)


def _translate_manifest(source: Path) -> BatchManifest:
    return BatchManifest(
        images=[
            ImageManifest(
                source_path=source,
                source_hash=sha256_file(source),
                regions=[
                    RegionManifest(
                        id="text-1",
                        polygon=[[18, 18], [102, 18], [102, 52], [18, 52]],
                        action=RegionAction.translate,
                        source_text="测试",
                        translated_text="SAMPLE",
                        style={"alignment": "center", "color_rgb": [0, 0, 0]},
                    ),
                    RegionManifest(
                        id="corner",
                        polygon=[[0, 0], [12, 0], [12, 12], [0, 12]],
                        action=RegionAction.preserve,
                        source_text="logo",
                    ),
                ],
            )
        ]
    )


def test_single_image_processing_writes_candidate_and_backup(tmp_path: Path):
    source = tmp_path / "product.jpg"
    output_root = tmp_path / "product_processed"
    _write_test_image(source)

    summary = process_batch(
        _translate_manifest(source),
        ProcessingOptions(output_root=output_root),
    )

    assert summary["counts"]["success"] == 1
    candidate = output_root / "product.jpg"
    preserved = output_root / "original_product.jpg"
    diagnostic = output_root / "metadata" / "product.json"
    assert candidate.is_file()
    assert preserved.is_file()
    assert diagnostic.is_file()
    assert files.bytes_equal(source, preserved)

    before = source.read_bytes()
    assert candidate.read_bytes() != before


def test_preserve_region_pixels_unchanged(tmp_path: Path):
    import cv2

    source = tmp_path / "preserve.png"
    output_root = tmp_path / "preserve_processed"
    _write_test_image(source)
    original = cv2.imread(str(source))

    process_batch(
        _translate_manifest(source),
        ProcessingOptions(output_root=output_root),
    )

    candidate = cv2.imread(str(output_root / "preserve.png"))
    assert np.mean(np.abs(
        original[0:12, 0:12].astype(np.float32) - candidate[0:12, 0:12].astype(np.float32)
    )) < 5.0


def test_missing_translated_text_fails_without_touching_source(tmp_path: Path):
    source = tmp_path / "bad.jpg"
    output_root = tmp_path / "bad_processed"
    _write_test_image(source)
    before = source.read_bytes()

    manifest = BatchManifest.model_construct(
        images=[
            ImageManifest.model_construct(
                source_path=source,
                regions=[
                    RegionManifest.model_construct(
                        id="text-1",
                        polygon=[[18, 18], [102, 18], [102, 52], [18, 52]],
                        action=RegionAction.translate,
                        source_text="测试",
                        translated_text="",
                    )
                ],
            )
        ]
    )

    summary = process_batch(
        manifest,
        ProcessingOptions(output_root=output_root),
    )
    assert summary["counts"]["failure"] == 1
    assert source.read_bytes() == before
    assert not (output_root / "bad.jpg").exists()


def test_blur_fallback_marks_partial_status(tmp_path: Path):
    source = tmp_path / "blur.jpg"
    output_root = tmp_path / "blur_processed"
    _write_test_image(source)

    manifest = BatchManifest(
        images=[
            ImageManifest(
                source_path=source,
                regions=[
                    RegionManifest(
                        id="text-1",
                        polygon=[[18, 18], [102, 18], [102, 52], [18, 52]],
                        action=RegionAction.translate,
                        source_text="测试",
                        translated_text="BLURRED",
                        reconstruction={"method": "blur_fallback", "blur_sigma": 8.0},
                    )
                ],
            )
        ]
    )

    summary = process_batch(manifest, ProcessingOptions(output_root=output_root))
    assert summary["counts"]["partial"] == 1
    diagnostic = json.loads((output_root / "metadata" / "blur.json").read_text(encoding="utf-8"))
    assert any(item["method"] == "blur_fallback" for item in diagnostic["processing_methods"])


def test_promote_replaces_source_after_backup(tmp_path: Path):
    source = tmp_path / "promote.jpg"
    output_root = tmp_path / "promote_processed"
    _write_test_image(source)
    before = source.read_bytes()

    process_batch(
        _translate_manifest(source),
        ProcessingOptions(output_root=output_root, promote=True),
    )

    preserved = output_root / "original_promote.jpg"
    assert preserved.is_file()
    assert preserved.read_bytes() == before
    assert source.read_bytes() != before


def test_cli_process_command(tmp_path: Path):
    source = tmp_path / "cli.jpg"
    output_root = tmp_path / "cli_processed"
    _write_test_image(source)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "contract_version": "1.0",
                "source_path": str(source),
                "regions": [
                    {
                        "id": "text-1",
                        "polygon": [[10, 10], [110, 10], [110, 60], [10, 60]],
                        "action": "translate",
                        "source_text": "测试",
                        "translated_text": "OK",
                        "style": {"alignment": "center", "color_rgb": [0, 0, 0]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    import typer
    from LocalImageProcessing.cli import process_command

    with pytest.raises(typer.Exit) as exc:
        process_command(
            manifest=manifest_path,
            output_folder=output_root,
            promote=False,
            overwrite=False,
        )
    assert exc.value.exit_code == 0
    assert (output_root / "cli.jpg").is_file()
