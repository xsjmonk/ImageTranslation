"""Tests for shared models – TextRegion, ImageJob, ProcessingResult."""

from __future__ import annotations

from pathlib import Path

from image_translation.models.image_job import ImageJob, JobStatus
from image_translation.models.processing_result import ProcessingResult
from image_translation.models.text_region import TextAction, TextRegion


class TestTextRegion:
    def test_create_and_serialize(self):
        region = TextRegion(
            id="text_001",
            source_text="加厚升级",
            confidence=0.98,
            polygon=[[100, 200], [500, 230], [490, 320], [90, 290]],
            action=TextAction.translate,
            action_reason="source_language",
            translation={"translated_text": "Upgraded, Thicker Design"},
        )
        d = region.to_dict()
        assert d["id"] == "text_001"
        assert d["source_text"] == "加厚升级"
        assert d["action"] == "translate"
        assert d["translation"]["translated_text"] == "Upgraded, Thicker Design"

    def test_roundtrip(self):
        region = TextRegion(
            id="r1",
            source_text="hello",
            confidence=0.5,
            polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
        )
        data = region.to_dict()
        region2 = TextRegion.from_dict(data)
        assert region2.id == region.id
        assert region2.source_text == region.source_text
        assert region2.confidence == region.confidence


class TestImageJob:
    def test_status_flow(self, tmp_path: Path):
        job = ImageJob(
            source_path=tmp_path / "in.jpg",
            output_path=tmp_path / "out.jpg",
        )
        assert job.status == JobStatus.pending
        job.status = JobStatus.processing
        job.mark_completed()
        assert job.status == JobStatus.completed
        assert job.error is None

    def test_mark_failed(self, tmp_path: Path):
        job = ImageJob(
            source_path=tmp_path / "in.jpg",
            output_path=tmp_path / "out.jpg",
        )
        job.mark_failed("OCR engine crashed")
        assert job.status == JobStatus.failed
        assert job.error == "OCR engine crashed"

    def test_to_dict(self, tmp_path: Path):
        job = ImageJob(
            source_path=tmp_path / "in.jpg",
            output_path=tmp_path / "out.jpg",
            image_width=1600,
            image_height=1200,
        )
        job.text_regions = [
            TextRegion(
                id="t1",
                source_text="test",
                confidence=0.9,
                polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
                action=TextAction.translate,
            )
        ]
        d = job.to_dict()
        assert d["width"] == 1600
        assert d["height"] == 1200
        assert len(d["regions"]) == 1


class TestProcessingResult:
    def test_summary(self):
        pr = ProcessingResult(total=4, succeeded=3, failed=1, skipped=0)
        s = pr.summary()
        assert "Processed: 4" in s
        assert "Succeeded: 3" in s
        assert "Failed: 1" in s

    def test_has_failures(self):
        pr = ProcessingResult(failed=1)
        assert pr.has_failures
        pr2 = ProcessingResult()
        assert not pr2.has_failures
