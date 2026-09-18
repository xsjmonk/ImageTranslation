"""CLI exit-code policy tests."""

from __future__ import annotations

import pytest
import typer

from LocalImageProcessing.exit_codes import exit_code_from_summary


def test_failure_returns_exit_code_one():
    assert exit_code_from_summary({"counts": {"failure": 1, "partial": 0, "review": 0}}) == 1


def test_partial_returns_exit_code_two():
    assert exit_code_from_summary({"counts": {"failure": 0, "partial": 1, "review": 0}}) == 2


def test_review_returns_exit_code_two():
    assert exit_code_from_summary({"counts": {"failure": 0, "partial": 0, "review": 1}}) == 2


def test_success_returns_zero():
    assert exit_code_from_summary({"counts": {"failure": 0, "partial": 0, "review": 0}}) == 0


def test_cli_failure_exits_nonzero(tmp_path):
    import json

    from LocalImageProcessing.cli import process_command

    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text(
        json.dumps(
            {
                "contract_version": "1.0",
                "source_path": str(tmp_path / "missing.jpg"),
                "regions": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(typer.Exit) as exc:
        process_command(
            manifest=manifest_path,
            output_folder=tmp_path / "out",
            promote=False,
            overwrite=False,
        )
    assert exc.value.exit_code == 1
