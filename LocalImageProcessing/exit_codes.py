"""Process exit-code policy for batch summaries."""

from __future__ import annotations

from typing import Dict


def exit_code_from_summary(summary: Dict) -> int:
    """Map a batch summary to a process exit code.

    - 1: at least one hard failure
    - 2: no failures, but partial/review outcomes remain
    - 0: all images succeeded or were skipped
    """
    counts = summary.get("counts", {})
    if counts.get("failure", 0) > 0:
        return 1
    if counts.get("partial", 0) > 0 or counts.get("review", 0) > 0:
        return 2
    return 0
