"""Path resolution and cwd-independent import tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_module_import_from_different_cwd():
    repo = Path(__file__).resolve().parents[2]
    code = (
        "import os; os.chdir('C:/');"
        "import LocalImageProcessing; "
        "from LocalImageProcessing.paths import resolve_repo_root; "
        "root = resolve_repo_root(r'%s'); "
        "assert (root / 'environment.yml').is_file()"
    ) % str(repo).replace("\\", "\\\\")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_resolve_repo_root_from_module_location():
    from LocalImageProcessing.paths import resolve_repo_root

    root = resolve_repo_root()
    assert (root / "LocalImageProcessing" / "README.md").is_file()
