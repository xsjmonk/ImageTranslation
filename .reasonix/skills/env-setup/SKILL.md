---
name: env-setup
description: Enforce environment.yml as the single source of truth for dependencies — no pyproject.toml, no pip install, only Initialize-Env.ps1.
---

# environment.yml — single source of truth

This project uses Conda. **`environment.yml`** is the single source of truth for all dependencies. **`Initialize-Env.ps1`** is the only way to ensure dependent packages are installed.

## Hard rules

1. **Never** create, edit, or reference a `pyproject.toml`, `setup.py`, `setup.cfg`, `requirements.txt`, `Pipfile`, or any other dependency file. `environment.yml` is the only dependency declaration.

2. **Never** run `pip install` directly or suggest it. To install or update dependencies, always use:

   ```powershell
   .\Initialize-Env.ps1
   ```

3. **Never** run `pip install -e .` or any editable/package install. The `src/` layout does not need installation — use `PYTHONPATH`:

   ```powershell
   $env:PYTHONPATH = ".\src"
   ```

4. When adding a new Python dependency, **edit `environment.yml`** (under `dependencies:` for conda packages, or under `- pip:` for pip-only packages). Then instruct the user to re-run `Initialize-Env.ps1`.

5. To run tests, use:

   ```powershell
   conda run -n dp python -m pytest tests/ -v
   ```

   `tests/conftest.py` handles adding `src/` to `sys.path` — no install step needed.

6. To run the tool:

   ```powershell
   $env:PYTHONPATH = ".\src"
   python -m image_translation <input-path> [-c <config>]
   ```

## Rationale

- `environment.yml` + `Initialize-Env.ps1` provides a single, reproducible Conda environment that handles both conda and pip packages, with validation of every dependency version.
- `Initialize-Env.ps1` is idempotent — safe to run repeatedly.
- No `pyproject.toml` keeps the project free of packaging concerns it doesn't need (this is a CLI tool, not a library).
