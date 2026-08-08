# AGENTS.md — Project conventions for all AI coding agents

## Dependency management

**`environment.yml` is the single source of truth for all dependencies.** There is no `pyproject.toml`, `setup.py`, `requirements.txt`, or `Pipfile` — and there must never be.

- To install/update dependencies: **only** `.\script\script\Initialize-Env.ps1`
- To add a dependency: **edit `environment.yml`**, then re-run `script\Initialize-Env.ps1`
- **Never** run `pip install`, `pip install -e .`, or suggest any pip command
- **Never** create a `pyproject.toml` or any packaging file

## Running the project

This is a `src/`-layout Python CLI — no install needed:

```powershell
$env:PYTHONPATH = ".\src"
python -m image_translation <input-path> [-c <config>]
```

## Testing

```powershell
conda run -n dp python -m pytest tests/ -v
```

`tests/conftest.py` adds `src/` to `sys.path` automatically — no setup beyond `script\Initialize-Env.ps1`.

## Environment

- Conda environment name: `dp`
- Python: 3.10
- CUDA: PyTorch 2.12.1+cu126 (NVIDIA GPU required for translation)

## Translation Server

```powershell
# Start the standalone translation API
.\script\Start-TranslationServer.ps1

# Or with explicit config
.\script\script\Start-TranslationServer.ps1 -Config ".\translation-server.config.json"

# Smoke test (no HTTP)
conda run -n dp python -c "from image_translation.translation import TranslationConfig, create_translator; t = create_translator(TranslationConfig()); print(t.translate_text('你好').translated_text)"
```

The shared translation module lives in `src/image_translation/translation/`.  
The FastAPI host lives in `src/translation_server/`.  
**Never** let the shared module import FastAPI. FastAPI depends on the shared module, not vice versa.
