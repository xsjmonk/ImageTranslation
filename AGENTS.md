# AGENTS.md — Project conventions for all AI coding agents

## Dependency management

**`environment.yml` is the single source of truth for all dependencies.** There is no `pyproject.toml`, `setup.py`, `requirements.txt`, or `Pipfile` — and there must never be.

- To install/update dependencies: **only** `.\script\Initialize-Env.ps1`
- To add a dependency: **edit `environment.yml`**, then re-run `.\script\Initialize-Env.ps1`
- **Never** run `pip install`, `pip install -e .`, or suggest any pip command
- **Never** create a `pyproject.toml` or any packaging file

## Running the project

This is a `src/`-layout Python CLI — no install needed:

```powershell
$env:PYTHONPATH = ".\src"
python -m image_translation <input-path> [-c <config>]
```

## Testing

Default unit suite (fast, mocked — no GPU model load):

```powershell
conda run -n dp python -m pytest tests/ -v
```

This skips GPU integration tests unless you opt in with environment variables.
On a CUDA machine, the full suite used to load real NLLB/Hy-MT2 weights and
could take 20+ minutes; the default run should finish in a few minutes.

Opt-in GPU/integration runs:

```powershell
$env:RUN_QUALITY_REGRESSION = "1"   # tests/translation/test_quality_regression.py
$env:RUN_HTML_GPU_QUALITY = "1"    # tests/translation/test_html_gpu_quality.py
$env:RUN_NLLB_SMOKE = "1"          # tests/translation/smoke_test.py
$env:RUN_HYMT2_SMOKE = "1"         # tests/translation/test_hymt2_smoke.py
$env:RUN_HYMT2_CACHED_SMOKE = "1"  # tests/translation/test_hymt2_cached_snapshot_smoke.py
```

Focused translation-server tests (~17 seconds):

```powershell
conda run -n dp python -m pytest tests/translation_server/ -v
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

## Local image processing (translate-image skill)

Deterministic pixel work lives in `LocalImageProcessing/` at the repository
root. The host agent performs recognition and translation; the module handles
manifest validation, masking, inpainting, rendering, diagnostics, and atomic
outputs. It does **not** OCR, translate, load models, or call remote services.

```powershell
$env:PYTHONPATH = "."
conda run -n dp python -m LocalImageProcessing process --manifest manifest.json -o .\work_processed
conda run -n dp python -m LocalImageProcessing check-env
```

`LocalImageProcessing` is self-contained and does not import `image_translation`.
Tests use the writable `.pytest_tmp/` directory when the system temp folder is inaccessible.

Environment setup and read-only dependency checks:

```powershell
.\script\Initialize-Env.ps1
.\script\Check-LocalImageEnv.ps1
```

Skill wrappers:

```powershell
D:\Drop\outlook.com\LocalBox\Code\Skills\ImageTranslation\scripts\check-local-image-env.ps1
D:\Drop\outlook.com\LocalBox\Code\Skills\ImageTranslation\scripts\translate-image.ps1 -Manifest manifest.json
```

Authoritative contract: `LocalImageProcessing/README.md`  
Overview: `docs/local-image-tools.md`

## Architecture skill (all agents)

Before writing or editing code, tests, scripts, configuration, or
infrastructure, read and follow the canonical skill at
`.agent-skills/reuse-first-architecture/SKILL.md`. Do not copy its contents.

This gate is mandatory for every coding agent and every edit type, including
feature additions, bug fixes, refactors, tests, scripts, configuration,
library/dependency selection, and infrastructure definitions — even when the
change appears small or trivial.
