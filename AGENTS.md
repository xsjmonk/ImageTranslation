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

## Architecture skill (all agents)

Before writing or editing any programming code, tests, scripts,
configuration, or infrastructure definition, every coding agent must read
and follow `.agent-skills/reuse-first-architecture/SKILL.md`. The agent must
complete its mandatory pre-coding gate before making edits, even when the
change appears small or trivial.

This gate is mandatory workflow, not optional advice, and applies to:

- feature additions;
- bug fixes;
- refactors;
- tests and test fixtures;
- scripts and configuration;
- library/dependency selection;
- infrastructure and deployment definitions.

The skill enforces: search before building (reuse/adapt/extend/replace
with named-component evidence), SOLID and dependency inversion,
preservation of public contracts and HTML/cache/GPU invariants, and
evidence-based implementation plus verification in one round. Its
references (`references/`) contain the repository-specific architecture
contracts and the verification checklist. Do not duplicate the skill text
in this file; point to the canonical path.
