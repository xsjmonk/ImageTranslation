# Local image tools (conda environment)

The `translate-image` skill uses deterministic local pixel operations from the
ImageTranslation repository. Recognition and translation stay with the agent;
this environment supports masks, inpainting, compositing, and manifest I/O only.

## Environment

- **Single conda environment:** `dp` (declared in `environment.yml`)
- **Initialize / update (idempotent):**

```powershell
cd D:\Drop\outlook.com\LocalBox\ImageTranslation
.\script\Initialize-Env.ps1
```

This creates or updates `dp`, validates every dependency in `environment.yml`,
and runs a local image import smoke test (`python -m image_translation.local_manifest --check-env`).

Do **not** run `pip install` during normal image processing. Add packages only
through `environment.yml`, then re-run `Initialize-Env.ps1`.

## Required packages (local tools)

| Capability | Packages |
|------------|----------|
| Load/save JPG, PNG, WebP, BMP, TIFF | `pillow`, `opencv`, `imageio` |
| Arrays and blur/inpaint math | `numpy`, `scipy`, `scikit-image` |
| Polygon masks | `shapely`, `pyclipper`, `opencv` |
| Manifest validation / CLI | `pydantic`, `jsonschema`, `typer`, `rich` |

Translation/OCR packages in the same environment are unrelated to local manifest
processing and are not initialized by these launchers.

## Command contract

From the repository:

```powershell
.\script\Process-ImageManifest.ps1 check-env
.\script\Process-ImageManifest.ps1 --manifest <manifest.json> [--output <folder>] [--dry-run]
```

From the skill folder:

```powershell
D:\Drop\outlook.com\LocalBox\Code\Skills\ImageTranslation\scripts\process-image-manifest.ps1 check-env
```

Launchers resolve:

- repository root (`IMAGE_TRANSLATION_REPO` override supported);
- `PYTHONPATH=<repo>\src`;
- conda env `dp` (`IMAGE_TRANSLATION_CONDA_ENV` override supported).

## Failure messages

| Problem | Action |
|---------|--------|
| Conda not found | Install Miniconda/Anaconda |
| `dp` missing | Run `.\script\Initialize-Env.ps1` |
| Import check fails | Re-run `Initialize-Env.ps1` after editing `environment.yml` |
| Launcher not found from skill | Set `IMAGE_TRANSLATION_REPO` to the ImageTranslation root |

## Python entry point

```powershell
$env:PYTHONPATH = "D:\Drop\outlook.com\LocalBox\ImageTranslation\src"
conda run -n dp python -m image_translation.local_manifest --check-env
```
