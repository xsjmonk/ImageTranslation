# Image Translation

Modular Python CLI for translating Chinese text in Amazon listing images.

## Quick Start

```powershell
# Ensure the conda environment is ready (run once)
.\script\Initialize-Env.ps1

# Set PYTHONPATH so the package is importable
$env:PYTHONPATH = ".\src"

# Process a folder of images
python -m image_translation "D:\Products\ABC\original_photos"

# Process a single image
python -m image_translation "D:\Products\ABC\original_photos\01.jpg"

# With explicit config
python -m image_translation "D:\Products\ABC\original_photos" -c ".\config.json"
python -m image_translation "D:\Products\ABC\original_photos" --config ".\config.json"
```

Or using `conda run`:

```powershell
conda run -n dp env PYTHONPATH=".\src" python -m image_translation "D:\Products\ABC\original_photos"
```

## Setup

```powershell
# Create/update the conda environment — the only dependency step needed
.\script\Initialize-Env.ps1
```

All dependencies are declared in `environment.yml`. `script\Initialize-Env.ps1` is the single source of truth for ensuring the `dp` conda environment is ready.

## Output Structure

Input:
```
src\original_photos\
    01.jpg
    02.png
```

Output:
```
src\original_photos_processed\
    01.jpg
    02.png

    metadata\
        01.json
        02.json

    masks\
        01.png
        02.png

    cleaned\
        01.png
        02.png
```

- Folder input → `<folder>_processed` in the same parent directory.
- Single image → `<parent>_processed\<filename>`.
- Original files are never modified.

## Configuration

Copy `config.example.json` to `config.json` in the repo root, or pass `-c <path>`.

The config is optional — built-in defaults are used when no config file is present.

Key settings:
- `ocr.min_confidence` (0–1): filter low-confidence text
- `translation.preserve_terms`: brand names to keep untranslated
- `translation.preserve_patterns`: regex patterns for text to preserve
- `input.recursive`: scan subfolders (excludes `_processed` dirs)
- `output.overwrite_existing`: re-process images with existing output

## Architecture

```
src/image_translation/
    cli.py              # CLI entry point
    pipeline.py          # Orchestration only
    __main__.py          # python -m entry

    input/               # CLI args, path resolution, validation
    config/              # JSON config loading + pydantic models
    ocr/                 # OcrEngine interface + PaddleOCR (lazy-loaded)
    translation/         # Translator interface + classifier + noop placeholder
    imaging/             # ImageProcessor interface + mask + inpainting
    revision/            # Layout + text rendering + compositing
    models/              # TextRegion, ImageJob, ProcessingResult
    utilities/           # files, folders, json_utils, images
```

Dependency boundaries:
- Pipeline depends on abstractions (interfaces), not concrete libraries.
- Heavy models (PaddleOCR) are lazy-loaded on first use.
- Constructor injection — no DI framework.

## Tests

```powershell
conda run -n dp python -m pytest tests/ -v
```

Tests cover input resolution, config validation, enumeration, JSON utilities, classifier, models, mask generation, translation module, and FastAPI server — no real OCR/GPU models required for unit tests.

## Translation Server (standalone GPU API)

A standalone FastAPI server exposes the M2M100 zh→en translator over HTTP.

### Start the server

```powershell
.\script\Start-TranslationServer.ps1
```

With explicit config:

```powershell
.\script\script\Start-TranslationServer.ps1 -Config ".\translation-server.config.json"
```

The first launch downloads the model (~1.7 GB). Subsequent starts are fast.  
**Requires NVIDIA CUDA GPU** — defaults to `cuda:0` with no CPU fallback.

### API

**Health check:**

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8091/health' -Method Get
```

**Translate:**

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8091/translate' -Method Post -ContentType 'application/json; charset=utf-8' -Body (@{ text='加厚防水面料' } | ConvertTo-Json)
```

Response: `{"translation": "Thickened waterproof fabric"}`

### Smoke test (direct module, no HTTP)

```powershell
conda run -n dp python -c "from image_translation.translation import TranslationConfig, create_translator; t = create_translator(TranslationConfig()); print(t.translate_text('你好').translated_text)"
```

### Architecture

```
               shared translation module
                 /                   \
                ↓                     ↓
   ImageTranslation tool      FastAPI translation app
```

The shared module owns translation. FastAPI owns HTTP transport only.
