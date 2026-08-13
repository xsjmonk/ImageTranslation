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

With explicit config (relative paths resolve from the current directory):

```powershell
.\script\Start-TranslationServer.ps1 -Config ".\translation-server.config.json"
```

The first launch downloads the model (~1.7 GB). Subsequent starts are fast.  
**Requires NVIDIA CUDA GPU** — defaults to `cuda:0` with no CPU fallback.

### Runtime configuration

`translation-server.config.json`:

```json
"runtime": {
  "warmup_on_start": true
}
```

- `warmup_on_start: true` (default) — model loads before the API becomes ready; startup fails if the model cannot load.
- `warmup_on_start: false` — model loads lazily on the first `/translate` request; `/health` reports `"status": "starting", "ready": false` until then.

### Long-running requests

Translation may take several seconds. The HTTP request stays open until inference completes — there is **no server-side timeout**. GPU inference runs in the threadpool and never blocks the event loop, so `/health` stays responsive. Callers must use a sufficiently long HTTP client timeout (PowerShell's `Invoke-RestMethod` waits by default).

### API

**Health check:**

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8091/health' -Method Get
```

**Translate:**

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8091/translate' -Method Post -ContentType 'application/json; charset=utf-8' -Body (@{ text='加厚防水面料' } | ConvertTo-Json)
```

Response: `{"translation": "Increased waterproof."}`

Errors are JSON envelopes (`{"error": "...", "correlation_id": "..."}`): invalid input → 400, translator/model unavailable → 503, unexpected failure → 500.

### HTML-aware translation (`format: "html"`)

Long chapters with mixed Chinese/English text and HTML markup are translated
structurally — tags, attributes, comments, scripts, and excluded regions are
preserved; only translatable text content changes:

```powershell
$body = @{
    text    = '<p>这是一款 <strong>加厚防水面料</strong>，适合 daily use。</p>'
    format  = 'html'
} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8091/translate' -Method Post `
    -ContentType 'application/json; charset=utf-8' -Body $body
```

Behavior contract (documented):
- `format` is optional and defaults to `"plain"` — existing callers are unaffected; no HTML auto-detection.
- HTML is parsed with html5lib (reference HTML5 parser; no regex) in fragment mode. Malformed input is **deterministically normalized** (stray end tags dropped, implied end tags applied, void tags canonicalized, character references decoded). Guarantee: *semantic round-trip* — tag nesting, attributes, comments, doctype (kept verbatim as a prefix), text, whitespace, and excluded subtrees are preserved as data; byte-for-byte form is not claimed. A leading `<!DOCTYPE ...>` is preserved verbatim.
- Text is always re-escaped on output (`& < >`), so markup produced by the model can never inject tags.
- **Protected-span translation**: every text node is split into explicit runs. Chinese runs are translated; ordinary English, identifiers, URLs, and inline tags are replaced with collision-resistant placeholders BEFORE inference and restored from the ORIGINAL text afterwards — the model can never rewrite, drop, or paraphrase English, so exact preservation does not depend on post-hoc detection. English split around `<strong>/<em>/<span>` is restored exactly; whitespace between CJK spans stays inside the translated run so spacing renders naturally.
- Blocks that are entirely English-only are never sent to the model (byte-identical).
- `translate="no"` attributes, `notranslate` classes, and `<script>/<style>/<code>/<pre>` subtrees are never sent to the model; their content must be byte-identical after translation or the request fails closed.
- Chapters larger than 4,000 characters are segmented (sentence → clause → hard split) with a real tokenizer (truncation=False); nothing is silently truncated — over-budget plain input is rejected with a clear error.
- Every placeholder must appear exactly once, in source order (all kinds: tags, identifiers, English). Failures retry with a stricter placeholder prefix, then fall back to per-chinese-run translation; if that still fails, the request errors rather than returning corrupted HTML.
- Reconstruction is driven by explicit run metadata (node id, kind, raw text, character offsets, slot index) with build-time coverage invariants (concatenated runs == block text; offsets contiguous; every node covered exactly once); model-output pieces map onto recorded slots, never by string splitting.
- The request's `source_language`/`target_language` propagate to every model call (`forced_bos_token_id` is resolved from the target language); language pairs are not hardcoded.
- `translatable_attributes` values are translated as segments; URL/code/style attributes are never in the allowlist and are never translated.

Structured configuration (`structured` section in `translation-server.config.json`):

```json
"structured": {
  "max_chapter_characters": 100000,
  "max_segment_tokens": 450,
  "max_target_tokens": 400,
  "context_window_tokens": 0,
  "glossary": [
    {"source": "充电器", "target": "Charger", "exact": true},
    {"source": "防水面料", "target": "Waterproof Fabric", "exact": true}
  ],
  "translatable_attributes": [],
  "excluded_tags": ["script", "style", "code", "pre"],
  "excluded_classes": ["notranslate"],
  "segment_warning_seconds": 60.0,
  "max_total_seconds": 600.0,
  "max_retries_per_segment": 1,
  "concurrency": 1
}
```

Notes:
- `context_window_tokens: 0` — context injection is **not implemented** (M2M100's `generate()` has no reliable context API) and this value is reserved; it MUST stay 0. Segment adjacency is recorded for diagnostics only, not as context. Terminology consistency across segments is guaranteed for protected terms (identifiers/English restore identical text); for translatable Chinese terms it depends on model determinism (fixed beams → deterministic output for identical input).
- **Terminology memory (glossary)**: a chapter-scoped terminology map. Configured entries (`{"source": "充电器", "target": "Charger", "exact": true}`) are replaced with protected placeholders BEFORE inference and restored to the exact target term afterwards — the same term maps to the same target in EVERY segment (consistent by construction; the model can never paraphrase it). `exact: true` never matches inside a latin word (`cat` ≠ `catalog`); CJK ideograph neighbors are accepted (Chinese has no spaces). Protected identifiers (URLs, codes) always win over glossary terms. After reconstruction, a consistency check counts target occurrences and fails closed on any mismatch.
- Repeated Chinese terms (CJK bigrams/trigrams, ≥3 occurrences) are collected and reported in the metrics for review; they are informational — only configured glossary entries drive replacement.
- `context_window_tokens` is **explicitly unsupported**: non-zero values raise a configuration error. M2M100's `generate()` has no reliable context API; segment adjacency ids are diagnostics only and are never sent to the model.
- Machine-readable metrics: the structured result exposes `to_dict()` (segment count, source tokens, target budgets, protected-run count, terminology occurrences with segment ids, repeated terms, elapsed time, retry/fallback counts, validation status) — emitted as JSON by the GPU quality gate.
- `translatable_attributes` is an allowlist of human-readable attribute values that may be translated, e.g. `["alt", "title", "aria-label"]` (empty = none).
- `segment_warning_seconds` is a **warning threshold only** — a slow segment logs a warning but is never cancelled.
- `max_total_seconds` is a **real deadline** enforced between segments — an in-flight segment is not interruptible; once it returns, work stops and the request fails with a clear error and no partial output.
- Concurrent translation requests are bounded by `concurrency` (GPU holds one model); the bound covers lazy model loading too.

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
