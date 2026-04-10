# Phase 1: Project Scaffold + Image Loading — Summary

**Status:** Complete (merged via PR #1)
**Commits:** 5472cec through ddfe2c3 (9 commits total)
**Date completed:** Prior to 2026-02-15

---

## What Was Built

### 1. Project Infrastructure
- **pyproject.toml** — Modern Python project config with 25+ dependencies, build system, tool configs (ruff, mypy)
- **requirements.txt** — Pinned dependency versions for reproducibility
- **.env.example** — Template for API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY)
- **.gitignore** — Exclusions for .venv, test-images, output, debug, __pycache__, etc.
- **scripts/fetch-test-images.sh** — Downloads test images from GitHub releases

### 2. Full Directory Structure
All module directories created with `__init__.py` files, matching the project structure in CLAUDE.md:
- `src/preprocessing/` — Implemented (loader.py, normalizer.py)
- `src/page_detection/` — Skeleton only
- `src/glare/` — Skeleton only
- `src/photo_detection/` — Skeleton only
- `src/geometry/` — Skeleton only
- `src/color/` — Skeleton only
- `src/ai/` — Skeleton only
- `src/utils/` — Implemented (debug.py)

### 3. Core Modules

#### Image Loader (`src/preprocessing/loader.py`, ~230 lines)
- Universal `load_image(path)` function with auto-detection by file extension
- **HEIC/HEIF** loading via `pillow-heif` with `register_heif_opener()`
- **DNG/RAW** loading via `rawpy` — demosaicing with camera white balance, sRGB output, 16-bit
- **JPEG/PNG/TIFF** loading via PIL
- EXIF orientation handling (all 8 rotation/flip variants)
- All formats normalize to **float32 RGB [0, 1]** numpy arrays
- `ImageMetadata` class tracking original_size, format, bit_depth, orientation

#### Image Normalizer (`src/preprocessing/normalizer.py`, ~97 lines)
- Resize to configurable max working resolution (default 4000px longest edge)
- Uses `cv2.INTER_AREA` for high-quality downscaling
- Generates thumbnails (400px longest edge)
- Tracks scale factor for coordinate mapping in later phases
- `NormalizationResult` dataclass with image, thumbnail, scale_factor

#### Pipeline Orchestrator (`src/pipeline.py`, ~200 lines)
- `PipelineConfig` dataclass with all tunable parameters from CLAUDE.md
- `Pipeline` class with `process()` (single image) and `process_batch()` (multiple)
- Per-step timing tracked in `step_times` dict
- Currently runs 2 steps: load, normalize
- Debug output integration at each step
- Graceful error handling with logging

#### CLI (`src/cli.py`, ~230 lines)
- Click-based with 3 commands: `process`, `check` (placeholder), `compare` (placeholder)
- `process` supports: single file, directory with `--batch`, `--filter` glob, `--debug`, `--steps`, `--verbose`
- Per-file processing summary with format, size, timing
- Overall batch summary at completion

#### Debug Utilities (`src/utils/debug.py`, ~158 lines)
- `save_debug_image()` — saves float32 or uint8 arrays as JPEG with automatic RGB/BGR conversion
- `save_debug_text()` — saves text annotations
- `create_comparison_image()` — side-by-side image comparisons

### 4. Test Suite (`tests/test_loader.py`, ~261 lines)
- `test_load_heic()` — HEIC format loading and metadata verification
- `test_load_dng()` — DNG/ProRAW loading with 16-bit validation
- `test_heic_vs_dng_same_scene()` — Verifies matching pairs produce similar results
- `test_prores_raw_characteristics()` — File size ratio validation (5x+ for DNG vs HEIC)
- `test_all_heic_files_load()` — Batch validation of all 5 HEIC files
- `test_all_dng_files_load()` — Batch validation of all 5 DNG files
- `test_file_not_found()` and `test_unsupported_format()` — Error handling tests
- Tests gracefully skip when test images are not present

---

## Key Design Decisions

1. **Float32 [0, 1] internally** — All image data uses consistent float32 representation. Conversion to/from uint8 happens only at I/O boundaries.

2. **RGB color order everywhere** — Internal processing always uses RGB. BGR conversion only for OpenCV calls, immediately converted back.

3. **Immutable pipeline** — Each step receives input and returns output. No in-place mutation.

4. **Config dataclass** — Single `PipelineConfig` holds all tunable parameters. No magic numbers scattered through code.

5. **Logging over print** — Python `logging` module with DEBUG/INFO/WARNING levels throughout.

6. **Graceful test skipping** — Tests skip (not fail) when test images aren't available, making CI viable.

---

## Learnings

### HEIC Handling
- `pillow-heif` must call `register_heif_opener()` before `Image.open()` — this registers the HEIF format with PIL globally
- iPhone photos store orientation via EXIF tag 274 — must apply rotation before any processing
- 24MP HEIC files load quickly and are the right choice for fast iteration

### DNG/ProRAW Handling
- `rawpy.postprocess()` handles demosaicing, white balance, and gamma correction in one call
- `use_camera_wb=True` applies the white balance the phone calculated at capture — good default
- 48MP DNG files are significantly larger (5-30x) and slower to process but provide 16-bit depth
- Output is 16-bit (0-65535), needs division by 65535.0 for float32 normalization

### Architecture
- Pre-creating all module directories with `__init__.py` files makes future phases easier — no structural changes needed, just implementation
- Per-step timing in the pipeline is valuable for performance profiling as complexity grows
- Debug output at every step is essential — can't improve what you can't see

### Testing
- Testing with real iPhone photos (not synthetic data) catches format-specific edge cases
- Paired HEIC/DNG testing of the same scene validates format-independent behavior
- File existence fixtures allow the same tests to run in CI (skip) and locally (execute)

---

## Statistics

| Metric | Value |
|--------|-------|
| Python files | 14 |
| Lines of code (src/) | ~485 |
| Lines of code (tests/) | ~261 |
| Test cases | 10 |
| Module directories | 8 |
| Commits | 9 |
| Dependencies | 25+ |

---

## What's Next: Phase 2

Phase 2 implements **Page Detection & Perspective Correction** — detecting album page boundaries and applying homographic transforms to correct perspective distortion. This is critical for the album page shots (three_pics, two_pics) which were photographed at angles.
