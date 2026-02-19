# CLAUDE.md — Sunday Album Processing Engine

## Project Overview

Sunday Album is a free tool that digitizes physical photo album pages into clean individual digital photos. This repository contains the **local processing engine** — a Python CLI that takes album page photos as input and produces corrected, split individual photos as output.

This is the foundational image processing pipeline. No web UI, no cloud deployment — just raw image quality. Get the processing right first, then wrap infrastructure around it.

## Development Machine

- **MacBook Air M4, 24 GB RAM**
- **macOS Tahoe 26.2**
- **Apple Silicon (ARM64)** — all native dependencies must be ARM-compatible
- **Homebrew** is installed and should be used for system-level dependencies (opencv, libheif, libraw, etc.)
- **Python via venv** — use a virtual environment, never install globally

## What This Code Does

Takes a photo of a physical album page (shot with a phone or webcam) and:
1. **Detects and removes glare** from glossy plastic album sleeves (Priority 1)
2. **Detects and splits multiple photos** on a single album page into individual images (Priority 2)
3. **Corrects perspective/keystone distortion** from angled capture (Priority 3)
4. **Restores color** — fixes fading, yellowing, contrast loss from aged prints (Priority 4)

## Test Images

Located in `test-images/`. Two formats per scene — 24MP HEIC (normal) and 48MP DNG (ProRes):

### Individual Photos (glossy finish, single photo per shot)
These are photos of **individual prints with a glossy finish** — they exhibit glare from the contours/curvature of the glossy paper surface. No plastic sleeves, no multi-photo album pages.

| File | Format | Resolution | Description |
|------|--------|-----------|-------------|
| `IMG_cave_normal.HEIC` | HEIC | 24 MP | Cave photo, glossy finish with contour glare |
| `IMG_cave_prores.DNG` | DNG | 48 MP | Same scene, ProRes RAW |
| `IMG_harbor_normal.HEIC` | HEIC | 24 MP | Harbor photo, glossy finish with contour glare |
| `IMG_harbor_prores.DNG` | DNG | 48 MP | Same scene, ProRes RAW |
| `IMG_skydiving_normal.HEIC` | HEIC | 24 MP | Skydiving photo, glossy finish with contour glare |
| `IMG_skydiving_prores.DNG` | DNG | 48 MP | Same scene, ProRes RAW |

### Album Pages (glossy plastic sleeves, multiple photos)
These are photos of **album pages with glossy plastic sleeves** — the primary use case. Multiple photos per page, glare from the plastic sleeve.

| File | Format | Resolution | Description |
|------|--------|-----------|-------------|
| `IMG_three_pics_normal.HEIC` | HEIC | 24 MP | Album page with 3 photos behind plastic sleeve |
| `IMG_three_pics_prores.DNG` | DNG | 48 MP | Same page, ProRes RAW |
| `IMG_two_pics_vertical_horizontal_normal.HEIC` | HEIC | 24 MP | Album page with 2 photos (one portrait, one landscape) behind plastic sleeve |
| `IMG_two_pics_vertical_horizontal_prores.DNG` | DNG | 48 MP | Same page, ProRes RAW |

### What to Expect

- **HEIC files** (24MP): Standard iPhone 17 Pro photos. Need `pillow-heif` or `pyheif` to read. These are the "normal user" input format.
- **DNG files** (48MP ProRes): Apple ProRAW. Much larger, 12-bit or 14-bit depth, linear color space. Need `rawpy` (which wraps LibRaw) to read. These give us maximum data to work with for quality comparisons.
- **Glare patterns differ**: Individual glossy prints have curved/contour glare (from the photo surface itself bowing). Album pages with plastic sleeves have flatter, broader glare patches (from the flat plastic surface).
- **The two album page shots are the critical test cases** — multi-photo splitting + glare removal together. If these look great, the product works.

### How to retrieve 
Use `scripts/fetch-test-images.sh` which will download from github release

## Tech Stack

- **Python 3.12+** (via venv)
- **System dependencies via Homebrew:**
  - `opencv` — core image processing (install: `brew install opencv`)
  - `libheif` — HEIC decoding (install: `brew install libheif`)
  - `libraw` — DNG/RAW decoding (install: `brew install libraw`)
  - `imagemagick` — format conversion fallback (install: `brew install imagemagick`)
- **Python packages (pip in venv):**
  - `opencv-python-headless` — OpenCV Python bindings
  - `numpy`, `scipy` — array ops, interpolation, optimization
  - `scikit-image` — advanced image processing (CLAHE, morphology, segmentation)
  - `Pillow` — image I/O, format handling, EXIF
  - `pillow-heif` — HEIC/HEIF format support for Pillow
  - `rawpy` — RAW/DNG file reading (wraps LibRaw)
  - `click` — CLI framework
  - `python-dotenv` — environment variable management
  - `matplotlib` — debug visualizations
  - `anthropic` — Claude vision API
  - `openai` — GPT-4o vision API (fallback)
  - `pytest`, `mypy`, `ruff` — testing, type checking, linting

## Setup Instructions

```bash
# Install system dependencies via Homebrew
brew install opencv libheif libraw imagemagick

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Or install as editable package
pip install -e ".[dev]"

# Verify HEIC and DNG support
python -c "from pillow_heif import register_heif_opener; print('HEIC OK')"
python -c "import rawpy; print('DNG/RAW OK')"
python -c "import cv2; print(f'OpenCV {cv2.__version__} OK')"
```

## Project Structure

```
sundayalbum-claude/
├── CLAUDE.md                    # This file
├── pyproject.toml               # Project config and dependencies
├── requirements.txt             # Pinned dependencies
├── secrets.json                 # API keys for OPENAI and ANTHROPIC (gitignored)
├── .gitignore
│
├── docs/
│   ├── PRD_Album_Digitizer.md              # Product requirements doc
│   ├── PHASED_PLAN_Claude_Code.md          # Phased plan of implementation starting with a local image processing engine
│   ├── Implementation_Album_Digitizer.md   # Implementation guide for the full project
│   ├── UI_Design_Album_Digitizer.md        # UI Design for the full project
│   ├── IMG_harbor_prores.DNG
│
├── journal/                     # Phase summaries and development journal
│   ├── phase-1-summary.md       # Phase 1: Project Setup
│   ├── phase-2-summary.md       # Phase 2: Image Loading
│   ├── phase-3-summary.md       # Phase 3: Glare Detection
│   ├── phase-4-summary.md       # Phase 4: Glare Removal
│   ├── phase-6-summary.md       # Phase 6: Photo Detection Improvements
│   ├── phase-7-summary.md       # Phase 7: Geometry Correction
│   ├── phase-8-summary.md       # Phase 8: Color Restoration
│   ├── phase-9-summary.md       # Phase 9: Pipeline Integration
│   ├── 2026-02-16-single-print-fix.md     # Fix: single-print false splits
│   ├── 2026-02-17-multi-photo-detection-fix.md  # Fix: three_pics detection
│   └── 2026-02-18-openai-glare-and-orientation.md  # OpenAI glare + AI orientation
│
├── scripts/
│   └── fetch-test-images.sh     # Download test images from GitHub releases
│
├── test-images/                 # Real test images (gitignored — large files)
│   ├── IMG_cave_normal.HEIC
│   ├── IMG_cave_prores.DNG
│   ├── IMG_harbor_normal.HEIC
│   ├── IMG_harbor_prores.DNG
│   ├── IMG_skydiving_normal.HEIC
│   ├── IMG_skydiving_prores.DNG
│   ├── IMG_three_pics_normal.HEIC
│   ├── IMG_three_pics_prores.DNG
│   ├── IMG_two_pics_vertical_horizontal_normal.HEIC
│   └── IMG_two_pics_vertical_horizontal_prores.DNG
│
├── src/
│   ├── __init__.py
│   ├── cli.py                   # CLI entry point (click-based)
│   ├── pipeline.py              # Main pipeline orchestrator
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── loader.py            # Image loading — HEIC, DNG, JPEG, PNG support
│   │   └── normalizer.py        # Resize, orientation fix, thumbnail generation
│   │
│   ├── page_detection/
│   │   ├── __init__.py
│   │   ├── detector.py          # Detect album page boundaries in image
│   │   └── perspective.py       # Homographic transform to fronto-parallel view
│   │
│   ├── glare/
│   │   ├── __init__.py
│   │   ├── detector.py          # Detect glare/reflection regions (used by OpenCV path)
│   │   ├── remover_single.py    # Single-shot glare inpainting (OpenCV fallback)
│   │   ├── remover_multi.py     # Multi-shot glare compositing (not yet integrated)
│   │   ├── remover_openai.py    # OpenAI gpt-image-1.5 glare removal (default path)
│   │   └── confidence.py        # Glare removal confidence scoring
│   │
│   ├── photo_detection/
│   │   ├── __init__.py
│   │   ├── detector.py          # Detect individual photo boundaries
│   │   ├── splitter.py          # Extract individual photo crops
│   │   └── classifier.py        # Classify regions (photo vs decoration vs caption)
│   │
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── keystone.py          # Per-photo perspective correction
│   │   ├── dewarp.py            # Bulge/curvature correction
│   │   └── rotation.py          # Rotation detection and correction
│   │
│   ├── color/
│   │   ├── __init__.py
│   │   ├── white_balance.py     # Auto white balance correction
│   │   ├── restore.py           # Fade restoration (CLAHE, saturation)
│   │   ├── deyellow.py          # Yellowing removal
│   │   └── enhance.py           # Sharpening, contrast, final touches
│   │
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── claude_vision.py     # Anthropic Claude API integration
│   │   ├── openai_vision.py     # OpenAI GPT-4o API integration
│   │   └── quality_check.py     # AI-powered quality assessment
│   │
│   └── utils/
│       ├── __init__.py
│       ├── debug.py             # Debug visualization helpers
│       ├── metrics.py           # Image quality metrics (SSIM, sharpness, etc.)
│       ├── io.py                # File I/O helpers
│       └── secrets.py           # Load API keys from secrets.json
│
├── tests/
│   ├── __init__.py
│   ├── test_loader.py           # Test HEIC, DNG, JPEG loading
│   ├── test_glare.py
│   ├── test_photo_detection.py
│   ├── test_geometry.py
│   ├── test_color.py
│   └── test_pipeline.py
│
├── output/                      # Default output directory (gitignored)
├── debug/                       # Debug visualizations (gitignored)
└── models/                      # Pre-trained model weights (gitignored)
```

## CLI Interface

The CLI provides three main commands: `status`, `process`, and (planned) quality checking.

### Status Command

Check pipeline implementation progress:

```bash
# Show pipeline status and progress
python -m src.cli status
```

**Shows:**
- All 14 pipeline steps with implementation status
- Progress by priority (Glare, Splitting, Geometry, Color)
- Detailed descriptions of each step
- Comprehensive usage examples

**Current Progress (as of 2026-02-19):**
- **Overall:** All major pipeline steps implemented and producing quality output
- **Priority 1 (Glare):** OpenAI `gpt-image-1.5` removal is the default path. OpenCV inpainting is the fallback when no OpenAI key is available. Multi-shot compositing deferred (needs multi-angle images).
- **Priority 2 (Splitting):** 2/2 steps complete. GrabCut page detection + contour photo detection working on all test images.
- **Priority 3 (Geometry):** AI orientation correction (Step 4.5) handles gross 90°/180°/270° errors per-photo. Small-angle Hough-line rotation disabled (fires on image content; border-based replacement pending).
- **Priority 4 (Color):** 4/4 steps complete (white balance, deyellowing, CLAHE fade restore, sharpening).

### Process Command

Process album pages and extract individual photos:

```bash
# Process a single album page (HEIC) — OpenAI glare removal on by default
python -m src.cli process test-images/IMG_three_pics_normal.HEIC --output ./output/

# Process the high-res DNG version
python -m src.cli process test-images/IMG_three_pics_prores.DNG --output ./output/

# Process with debug visualizations
python -m src.cli process test-images/IMG_three_pics_normal.HEIC --output ./output/ --debug

# Process all test images in batch mode
python -m src.cli process test-images/ --output ./output/ --batch

# Process only HEIC files (skip DNG for faster iteration)
python -m src.cli process test-images/ --output ./output/ --batch --filter "*.HEIC"

# Fall back to OpenCV inpainting (no OpenAI API call)
python -m src.cli process test-images/IMG_harbor_normal.HEIC --output ./output/ --no-openai-glare

# Provide an explicit scene description (skips Claude description, still does orientation)
python -m src.cli process test-images/IMG_cave_normal.HEIC --output ./output/ \
  --scene-desc "A cave interior with warm amber light"

# Disable AI orientation correction
python -m src.cli process test-images/IMG_harbor_normal.HEIC --output ./output/ --no-ai-orientation

# Run only specific pipeline steps
python -m src.cli process test-images/IMG_cave_normal.HEIC --output ./output/ --steps load,normalize,page_detect
python -m src.cli process test-images/IMG_cave_normal.HEIC --output ./output/ --steps photo_detect,ai_orientation

# Available step IDs:
# load, normalize, page_detect, photo_detect, ai_orientation, glare_detect,
# keystone_correct, dewarp, rotation_correct, white_balance, color_restore, deyellow, sharpen
```

### Quality Check Commands (Planned)

```bash
# Quality check using AI vision
python -m src.cli check output/photo_01.jpg --original test-images/IMG_cave_normal.HEIC

# Compare before/after
python -m src.cli compare test-images/IMG_cave_normal.HEIC output/photo_01.jpg --save comparison.jpg
```

## Pipeline Flow

```
Input Image (HEIC, DNG, JPEG, or PNG)
    │
    ▼
[1. Load & Preprocess]
    - Detect format: HEIC → pillow-heif, DNG → rawpy, JPEG/PNG → Pillow
    - For DNG: apply basic demosaicing, convert from linear to sRGB gamma
    - Read & apply EXIF orientation
    - Generate working copy (cap 4000px longest edge for pipeline)
    │
    ▼
[2. Page Detection]
    - GrabCut segmentation to find album page boundary quadrilateral
    - Apply perspective correction (homographic warp to fronto-parallel view)
    - If no clear boundary found, treat full frame as the subject
    │
    ▼
[3. Photo Detection & Splitting]  ← PRIORITY 2
    - Contour-based detection of individual photo boundaries on the page
    - Album pages (three_pics, two_pics): split into N individual crops
    - Single prints (cave, harbor, skydiving): treat entire page as one photo
    - Each crop is perspective-corrected using its detected corner quadrilateral
    │
    ▼
[4.5. AI Orientation Correction]  ← per extracted photo
    - One Claude Haiku API call per photo
    - Returns: clockwise rotation (0/90/180/270°), flip flag, scene description
    - Corrects gross errors from prints inserted sideways or upside-down in album sleeves
    - Scene description is passed to the OpenAI glare step
    - Applied before glare removal so the glare API receives a semantically upright image
    │
    ▼
[5. Glare Removal]  ← PRIORITY 1, per extracted photo
    DEFAULT PATH (OpenAI):
    - Send the oriented photo + scene description to OpenAI gpt-image-1.5 images.edit
    - Model performs semantic, diffusion-based glare removal
    - Returns at most 1536×1024; resolution reduction is accepted for quality gain
    FALLBACK PATH (OpenCV, used when OPENAI_API_KEY is absent or --no-openai-glare):
    - detect_glare() builds a mask of specular highlights
    - remove_glare_single() inpaints masked regions using surrounding context
    │
    ▼
[6. Per-Photo Geometry Correction]  ← PRIORITY 3
    - Dewarp (barrel distortion): DISABLED by default — false positives on content-rich
      photos; iPhone corrects lens distortion in-camera before writing HEIC
    - Small-angle rotation (Hough-line): DISABLED — fires on image content rather than
      the photo frame, producing incorrect corrections; to be replaced with border detection
    │
    ▼
[7. Color Restoration]  ← PRIORITY 4
    - Auto white balance (gray-world method)
    - Deyellowing (adaptive LAB b* shift)
    - Fade restoration (CLAHE on L channel + saturation boost)
    - Sharpening (unsharp mask)
    │
    ▼
[8. Output]
    - Encode to JPEG (92% quality default), PNG, or TIFF
    - Save individual photos with naming convention
    - Save debug visualizations if --debug flag
    - Print summary (photos found, processing time per step)
```

## Image Loading Strategy

HEIC and DNG require special handling. This is critical — get loading right first.

```python
import numpy as np
from pathlib import Path
from PIL import Image

def load_image(path: str) -> np.ndarray:
    """Load image from any supported format. Returns float32 RGB array [0, 1]."""
    ext = Path(path).suffix.lower()
    
    if ext in ('.heic', '.heif'):
        # Use pillow-heif to register HEIF opener with Pillow
        from pillow_heif import register_heif_opener
        register_heif_opener()
        img = Image.open(path)
        arr = np.array(img).astype(np.float32) / 255.0
        
    elif ext in ('.dng', '.cr2', '.nef', '.arw'):
        # Use rawpy for RAW files
        import rawpy
        with rawpy.imread(path) as raw:
            # Demosaic with camera white balance, sRGB output, 16-bit
            rgb = raw.postprocess(
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=16,
                no_auto_bright=False,
            )
        arr = rgb.astype(np.float32) / 65535.0  # 16-bit to [0, 1]
        
    elif ext in ('.jpg', '.jpeg', '.png', '.tiff', '.tif'):
        img = Image.open(path)
        arr = np.array(img).astype(np.float32) / 255.0
        
    else:
        raise ValueError(f"Unsupported format: {ext}")
    
    # Ensure RGB (not RGBA, not grayscale)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    
    return arr  # float32 RGB [0, 1], shape (H, W, 3)
```

**DNG processing notes:**
- DNG files from iPhone 17 Pro (ProRAW) are 48 MP — these are large (8064×6048 or similar).
- `rawpy.postprocess()` handles demosaicing, white balance, and gamma correction.
- Use `use_camera_wb=True` to apply the white balance the phone calculated at capture time.
- The 48MP DNG gives us 2× the resolution of the 24MP HEIC — useful for quality comparison but slower to process. Default to HEIC for iteration speed, DNG for final quality validation.

**HEIC processing notes:**
- `pillow-heif` must be installed and `register_heif_opener()` must be called before `Image.open()`.
- HEIC from iPhone includes EXIF orientation — must apply before processing.
- 24MP is plenty of resolution for the pipeline. Use HEIC files for day-to-day iteration.

## Coding Conventions

- **Type hints** on all functions, enforced by mypy in strict mode.
- **Docstrings** on all public functions (Google style).
- **No print statements** — use Python `logging` module. DEBUG for intermediates, INFO for progress, WARNING for quality concerns.
- **NumPy arrays** for all image data internally. Use `np.ndarray` type hints with shape comments.
- **RGB color order** for internal processing. Convert to BGR only when calling OpenCV functions that require it, and convert back immediately.
- **Float32 [0, 1] range** for processing. Convert from/to uint8 at I/O boundaries.
- **Immutable pipeline** — each step receives input and returns output. No in-place mutation.
- **Config over magic numbers** — all thresholds and parameters in a dataclass config.

## Configuration Pattern

```python
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    """All tunable parameters in one place."""
    # Preprocessing
    max_working_resolution: int = 4000  # px, longest edge

    # Page detection (GrabCut)
    page_detect_min_area_ratio: float = 0.3
    page_detect_grabcut_iterations: int = 5
    page_detect_grabcut_max_dimension: int = 800

    # Glare detection (OpenCV fallback path only)
    glare_intensity_threshold: float = 0.85
    glare_saturation_threshold: float = 0.15
    glare_min_area: int = 100
    glare_inpaint_radius: int = 5
    glare_feather_radius: int = 5
    glare_type: str = "auto"  # "auto", "sleeve" (flat plastic), or "print" (curved glossy)

    # Photo detection
    photo_detect_method: str = "contour"  # "contour", "yolo", or "claude"
    photo_detect_min_area_ratio: float = 0.02  # 2% of page area
    photo_detect_max_count: int = 8

    # Geometry
    keystone_max_angle: float = 40.0
    rotation_auto_correct_max: float = 15.0  # used by correct_rotation(); currently a no-op
    dewarp_detection_threshold: float = 0.02
    use_dewarp: bool = False  # Disabled: false positives on content-rich photos; iPhone corrects in-camera

    # Color
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple = (8, 8)
    sharpen_radius: float = 1.5
    sharpen_amount: float = 0.5
    saturation_boost: float = 0.15

    # Output
    output_format: str = "jpeg"
    jpeg_quality: int = 92

    # AI orientation correction (Step 4.5) — Claude Haiku, one call per photo
    use_ai_orientation: bool = True
    ai_orientation_model: str = "claude-haiku-4-5-20251001"
    ai_orientation_min_confidence: str = "medium"  # ignore "low" confidence results

    # OpenAI glare removal (default) — gpt-image-1.5, one call per photo
    use_openai_glare_removal: bool = True
    openai_model: str = "gpt-image-1.5"
    openai_glare_quality: str = "high"
    openai_glare_input_fidelity: str = "high"
    forced_scene_description: Optional[str] = None  # override Claude's description

    # Legacy AI flags (not used in active pipeline)
    use_ai_quality_check: bool = False
    use_ai_fallback_detection: bool = False
```

## secrets.json

```json
{
  "ANTHROPIC_API_KEY": "sk-ant-...",
  "OPENAI_API_KEY": "sk-..."
}
```

Loaded by `src/utils/secrets.py` → `load_secrets()`. Falls back to environment variables `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` if the file is absent. File is gitignored.

## Key Technical Decisions

### Glare Removal: OpenAI by Default

OpenAI `gpt-image-1.5` (images.edit endpoint) is the default glare removal path. It handles both glare types well because it performs semantic, diffusion-based inpainting with scene understanding. The prompt includes a one-sentence description of the scene (generated by the Claude orientation step) so the model knows what to restore.

Two glare patterns still exist in the test images and must be understood when evaluating output quality:

1. **Plastic sleeve glare** (three_pics, two_pics images): Broad, flat patches that shift position when the viewing angle changes. Affects large areas of the photo.

2. **Glossy print glare** (cave, harbor, skydiving images): Contoured highlights that follow the curvature of the print surface. More complex shape, tied to the physical surface.

The OpenCV inpainting fallback (`remover_single.py`) is retained for when the OpenAI key is absent or `--no-openai-glare` is passed. It uses the glare detector mask. Quality is significantly worse than the OpenAI path on both glare types.

### AI Orientation Correction (Step 4.5)

Prints in album sleeves are frequently inserted sideways or upside-down. The AI orientation step corrects these gross errors (multiples of 90°) before glare removal. This is essential because:
- `_pick_api_size()` in `remover_openai.py` selects portrait or landscape output size based on pixel dimensions — a sideways photo would request the wrong size
- OpenAI inpaints better with a semantically upright image

One Claude Haiku call per photo returns `rotation_degrees` (0/90/180/270, clockwise to apply), `flip_horizontal` (rare), and `scene_description`. Combining orientation and description in one call saves latency.

The prompt uses explicit spatial descriptions of each rotation value ("rotate 90° clockwise: the left edge becomes the new top") to help the model reason about direction rather than guessing.

### Hough-Line Rotation Detection: Disabled

`_detect_small_rotation()` in `src/geometry/rotation.py` uses Hough line transform to find dominant lines and computes their median angle as "rotation to correct." This fires on **image content** (boat rigging, car bodies, rock edges, road lines) rather than the photo frame, producing false corrections on already-correct images.

The function returns `0.0` unconditionally. The `correct_rotation()` function remains in place as the intended home for a future border-based implementation. The right approach is to detect the white border of the physical print and use its angle — not content lines.

### Dewarp: Disabled

`correct_warp()` in `src/geometry/dewarp.py` uses Hough lines to detect curvature and `cv2.undistort()` to correct it. Two problems:
- iPhone corrects barrel/pincushion distortion in-camera before writing HEIC, so there is no lens distortion to correct in the pipeline
- The Hough detector finds curved content edges (rock walls, curved roads) and fires false positives
- The float32→uint8→float32 round-trip and bilinear interpolation introduce subtle color shifts

`use_dewarp: bool = False` in `PipelineConfig`. The code is kept for potential future use with explicit distortion calibration.

### DNG vs HEIC Workflow

- **Use HEIC files for fast iteration.** They're 24MP, load quickly, and represent what most users will actually upload.
- **Use DNG files for quality validation.** 48MP with 12/14-bit depth gives us maximum data. If the pipeline produces great results on DNG, it will work even better on the compressed HEIC.
- **The pipeline should handle both transparently.** The loader auto-detects format and normalizes to float32 RGB.
- **For processing speed:** At 24MP, each pipeline step should complete in 1–3 seconds on M4. At 48MP, allow 3–10 seconds.

### M4-Specific Notes

- The M4 chip has excellent single-thread performance and a fast unified memory subsystem. NumPy/OpenCV operations will be efficient.
- 24 GB unified RAM is plenty — even a 48MP float32 RGB image is only ~550 MB. Multiple copies during pipeline processing will fit comfortably.
- For parallel batch processing, use Python's `concurrent.futures.ProcessPoolExecutor` with 4–6 workers (M4 has 10 CPU cores).
- OpenCV's `cv2.setNumThreads()` can be tuned — default should work fine on M4.

## Debug Output

When `--debug` flag is used, each pipeline step saves its intermediate result to `debug/`:

```
debug/
├── 01_loaded.jpg                         # After format load + EXIF orientation
├── 02_page_detected.jpg                  # Page boundary overlay
├── 03_page_warped.jpg                    # After perspective correction (if boundary found)
├── 04_photo_boundaries.jpg               # Detected photo bounding boxes overlay
├── 05_photo_01_raw.jpg                   # Extracted photo 1 (before orientation)
├── 05_photo_02_raw.jpg                   # Extracted photo 2 (before orientation)
├── 05b_photo_01_oriented.jpg             # Photo 1 after AI orientation correction
├── 05b_photo_02_oriented.jpg             # Photo 2 after AI orientation correction
├── 07_photo_01_deglared.jpg              # Photo 1 after glare removal
├── 07_photo_02_deglared.jpg              # Photo 2 after glare removal
│   (06_* glare mask/overlay only appear on OpenCV fallback path)
├── 10_photo_01_geometry_final.jpg        # After geometry corrections (if any applied)
├── 11_photo_01_wb.jpg                    # After white balance
├── 12_photo_01_deyellow.jpg             # After deyellowing
├── 13_photo_01_restored.jpg             # After fade restoration
└── 14_photo_01_enhanced.jpg             # After sharpening (final output)
```

## Testing Approach

Tests use images from `test-images/`. Prefer HEIC files for test speed.

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_loader.py -v
pytest tests/test_glare.py -v
pytest tests/test_photo_detection.py -v
```

### Integration Tests with Debug Visualizations

Integration tests process real images and generate debug visualizations in `debug/`:

```bash
# Run all tests including integration
pytest tests/ -v -s

# Process a real image with debug output to inspect intermediate steps
python -m src.cli process test-images/IMG_three_pics_normal.HEIC --output ./output/ --debug

# View debug output
ls -lh debug/
open debug/05b_photo_01_oriented.jpg   # Check orientation correction
open debug/07_photo_01_deglared.jpg    # Check glare removal
```

**Debug visualizations are the primary tool for algorithm development.** After any pipeline change, run on real HEIC test images with `--debug` and inspect the numbered sequence.

## Important Notes

- **Test on real images constantly.** The test-images/ folder has your actual photos — use them for every iteration.
- **HEIC first, DNG second.** Iterate on HEIC for speed, validate on DNG for quality.
- **Two glare types matter.** Don't assume one approach works for both sleeve glare and print glare.
- **The album page shots (three_pics, two_pics) are the critical benchmark.** If glare removal + photo splitting works well on these, the product is viable.
- **Debug output is mandatory.** Always implement debug visualization for new pipeline steps.
- **Fail gracefully.** If a step can't do its job, pass through input unchanged. Log a warning.
- **Keep steps independent.** Each pipeline step should be runnable and testable in isolation.
