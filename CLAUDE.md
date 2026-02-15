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
├── .env                         # API keys (gitignored)
├── .gitignore
│
├── docs/
│   ├── PRD_Album_Digitizer.md              # Product requirements doc
│   ├── PHASED_PLAN_Claude_Code.md          # Phased plan of implementation starting with a local image processing engine
│   ├── Implementation_Album_Digitizer.md   # Implementation guide for the full project
│   ├── UI_Design_Album_Digitizer.md        # UI Design for the full project
│   ├── IMG_harbor_prores.DNG
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
│   │   ├── detector.py          # Detect glare/reflection regions
│   │   ├── remover_single.py    # Single-shot glare inpainting
│   │   ├── remover_multi.py     # Multi-shot glare compositing
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
│       └── io.py                # File I/O helpers
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

```bash
# Process a single album page (HEIC)
python -m src.cli process test-images/IMG_three_pics_normal.HEIC --output ./output/

# Process the high-res DNG version
python -m src.cli process test-images/IMG_three_pics_prores.DNG --output ./output/

# Process all test images
python -m src.cli process test-images/ --output ./output/ --batch

# Process with debug visualizations
python -m src.cli process test-images/IMG_three_pics_normal.HEIC --output ./output/ --debug

# Process only HEIC files (skip DNG for faster iteration)
python -m src.cli process test-images/ --output ./output/ --batch --filter "*.HEIC"

# Process with multi-shot glare removal
python -m src.cli process shot1.HEIC shot2.HEIC shot3.HEIC --multi-shot --output ./output/

# Run only specific pipeline steps
python -m src.cli process test-images/IMG_cave_normal.HEIC --output ./output/ --steps glare,color

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
    - Strip EXIF metadata (privacy)
    - Generate working copy (cap 4000px longest edge for pipeline, preserve original for final output)
    │
    ▼
[2. Page Detection]
    - Detect if this is an album page (multi-photo) or a single glossy print
    - For album pages: find page boundary quadrilateral, apply perspective correction
    - For single prints: detect print boundary, crop to print edges
    - If no clear boundary found, assume full image is the subject
    │
    ▼
[3. Glare Detection & Removal]  ← PRIORITY 1
    - Detect specular highlights (intensity + saturation thresholding + learned model)
    - Two glare patterns to handle:
      a. PLASTIC SLEEVE GLARE: broad, flat, from album page plastic. Affects three_pics and two_pics samples.
      b. GLOSSY PRINT GLARE: contoured, follows the curvature of the print surface. Affects cave, harbor, skydiving samples.
    - Single-shot: inpaint glare regions using surrounding context
    - Multi-shot: align images via feature matching, composite glare-free pixels
    - Generate confidence map
    │
    ▼
[4. Photo Detection & Splitting]  ← PRIORITY 2
    - For album pages (three_pics, two_pics): detect individual photo boundaries, extract as separate images
    - For single prints (cave, harbor, skydiving): skip splitting, treat entire image as one photo
    - Handle mixed orientations (two_pics has one portrait + one landscape)
    - Fallback: Claude vision API for complex layouts
    │
    ▼
[5. Per-Photo Geometry Correction]  ← PRIORITY 3
    - Fine keystone correction per photo
    - Bulge/warp detection and dewarping (especially for glossy prints that bow)
    - Rotation correction (small angle + 90° orientation)
    │
    ▼
[6. Color Restoration]  ← PRIORITY 4
    - Auto white balance (gray-world + album page reference)
    - Fade restoration (CLAHE on L channel in LAB space)
    - Yellowing removal (b* channel shift in LAB)
    - Sharpening (unsharp mask on L channel)
    │
    ▼
[7. Output]
    - Encode to requested format (JPEG default at 92% quality, PNG, TIFF)
    - Save individual photos with naming convention
    - Save debug visualizations if --debug flag
    - Print summary (photos found, confidence scores, processing time)
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
    prefer_heic_for_iteration: bool = True
    
    # Page detection
    page_detect_blur_kernel: int = 5
    page_detect_canny_low: int = 50
    page_detect_canny_high: int = 150
    page_detect_min_area_ratio: float = 0.3
    
    # Glare detection — two profiles for two glare types
    glare_intensity_threshold: float = 0.85
    glare_saturation_threshold: float = 0.15
    glare_min_area: int = 100
    glare_inpaint_radius: int = 5
    glare_type: str = "auto"  # "auto", "sleeve" (flat plastic), or "print" (curved glossy)
    
    # Photo detection
    photo_detect_method: str = "contour"  # "contour", "yolo", or "claude"
    photo_detect_min_area_ratio: float = 0.05
    photo_detect_max_count: int = 8
    
    # Geometry
    keystone_max_angle: float = 40.0
    rotation_auto_correct_max: float = 15.0
    
    # Color
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple = (8, 8)
    sharpen_radius: float = 1.5
    sharpen_amount: float = 0.5
    saturation_boost: float = 0.15
    
    # Output
    output_format: str = "jpeg"
    jpeg_quality: int = 92
    
    # AI
    use_ai_quality_check: bool = False
    use_ai_fallback_detection: bool = False
    anthropic_model: str = "claude-sonnet-4-5-20250929"
```

## Environment Variables

```bash
# .env file (gitignored)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
SUNDAY_ALBUM_DEBUG=1
SUNDAY_ALBUM_LOG_LEVEL=DEBUG
```

## Key Technical Decisions

### Two Types of Glare

Our test images reveal two distinct glare patterns that need different handling:

1. **Plastic sleeve glare** (three_pics, two_pics images): The plastic sheet sits flat over the photos. Glare is broad, relatively uniform, and shifts position when the viewing angle changes. This is the classic "album digitizing" problem. Multi-shot compositing works extremely well here because the glare moves between shots while the photos stay put.

2. **Glossy print glare** (cave, harbor, skydiving images): The photo paper itself has a glossy finish and may be slightly bowed or curved. Glare follows the contour of the paper — it's more complex in shape and tied to the physical surface. Single-shot inpainting is more relevant here since the glare is inherent to the print surface at any angle.

The `glare_type` config parameter controls this. In `"auto"` mode, the detector should classify which type it's dealing with based on glare pattern analysis (broad/flat = sleeve, contoured/curved = print).

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
├── 01_loaded.jpg                 # Original after format conversion + EXIF fix
├── 02_page_detected.jpg          # Page boundary overlay on original
├── 03_page_warped.jpg            # After perspective correction
├── 04_glare_mask.png             # Detected glare regions (binary mask)
├── 05_glare_overlay.jpg          # Glare regions highlighted on image
├── 05_glare_type.txt             # Detected glare type: "sleeve" or "print"
├── 06_deglared.jpg               # After glare removal
├── 07_photo_boundaries.jpg       # Detected photo boundaries overlay
├── 08_photo_01_raw.jpg           # Extracted photo 1
├── 08_photo_02_raw.jpg           # Extracted photo 2
├── 09_photo_01_geometry.jpg      # After perspective + dewarp
├── 10_photo_01_color.jpg         # After color restoration
├── 11_photo_01_final.jpg         # Final output
└── pipeline_log.txt              # Detailed timing and parameter log
```

## Testing Approach

Tests use images from `test-images/`. Prefer HEIC files for test speed.

```bash
pytest tests/ -v
pytest tests/test_loader.py -v
```

## Important Notes

- **Test on real images constantly.** The test-images/ folder has your actual photos — use them for every iteration.
- **HEIC first, DNG second.** Iterate on HEIC for speed, validate on DNG for quality.
- **Two glare types matter.** Don't assume one approach works for both sleeve glare and print glare.
- **The album page shots (three_pics, two_pics) are the critical benchmark.** If glare removal + photo splitting works well on these, the product is viable.
- **Debug output is mandatory.** Always implement debug visualization for new pipeline steps.
- **Fail gracefully.** If a step can't do its job, pass through input unchanged. Log a warning.
- **Keep steps independent.** Each pipeline step should be runnable and testable in isolation.
