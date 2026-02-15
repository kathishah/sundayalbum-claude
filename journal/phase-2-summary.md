# Phase 2: Page Detection & Perspective Correction — Summary

**Status:** Complete
**Commit:** 66fadc4
**Date completed:** 2026-02-15

---

## What Was Built

### 1. Page Detection (`src/page_detection/detector.py`, ~367 lines)

Multi-strategy page boundary detection that handles both album pages and individual prints.

#### Detection Strategies (tried in order)
1. **Contour** — Canny edges + dilation + largest quadrilateral contour
2. **Contour Strong** — Lower Canny thresholds + heavier dilation for subtle boundaries
3. **Hough Lines** — Hough line detection, classifies lines as horizontal/vertical, finds 4 boundary lines and computes intersections
4. **Adaptive Threshold** — Adaptive Gaussian thresholding + morphological close + contour detection

If none succeed, falls back to full-frame (entire image treated as the page).

#### Key Data Structures
- `PageDetection` dataclass: `corners` (4x2 ndarray), `confidence` (float 0-1), `is_full_frame` (bool)
- Corner ordering via sum/diff method: top-left, top-right, bottom-right, bottom-left

#### Confidence Scoring
Confidence is computed from two factors:
- **Area ratio** — detected quad area / total image area
- **Rectangularity** — quad area / minimum bounding rect area
- Formula: `confidence = min(1.0, area_ratio * rectangularity * 2.0)`

### 2. Perspective Correction (`src/page_detection/perspective.py`, ~93 lines)

- Computes output dimensions from max of opposite edge lengths (preserves aspect ratio)
- Applies `cv2.getPerspectiveTransform` + `cv2.warpPerspective` with `INTER_CUBIC` interpolation
- Uses `BORDER_REPLICATE` to avoid black edges
- Converts to uint8 for warping (better interpolation), back to float32 after

### 3. Pipeline Integration (`src/pipeline.py`)

Page detection wired in as Step 3 (after load + normalize):
- Runs `detect_page()` with configurable parameters from `PipelineConfig`
- Saves debug overlays: `02_page_detected.jpg` (green quad + red corners), `03_page_warped.jpg` (corrected result)
- Skips perspective correction if `is_full_frame=True`
- Graceful fallthrough on errors

### 4. Test Suite (`tests/test_page_detection.py`, ~289 lines, 16 tests)

**Unit tests (synthetic images):**
- `TestOrderCorners` — Corner ordering with pre-ordered and shuffled inputs
- `TestComputeOutputDimensions` — Rectangle and skewed quad dimension computation
- `TestDetectPage` — Clear boundary detection, full-frame fallback, corner count, confidence range
- `TestCorrectPerspective` — Identity transform, crop region, value range preservation
- `TestDrawPageDetection` — Overlay size and full-frame annotation

**Integration tests (real test images):**
- `test_detect_page_three_pics` — Album page with 3 photos (should find page boundary)
- `test_detect_page_cave` — Individual print (should find print boundary or go full-frame)
- `test_detect_all_heic_images` — Runs on all 5 HEIC files, prints results summary table

---

## Results Per Test Image

| Image | Detection Method | is_full_frame | area_ratio | Confidence | Notes |
|-------|-----------------|---------------|------------|------------|-------|
| `IMG_cave_normal.HEIC` | contour_strong | false | 0.989 | high | Nearly full frame — print fills the shot |
| `IMG_harbor_normal.HEIC` | contour_strong | false | 0.562 | moderate | Cropped to photo boundary within the frame |
| `IMG_skydiving_normal.HEIC` | contour_strong | false | 0.992 | high | Nearly full frame — print fills the shot |
| `IMG_three_pics_normal.HEIC` | hough | false | 0.420 | moderate | Album page boundary found via Hough lines |
| `IMG_two_pics_vertical_horizontal_normal.HEIC` | hough | false | 0.672 | moderate | Album page boundary found via Hough lines |

**Key observations:**
- Individual prints (cave, skydiving) fill most of the frame, so contour_strong finds them with high area ratios (~0.99)
- Harbor has a tighter crop with more background visible, so area_ratio is lower (0.562)
- Album pages (three_pics, two_pics) require Hough line detection — the contour methods don't find clean quadrilaterals due to the complexity of multiple photos on the page
- No images fell through to full-frame fallback — all 5 had detectable boundaries

---

## Key Design Decisions

1. **Multi-strategy cascade** — Rather than one detection algorithm, four strategies are tried in sequence. Contour-based methods work for clean boundaries; Hough lines handle noisier edges (album pages with multiple photos); adaptive threshold catches low-contrast boundaries.

2. **Full-frame as safe fallback** — If no boundary is detected, the entire image is passed through unchanged. This ensures the pipeline never crashes on unusual inputs.

3. **Separate confidence from is_full_frame** — A detection can have low confidence but still not be full-frame. This lets downstream steps decide whether to trust the detection.

4. **uint8 conversion for OpenCV operations** — Edge detection and warping work in uint8 space for better performance and interpolation quality, with conversion back to float32 for the pipeline.

---

## What This Phase Does NOT Do

- **Does not count individual photos** — That's Phase 6 (Photo Detection & Splitting). This phase only finds the outer page/print boundary.
- **Does not detect glare** — That's Phase 3.
- **Does not handle rotation correction** — That's Phase 7 (Geometry Correction).

---

## Statistics

| Metric | Value |
|--------|-------|
| New files | 3 (detector.py, perspective.py, test_page_detection.py) |
| Lines of code added | ~960 |
| Tests added | 16 |
| Total tests passing | 24 (8 from Phase 1 + 16 new) |
| Detection strategies | 4 (contour, contour_strong, hough, adaptive) |

---

## What's Next: Phase 3

Phase 3 implements **Glare Detection** — identifying specular highlights and reflections from glossy prints and plastic sleeves. This is Priority 1 and the most important feature to get right. Two glare types to handle: broad sleeve glare (three_pics, two_pics) and contoured print glare (cave, harbor, skydiving).
