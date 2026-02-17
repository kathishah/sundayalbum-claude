# Fix: Multi-Photo Detection for Album Pages

**Date:** 2026-02-17
**Branch:** `claude/review-grabcut-progress-uIC5c`

---

## Problem

After merging the GrabCut page detection PR, multi-photo album pages regressed in two distinct ways.

### Bug 1 — `IMG_three_pics_normal.HEIC`: Zero photos detected

```
Page detected via grabcut: area_ratio=0.618, confidence=1.000
Single print isolated by page detection (area_ratio=0.618 < 0.8), skipping photo detection
Photos extracted: 1
```

The pipeline skipped photo detection entirely, returning the whole album page as a single photo.

### Bug 2 — `IMG_two_pics_vertical_horizontal_normal.HEIC`: Wrong split (≈1.5 : 0.5)

```
Detected 2 photos using method 'contour'
Successfully extracted 2/2 photos
```

Two photos were detected but the bounding box split was severely disproportionate — one photo received ~75% of the image area, the other ~25%.

---

## Root Causes

### Bug 1: Flawed `_is_isolated_print` bypass in `pipeline.py`

The pipeline had a shortcut: if GrabCut found a subject with `area_ratio < 0.80`, photo detection was skipped and the corrected image was returned as-is (assumed to be a single print).

This fails because **both single prints and album pages can have low area_ratios** when shot with visible background. `IMG_three_pics` was photographed with the album page at 61.8% of the frame — a normal shooting angle — but the bypass treated this as a single isolated print.

`area_ratio` alone cannot distinguish "single print against background" from "album page against background".

### Bug 2: Adaptive thresholding is brightness-sensitive

The contour detector uses `THRESH_BINARY_INV` (assumes photos are darker than background). When a portrait and landscape photo on the same page have very different overall brightness levels (e.g., one dark, one bright), the threshold works well for one and poorly for the other. The resulting blobs don't correspond to actual photo boundaries, and morphological operations compound the error into a badly disproportionate split.

---

## Fixes

### Fix 1 — Remove the `_is_isolated_print` bypass (`src/pipeline.py`)

- Removed `page_detect_single_print_threshold` from `PipelineConfig`
- Removed the entire bypass block
- Photo detection now **always runs** after page detection

Safe because the photo detection algorithm already has its own `is_single_print` heuristic (1 large contour ≥30%, no medium contours). Single prints still produce 1 output photo; the bypass is simply no longer needed.

### Fix 2 — Projection-profile fallback for disproportionate splits (`src/photo_detection/detector.py`)

Added `_detect_photos_by_projection()` which is called when contour results have area imbalance > 2.5×:

1. Compute Sobel edge magnitude across the whole image
2. Sum along rows → column profile; sum along columns → row profile
3. Smooth both profiles (kernel ∝ image size)
4. Find the deepest valley in the central 20–80% of each axis
5. Choose the axis with the lower relative valley depth (cleaner physical gap)
6. Split the image at that position → two rectangular regions

**Why this works:** Inside a photo, Sobel edges are high (rich texture). At the album page divider between photos, edges are low (uniform material). The projection minimum reliably falls at the divider regardless of the brightness of the photos on either side.

**Trigger threshold:** 2.5× imbalance between largest and smallest detected area. A legitimate portrait+landscape pair might be ~1.5× different; genuinely wrong splits are 3× or worse.

---

## Commits

- **419ef44** — Fix photo detection for multi-photo album pages

---

## Testing

```bash
# Bug 1: should now detect 3 photos (was: 1 — the whole page)
python -m src.cli process test-images/IMG_three_pics_normal.HEIC --output ./output --debug

# Bug 2: should now split correctly (was: ~1.5:0.5)
python -m src.cli process test-images/IMG_two_pics_vertical_horizontal_normal.HEIC --output ./output --debug

# Regression: single prints must still return 1 photo
python -m src.cli process test-images/IMG_cave_normal.HEIC --output ./output --debug
python -m src.cli process test-images/IMG_harbor_normal.HEIC --output ./output --debug
python -m src.cli process test-images/IMG_skydiving_normal.HEIC --output ./output --debug
```

---

## Lessons Learned

1. **Area ratio cannot distinguish single prints from album pages.** Both can have low ratios when shot with background visible. Let the photo detector — which uses content-aware heuristics — make that determination.

2. **Adaptive thresholding is brightness-dependent.** `THRESH_BINARY_INV` fails when adjacent photos have very different overall brightness. Structural edge-based approaches (Sobel + projection profiles) are more robust.

3. **Bypass logic that skips core algorithms is dangerous.** The bypass was designed to avoid false splits, but also prevented correct detection. Better pattern: always run detection, validate results, apply fallbacks when results look wrong.
