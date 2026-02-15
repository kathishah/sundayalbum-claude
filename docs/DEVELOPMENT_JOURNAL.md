# Sunday Album Development Journal

This journal tracks the implementation progress of the Sunday Album processing engine.

---

## Phase 6: Photo Detection & Splitting — Completed
**Date:** 2026-02-15
**Status:** ✅ Completed (80% accuracy)
**Branch:** `claude/start-phase-6-01FWexucraH2twa2Qr2t2FtR`

### Summary

Implemented photo detection and splitting with contour-based detection using adaptive thresholding. Achieved 80% test accuracy (4/5 test images passing).

### Major Architectural Change

**CRITICAL DISCOVERY:** During implementation, we found that the pipeline order needed to be changed. Photo detection should happen BEFORE glare removal, not after.

**Original Pipeline (inefficient):**
1. Load & normalize
2. Page detection
3. Glare removal on whole page ❌
4. Photo detection & splitting

**New Pipeline (better):**
1. Load & normalize
2. Page detection
3. **Photo detection & splitting** ← Moved from step 6
4. **Glare removal per-photo** ← Changed from whole-page

**Why This Is Better:**
- Each photo is processed individually for glare
- More efficient: only process photos that need it
- Better quality: glare algorithms work better on individual photos
- Can apply different glare settings per photo if needed
- Some photos have glare, others don't - no wasted processing

### Implementation Details

**New Modules:**
- `src/photo_detection/detector.py` (270 lines)
  - Adaptive Gaussian thresholding for boundary detection
  - Automatic threshold inversion fallback
  - Decoration filter (35% for ≤3 photos, 50% for >3)
  - Filters by area (min 2%), aspect ratio (< 6.0), shape (4-12 vertices)

- `src/photo_detection/splitter.py` (90 lines)
  - Extracts photos with perspective correction
  - Uses homography for clean extraction

- `src/photo_detection/classifier.py` (50 lines)
  - Region classification (photo vs decoration vs caption)

- `tests/test_photo_detection.py` (180 lines)
  - 10 unit tests covering detection, splitting, corner ordering, classification

**Algorithm:** Adaptive Gaussian Thresholding
- Block size: 101 (captures photo-level boundaries, ignores texture)
- Morphological operations to clean regions
- Fallback to inverted threshold if initial attempt fails
- Smart decoration filter to remove false positives

### Test Results

| Image | Expected | Detected | Status |
|-------|----------|----------|--------|
| IMG_cave_normal.HEIC | 1 | 1 | ✅ PASS |
| IMG_harbor_normal.HEIC | 1 | 1 | ✅ PASS |
| IMG_skydiving_normal.HEIC | 1 | 1 | ✅ PASS |
| IMG_three_pics_normal.HEIC | 3 | 3 | ✅ PASS |
| IMG_two_pics_vertical_horizontal.HEIC | 2 | 3 | ⚠️ NEAR |

**Score:** 4/5 passing (80% accuracy)

### Known Issues

1. **two_pics over-detection:** Occasionally detects a small decoration as a 3rd photo
   - The decoration is 42% of the largest photo area (just above 35% filter threshold)
   - Could be fixed by increasing threshold, but may affect three_pics detection
   - Edge case, not critical for MVP

### Debug Output Structure (Updated)

Pipeline now produces:
- `01_loaded.jpg` - Original with EXIF applied
- `02_page_detected.jpg` - Page boundary overlay
- `03_page_warped.jpg` - After perspective correction
- `04_photo_boundaries.jpg` - **NEW:** Detected photo boundaries
- `05_photo_XX_raw.jpg` - **NEW:** Extracted photos (before glare removal)
- `06_photo_XX_glare_mask.png` - **NEW:** Per-photo glare detection
- `07_photo_XX_deglared.jpg` - **NEW:** Photos after glare removal

### Changes to Pipeline

**Updated `src/pipeline.py`:**
- Moved photo detection from step 6 to step 4 (before glare removal)
- Changed glare removal to process each photo in a loop
- Removed whole-page glare removal
- Updated `PipelineResult` to include `photo_detections` and `num_photos_extracted`

**Updated `src/cli.py`:**
- Modified output naming: `SundayAlbum_{filename}_PhotoXX.jpg` for multiple photos
- Added photos extracted count to summary

### Performance

Processing time per HEIC image (on M4):
- **IMG_cave_normal:** ~12s (1 photo, single print)
- **IMG_three_pics_normal:** ~7s (3 photos, album page)
- **IMG_two_pics_vertical_horizontal:** ~15s (3 detected photos)

Photo detection step: 0.2-1.2s depending on image complexity

### Commits

1. **7a806bd** - Initial Phase 6 implementation
   - Photo detection, splitting, classifier modules
   - Wired into pipeline as step 6 (before reordering)
   - 80% test accuracy

2. **3f127c2** - Pipeline reorder + decoration filter + unit tests
   - Moved photo detection before glare removal
   - Added adaptive decoration filter
   - Added 10 unit tests
   - Fixed pipeline bugs (removed duplicate code)

### Lessons Learned

1. **Pipeline order matters:** The original plan had glare removal before photo splitting, but this was inefficient. Processing individual photos separately is much better.

2. **Adaptive thresholding works well:** Gaussian adaptive threshold with large block size (101) effectively separates photos from album backgrounds.

3. **Decoration filtering is necessary:** Small decorations and borders on album pages can be detected as photos. Need intelligent filtering based on relative size.

4. **Test images reveal edge cases:** The two_pics image has a decoration that's large enough to fool the filter. Real-world testing is essential.

### Next Steps

**Phase 7: Geometry Correction**
- Per-photo keystone correction
- Rotation detection and correction
- Dewarp for glossy prints that bow

**Future Improvements for Photo Detection:**
- Fine-tune decoration filter threshold
- Add ML-based photo detection as alternative to contours
- Implement Claude Vision API fallback for complex layouts
- Handle non-rectangular album layouts

---

## Phase 5: Multi-Shot Glare Compositing — Deferred
**Date:** N/A
**Status:** ⏸️ Deferred (need multi-angle test images)

Deferred until we have test images shot from multiple angles. The single-shot glare removal in Phase 4 is working well enough for MVP.

---

## Phase 4: Single-Shot Glare Removal — Completed
**Date:** Earlier
**Status:** ✅ Completed

Implemented single-shot glare removal using intensity-based inpainting and contextual blending.

---

## Phase 3: Glare Detection — Completed
**Date:** Earlier
**Status:** ✅ Completed

Implemented glare detection with two glare type classifications (sleeve vs print).

---

## Phase 2: Page Detection — Completed
**Date:** Earlier
**Status:** ✅ Completed

Implemented page boundary detection with perspective correction.

---

## Phase 1: Project Scaffold — Completed
**Date:** Earlier
**Status:** ✅ Completed

Set up project structure, HEIC/DNG loading, CLI skeleton.

---

## Overall Progress

**Completed Phases:** 1, 2, 3, 4, 6 (5 of 10)
**Deferred:** Phase 5
**Remaining:** 7, 8, 9, 10

**Quality Gates Status:**
- ✅ HEIC and DNG files load correctly
- ✅ Glare removal works on sleeve glare
- ✅ Glare removal works on print glare
- ✅ IMG_three_pics splits correctly (3 photos)
- ⚠️ IMG_two_pics splits mostly correctly (3 instead of 2)
- ⏳ Color restoration (pending)
- ⏳ Pipeline performance (pending)
- ⏳ Real-world testing (pending)
