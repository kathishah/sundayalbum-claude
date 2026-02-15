# Sunday Album — Claude Code Phased Build Plan

This document is your step-by-step guide for building the processing engine using Claude Code CLI. Each phase is a self-contained prompt session with clear inputs, outputs, and validation criteria.

**How to use this:** Work through each phase sequentially. Copy the prompt into Claude Code, let it build, then validate the output visually against your real test images before moving on.

---

## Pre-Setup

Before starting Phase 1, do this manually:

```bash
cd sundayalbum-claude

# Download test images (required for all phases)
# This fetches 10 test images (5 HEIC + 5 DNG) from GitHub releases
bash scripts/fetch-test-images.sh

# Verify CLAUDE.md and test images are in place
ls CLAUDE.md
ls test-images/
# Should show all 10 images (5 HEIC + 5 DNG)

# Install system dependencies via Homebrew
brew install opencv libheif libraw imagemagick

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Set up environment variables
cat > .env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
SUNDAY_ALBUM_DEBUG=1
SUNDAY_ALBUM_LOG_LEVEL=DEBUG
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
.venv/
.env
test-images/
output/
debug/
models/
__pycache__/
*.pyc
.mypy_cache/
.pytest_cache/
*.egg-info/
dist/
build/
EOF
```

---

## Phase 1: Project Scaffold + Image Loading (HEIC & DNG)

**Goal:** Project structure, dependencies, CLI skeleton, and — critically — reliable loading of HEIC and DNG files from your iPhone 17 Pro.

**Prerequisites — Test Images:**
```bash
# Ensure test images are downloaded before starting this phase
bash scripts/fetch-test-images.sh
ls test-images/  # Should show 10 files (5 HEIC + 5 DNG)
```

**Claude Code prompt:**
```
Read CLAUDE.md for project context. Set up the Sunday Album project:

1. Create pyproject.toml with these dependencies:
   - opencv-python-headless, numpy, scipy, scikit-image, Pillow, pillow-heif, rawpy
   - click, python-dotenv, matplotlib
   - anthropic, openai
   - Dev: pytest, mypy, ruff
   Note: system deps (opencv, libheif, libraw) are already installed via Homebrew.

2. Create requirements.txt by pinning current versions of the above

3. Create the full directory structure from CLAUDE.md (all __init__.py files, placeholder modules)

4. Implement src/preprocessing/loader.py:
   - load_image(path) that handles HEIC, DNG, JPEG, PNG
   - For HEIC: use pillow-heif register_heif_opener() then Pillow
   - For DNG: use rawpy with use_camera_wb=True, sRGB output, 16-bit
   - All formats normalize to float32 RGB [0, 1] numpy array
   - Read and apply EXIF orientation for HEIC (iPhone stores rotated + EXIF flag)
   - Strip EXIF metadata from the working array
   - Return the array plus metadata dict (original_size, format, bit_depth)

5. Implement src/preprocessing/normalizer.py:
   - normalize(image, config) — resize to max_working_resolution (4000px longest edge)
   - Preserve aspect ratio, use INTER_AREA for downscaling
   - Generate thumbnail (400px longest edge)
   - Return normalized image + thumbnail

6. Implement src/cli.py with Click:
   - "process" command: accepts input path(s), --output dir, --debug flag, --batch flag, --steps flag, --filter glob pattern
   - "check" command: placeholder
   - "compare" command: placeholder

7. Implement src/pipeline.py:
   - PipelineConfig dataclass with all defaults from CLAUDE.md
   - Pipeline class that runs steps sequentially (for now just load + normalize)

8. Implement src/utils/debug.py — save debug images with step numbering, always as JPEG (not HEIC/DNG) for easy viewing

9. Write tests/test_loader.py:
   - Test loading IMG_cave_normal.HEIC — verify shape, dtype, value range
   - Test loading IMG_cave_prores.DNG — verify shape, dtype, value range, check it's ~48MP
   - Test EXIF orientation is applied correctly
   - Compare HEIC vs DNG of same scene — should produce visually similar results

10. Make this work end to end:
    python -m src.cli process test-images/IMG_cave_normal.HEIC --output output/ --debug
    python -m src.cli process test-images/IMG_cave_prores.DNG --output output/ --debug
    python -m src.cli process test-images/ --output output/ --batch --filter "*.HEIC" --debug
```

**Validate:**
- `python -m src.cli process test-images/IMG_cave_normal.HEIC --output output/ --debug` runs without error
- `debug/01_loaded.jpg` exists, looks correct, proper orientation
- `python -m src.cli process test-images/IMG_cave_prores.DNG --output output/ --debug` also works
- The DNG output is visibly higher resolution than the HEIC
- `python -m src.cli process test-images/ --output output/ --batch --filter "*.HEIC"` processes all 5 HEIC files
- `pytest tests/test_loader.py -v` passes
- Check that HEIC loading doesn't crash on any of the 5 HEIC files
- Check that DNG loading doesn't crash on any of the 5 DNG files

**Common issues:**
- If `pillow-heif` fails to install: try `pip install pillow-heif --no-binary :all:` or check that `brew install libheif` completed
- If `rawpy` fails: ensure `brew install libraw` is done, then `pip install rawpy`
- If EXIF orientation is wrong: the cave/harbor/skydiving images may appear rotated — this means EXIF isn't being applied

---

## Phase 2: Page Detection & Perspective Correction

**Goal:** Detect album page boundaries and correct perspective. This matters most for the album page shots (three_pics, two_pics) which were likely photographed at a slight angle.

**Prerequisites — Test Images:**
```bash
# Ensure test images are downloaded before starting this phase
bash scripts/fetch-test-images.sh
ls test-images/  # Should show 10 files (5 HEIC + 5 DNG)
```

**Claude Code prompt:**
```
Read CLAUDE.md for context. Implement page detection and perspective correction for Sunday Album.

1. Implement src/page_detection/detector.py:
   - detect_page(image, config) -> PageDetection
   - PageDetection dataclass: corners (4 points as np.ndarray), confidence (float), is_full_frame (bool)
   - Algorithm: grayscale → Gaussian blur → Canny → Hough lines → find largest quadrilateral
   - Important: our test images have two categories:
     a. Album pages (three_pics, two_pics) — the album page itself should be detected as the boundary
     b. Individual prints (cave, harbor, skydiving) — the photo print edges should be detected
   - If no clear quad found, set is_full_frame=True and return full image bounds
   - Handle: page on tablecloth/surface, slight rotation, page not perfectly centered

2. Implement src/page_detection/perspective.py:
   - correct_perspective(image, corners) -> np.ndarray
   - Homographic transform from detected corners to proper rectangle
   - Determine output dimensions from page aspect ratio
   - cv2.warpPerspective with INTER_CUBIC

3. Wire into pipeline.py as step 2

4. Debug output:
   - 02_page_detected.jpg — original with green quad overlay and red corner dots
   - 03_page_warped.jpg — the corrected fronto-parallel result

5. Tests: test on IMG_three_pics_normal.HEIC (should find album page boundary) and IMG_cave_normal.HEIC (should find print boundary or go full-frame)

Run against ALL 5 HEIC test images with --debug. Show results summary.
```

**Validate:**
- `debug/02_page_detected.jpg` for three_pics — does the green quad match the album page edges?
- `debug/02_page_detected.jpg` for cave — does it find the photo print boundary?
- `debug/03_page_warped.jpg` — is the result rectangular and undistorted?
- Does it handle the two_pics image (which has two photos in different orientations)?

---

## Phase 3: Glare Detection (Priority 1a)

**Goal:** Accurately detect glare regions. Detection only — no removal yet. This is the most important step to get right.

**Prerequisites — Test Images:**
```bash
# Ensure test images are downloaded before starting this phase
bash scripts/fetch-test-images.sh
ls test-images/  # Should show 10 files (5 HEIC + 5 DNG)
```

**Claude Code prompt:**
```
Read CLAUDE.md for context. Implement glare detection for Sunday Album. This is the #1 priority feature.

Our test images have TWO types of glare (see CLAUDE.md "Two Types of Glare"):
- SLEEVE GLARE: broad, flat patches from plastic sleeves (three_pics, two_pics images)
- PRINT GLARE: contoured, curved highlights from glossy photo paper (cave, harbor, skydiving images)

1. Implement src/glare/detector.py:
   - detect_glare(image, config) -> GlareDetection
   - GlareDetection dataclass: mask (binary ndarray), regions (list of contours), severity_map (float ndarray 0-1), total_glare_area_ratio (float), glare_type ("sleeve" | "print" | "none")
   
   - DETECTION ALGORITHM:
     a. Convert to HSV color space
     b. Glare = high V (brightness) AND low S (saturation) — specular highlights are bright and desaturated
     c. Apply adaptive thresholding (don't use a single global threshold — lighting varies across the image)
     d. Morphological close to fill small gaps within glare regions
     e. Morphological open to remove salt noise (tiny false positives)
     f. Filter by minimum region area
   
   - GLARE TYPE CLASSIFICATION:
     a. Analyze the shape and distribution of detected glare regions
     b. Sleeve glare: fewer, larger, more uniform regions, often near center of page
     c. Print glare: more irregular, follows contours, may be elongated or curved
     d. Set glare_type accordingly — this will inform the removal strategy later
   
   - LOCAL CONTRAST CHECK (reduces false positives):
     a. Genuine glare has LOW local texture (it's a uniform bright wash)
     b. Bright photo content (white shirt, sky) has HIGHER local texture
     c. Compute local standard deviation in a sliding window
     d. Suppress detections where local texture is high (it's photo content, not glare)
   
   - SEVERITY MAP:
     a. For each glare pixel, severity = how far it deviates from the expected non-glare value
     b. Estimate expected value from surrounding non-glare pixels
     c. severity = 0.0 (barely glare) to 1.0 (completely washed out, no recoverable detail)

2. Implement src/glare/confidence.py:
   - compute_glare_confidence(original, glare_mask) -> float
   - 0.0 = severe glare everywhere, 1.0 = no glare at all
   - Based on: % of image affected, average severity, whether glare covers important content

3. Debug output:
   - 04_glare_mask.png — binary mask (white = detected glare)
   - 05_glare_overlay.jpg — original with semi-transparent orange/red overlay on glare regions, intensity = severity
   - 05_glare_type.txt — text file containing the classified glare type

4. Wire into pipeline as step 3

5. Write tests/test_glare.py

Run against ALL 5 HEIC test images with --debug. For each image, print a table:
  filename | glare_detected | glare_type | area_ratio% | confidence | num_regions
```

**Validate — spend real time here, this is the foundation:**
- Open `debug/05_glare_overlay.jpg` for EACH test image and carefully compare against the original:
  - **IMG_cave_normal.HEIC**: Does it detect the contour glare on the glossy surface? Does it miss any obvious glare spots? Does it false-positive on the bright parts of the cave photo itself?
  - **IMG_harbor_normal.HEIC**: Same questions — harbor scenes often have bright water reflections that are NOT glare.
  - **IMG_skydiving_normal.HEIC**: Bright sky is NOT glare. Does the detector distinguish correctly?
  - **IMG_three_pics_normal.HEIC**: This is the key test. Does it detect the plastic sleeve glare? Is the glare_type "sleeve"?
  - **IMG_two_pics_normal.HEIC**: Same — plastic sleeve glare detection?
- Is the severity map reasonable? (More intense at glare center, fading at edges)
- Does the glare type classification make sense for each image?

**Tuning — expect 2-4 iterations here:**
```
# If too many false positives (detecting bright photo content as glare):
"The glare detector is flagging bright sky / white surfaces in the harbor and skydiving
images as glare. Increase the local texture threshold — real glare has very low local
variance while bright photo content has texture. Also try requiring that glare regions
have a saturation below 0.10 instead of 0.15."

# If missing glare on the album pages:
"The three_pics image has visible glare from the plastic sleeve but the detector is
missing it. The glare on plastic sleeves may not be as extremely bright as on glossy
prints — try lowering the intensity threshold from 0.85 to 0.75 and see if it picks
up the sleeve reflections."

# If the mask is too noisy:
"The glare mask has lots of small disconnected blobs. Increase the morphological
closing kernel size and increase the minimum region area filter."
```

---

## Phase 4: Single-Shot Glare Removal (Priority 1b)

**Goal:** Remove detected glare from a single image using inpainting.

**Prerequisites — Test Images:**
```bash
# Ensure test images are downloaded before starting this phase
bash scripts/fetch-test-images.sh
ls test-images/  # Should show 10 files (5 HEIC + 5 DNG)
```

**Claude Code prompt:**
```
Read CLAUDE.md for context. Implement single-shot glare removal for Sunday Album.

1. Implement src/glare/remover_single.py:
   - remove_glare_single(image, glare_mask, severity_map, config) -> GlareResult
   - GlareResult dataclass: image (corrected ndarray), confidence_map (per-pixel 0-1), method_used (str)
   
   - THREE APPROACHES, combined as a hybrid:
   
   a. INTENSITY CORRECTION (for mild glare, severity < 0.4):
      - The underlying image is partially visible through mild glare (just washed out)
      - Sample the non-glare region statistics (mean, std) in a local neighborhood
      - Estimate what the pixel "should" look like by matching to neighbor distribution
      - Blend: corrected = original * (1 - severity) + estimated * severity
      - This preserves the most original detail
   
   b. OPENCV INPAINTING (for moderate glare, 0.4-0.7):
      - Use cv2.inpaint() with both INPAINT_TELEA and INPAINT_NS
      - Pick whichever produces lower error at the mask boundary
      - Inpaint radius should scale with glare region size
   
   c. CONTEXTUAL FILL (for severe glare, severity > 0.7):
      - For large, completely washed-out areas, simple inpainting produces smears
      - Sample texture and color from a wider surrounding area
      - Use a multi-scale approach: inpaint at low res first, then refine at high res
      - This won't perfectly reconstruct detail but will look plausible
   
   - HYBRID PIPELINE:
      - Classify each glare pixel by severity
      - Apply the appropriate method per-pixel
      - Blend transitions between methods using the severity map as a smooth weight
   
   - POST-PROCESSING:
      - Gaussian blur the mask boundary (feathering) to avoid hard transition edges
      - Match color/brightness statistics of repaired region to surrounding area
      - Generate per-pixel confidence map

2. Debug output:
   - 06_deglared.jpg — result after glare removal
   - 06_deglared_diff.jpg — abs difference between original and deglared (shows what changed)
   - 06_confidence_map.jpg — confidence visualization (green = high, red = low)

3. Wire into pipeline, update CLI

Run against all 5 HEIC test images with --debug. For each, print:
  filename | method_breakdown(%) | avg_confidence | processing_time_ms
```

**Validate — visual inspection is everything here:**
- **IMG_cave_normal.HEIC**: Does the glossy print glare disappear? Is the underlying cave detail plausible? Are transition edges invisible?
- **IMG_harbor_normal.HEIC**: Is the harbor scene intact after glare removal? No color shifts?
- **IMG_three_pics_normal.HEIC**: Does the broad sleeve glare get removed without damaging the three photos underneath?
- **Check the diff image**: Only glare regions should have changed. If the diff shows changes in non-glare areas, something is wrong.
- **Zoom in on boundaries**: The transition between corrected and uncorrected areas should be seamless.

**Expect 2-4 tuning iterations.** The most common issues:
- Smeary/blurry patches where large glare was inpainted → increase multi-scale refinement
- Color mismatch at boundaries → improve the color matching post-process
- Overcorrection (dark spots where glare used to be) → reduce the severity estimation

---

## Phase 5: Multi-Shot Glare Compositing (Priority 1c)

**Goal:** Best-quality glare removal using multiple shots of the same page at different angles.

**Prerequisites — Test Images:**
```bash
# Ensure test images are downloaded before starting this phase
bash scripts/fetch-test-images.sh
ls test-images/  # Should show 10 files (5 HEIC + 5 DNG)
```

**Before this phase:** Take 3-4 photos of the SAME album page (e.g., the three_pics page) at slightly different tilt angles. Save them in `test-images/multi_shot/`.

**Claude Code prompt:**
```
Read CLAUDE.md for context. Implement multi-shot glare compositing.

User has taken 3-4 photos of the same album page at different angles, saved in test-images/multi_shot/.

1. Implement src/glare/remover_multi.py:
   - remove_glare_multi(images: list[np.ndarray], config) -> GlareResult
   
   - Step 1: ALIGNMENT
     a. Pick the sharpest image as reference (Laplacian variance metric)
     b. For each other image: detect ORB features, match to reference, compute homography
     c. Warp all images into the reference frame
     d. Verify alignment: compute reprojection error, discard images with poor alignment
   
   - Step 2: PER-IMAGE GLARE DETECTION
     a. Run glare detector from Phase 3 on each aligned image
     b. Key insight: glare MOVES between shots but photo content STAYS FIXED
     c. Visualize: the glare masks should show glare in different positions across shots
   
   - Step 3: PIXEL-WISE COMPOSITING
     a. For each pixel, compute weights across all images: weight = 1.0 - glare_severity
     b. Non-glare pixels get high weight, glare pixels get low/zero weight
     c. Output pixel = weighted average across all images
     d. If ALL images have glare at a pixel, fall back to single-shot inpainting on the best one
   
   - Step 4: BLENDING & CLEANUP
     a. Smooth the weight maps with small Gaussian to avoid hard transitions
     b. Final color normalization to ensure consistency

2. Debug output:
   - 06_multi_alignment.jpg — all images overlaid at 50% opacity (alignment check)
   - 06_multi_glare_maps.jpg — side-by-side glare masks per input (should show glare moving)
   - 06_multi_weights.jpg — per-image weight maps
   - 06_multi_composite.jpg — final composite

3. Update CLI: python -m src.cli process test-images/multi_shot/ --multi-shot --output output/ --debug

4. Generate a comparison: single-shot result vs multi-shot result side by side

Print: alignment_scores per image, per-image glare coverage%, composite confidence, time.
```

**Validate:**
- Does alignment work? (`debug/06_multi_alignment.jpg` — images should overlap precisely)
- Does glare move between shots? (`debug/06_multi_glare_maps.jpg` — glare in different positions)
- Is multi-shot result clearly better than single-shot? (Compare side by side)
- Any ghosting? (Misalignment causes double edges — means alignment needs work)

---

## Phase 6: Photo Detection & Splitting (Priority 2)

**Goal:** Detect individual photos on album pages and extract them.

**Prerequisites — Test Images:**
```bash
# Ensure test images are downloaded before starting this phase
bash scripts/fetch-test-images.sh
ls test-images/  # Should show 10 files (5 HEIC + 5 DNG)
```

**Claude Code prompt:**
```
Read CLAUDE.md for context. Implement photo detection and splitting.

Critical: our test images have specific layouts:
- IMG_three_pics: album page with 3 photos behind plastic sleeve
- IMG_two_pics_vertical_horizontal: album page with 2 photos (one portrait, one landscape)
- IMG_cave, IMG_harbor, IMG_skydiving: single individual prints (should detect as 1 photo each)

1. Implement src/photo_detection/detector.py:
   - detect_photos(page_image, config) -> list[PhotoDetection]
   - PhotoDetection dataclass: bbox (x1,y1,x2,y2), corners (4 points), confidence, orientation, area_ratio

   - CONTOUR-BASED DETECTION (primary):
     a. The album page background (the page material between/around photos) is usually a uniform color
     b. Convert to grayscale, adaptive threshold to separate photos from background
     c. Find contours, filter by minimum area (>5% of page) and reasonable aspect ratio
     d. Approximate to quadrilaterals
     e. Sort left-to-right, top-to-bottom
   
   - For SINGLE PRINTS (cave, harbor, skydiving):
     a. Detect the print border against the background surface
     b. Should return exactly 1 PhotoDetection
   
   - For ALBUM PAGES (three_pics → 3 detections, two_pics → 2 detections):
     a. Detect each photo slot within the album page
     b. Handle mixed orientations in two_pics (portrait + landscape)
   
   - FALLBACK: Claude vision API if contour method confidence is low
     a. Send page image to Claude, ask for photo bounding boxes as JSON
     b. Parse response, convert to PhotoDetection objects

2. Implement src/photo_detection/splitter.py:
   - split_photos(page_image, detections) -> list[np.ndarray]
   - Extract each detected photo as a separate image
   - Apply minor perspective correction if corners aren't perfectly rectangular

3. Debug output:
   - 07_photo_boundaries.jpg — page with numbered colored rectangles on each detection
   - 08_photo_01_raw.jpg, 08_photo_02_raw.jpg, etc.

4. Wire into pipeline. Output saves each extracted photo as separate file.

Run against ALL 5 HEIC test images with --debug. For each:
  filename | photos_detected | per-photo confidence | detection_method
```

**Validate:**
- **IMG_three_pics**: Does it find exactly 3 photos? Are boundaries accurate?
- **IMG_two_pics_vertical_horizontal**: Does it find exactly 2 photos? Does it handle the mixed portrait/landscape orientation?
- **IMG_cave, IMG_harbor, IMG_skydiving**: Does it find exactly 1 photo each? Is the print boundary correct?
- Are extracted photos clean (no album background leaking in, no photo content cut off)?

---

## Phase 7: Per-Photo Geometry Correction (Priority 3)

**Prerequisites — Test Images:**
```bash
# Ensure test images are downloaded before starting this phase
bash scripts/fetch-test-images.sh
ls test-images/  # Should show 10 files (5 HEIC + 5 DNG)
```

**Claude Code prompt:**
```
Read CLAUDE.md for context. Implement per-photo geometry correction.

1. Implement src/geometry/keystone.py:
   - correct_keystone(photo, corners) -> np.ndarray
   - Homography from detected corners to proper rectangle
   - Determine correct aspect ratio from corner distances

2. Implement src/geometry/rotation.py:
   - correct_rotation(photo) -> tuple[np.ndarray, float]
   - Detect small rotation via dominant line analysis (Hough transform)
   - Auto-correct up to ±15°
   - Detect 90°/180° orientation errors using simple heuristics (face detection or scene analysis)

3. Implement src/geometry/dewarp.py:
   - correct_warp(photo) -> tuple[np.ndarray, bool]
   - Detect barrel distortion / bulging from glossy prints bowing behind sleeves
   - Find lines that should be straight, measure curvature
   - Apply inverse distortion correction
   - Important for our glossy print samples (cave, harbor, skydiving) where the paper may bow

4. Wire into pipeline as step 5 (run per extracted photo)

5. Debug output per photo:
   - 09_photo_01_keystone.jpg
   - 09_photo_01_rotation.jpg (with angle noted in filename)
   - 09_photo_01_dewarp.jpg (if warp detected)

Run against all HEIC samples. Print per photo: keystone_applied, rotation_angle, warp_detected.
```

---

## Phase 8: Color Restoration (Priority 4)

**Prerequisites — Test Images:**
```bash
# Ensure test images are downloaded before starting this phase
bash scripts/fetch-test-images.sh
ls test-images/  # Should show 10 files (5 HEIC + 5 DNG)
```

**Claude Code prompt:**
```
Read CLAUDE.md for context. Implement color restoration.

Note: our test images are from existing albums, so the photos may have some age-related fading or yellowing.

1. Implement src/color/white_balance.py:
   - auto_white_balance(photo, page_border=None) -> np.ndarray
   - Gray-world assumption (adjust channel means)
   - If album page border pixels available, use as neutral reference

2. Implement src/color/deyellow.py:
   - remove_yellowing(photo) -> tuple[np.ndarray, float]
   - LAB color space, analyze and correct b* channel
   - Be conservative — don't remove intentional warm tones

3. Implement src/color/restore.py:
   - restore_fading(photo) -> np.ndarray
   - CLAHE on L channel in LAB
   - Saturation boost only if photo is actually faded (check saturation histogram first)

4. Implement src/color/enhance.py:
   - enhance(photo, config) -> np.ndarray
   - Unsharp mask sharpening on L channel only
   - Subtle contrast with sigmoid tone curve
   - Conservative — photos should look natural, never over-processed

5. Wire into pipeline as step 6, in order: white_balance → deyellow → restore → enhance

6. Debug output per photo:
   - 10_photo_01_wb.jpg, 10_photo_01_deyellow.jpg, 10_photo_01_restored.jpg, 10_photo_01_enhanced.jpg

7. Add --compare flag to CLI: generates a side-by-side (original crop left, final right)

Run against all HEIC samples. Print: yellowing_score, saturation_before/after, sharpness_before/after.
```

---

## Phase 9: End-to-End Pipeline & AI Quality Check

**Prerequisites — Test Images:**
```bash
# Ensure test images are downloaded before starting this phase
bash scripts/fetch-test-images.sh
ls test-images/  # Should show 10 files (5 HEIC + 5 DNG)
```

**Claude Code prompt:**
```
Read CLAUDE.md for context. Wire up the full end-to-end pipeline and add AI quality assessment.

1. Update src/pipeline.py:
   - Full sequential pipeline: preprocess → page_detect → glare_remove → photo_detect → geometry → color → output
   - Graceful degradation: if any step fails, log warning and pass through unchanged
   - Track per-step timing
   - Support --steps flag for partial runs

2. Implement src/ai/claude_vision.py:
   - assess_quality(original_bytes, processed_bytes) -> QualityAssessment
   - Send before/after pair to Claude claude-sonnet-4-5-20250929 vision API
   - Structured JSON response: overall_score (1-10), glare_remaining, artifacts_detected, notes
   - Requires ANTHROPIC_API_KEY in .env

3. Implement src/ai/quality_check.py:
   - Programmatic metrics: SSIM, sharpness, histogram analysis
   - Optional AI assessment via Claude
   - Combined QualityReport

4. Output file naming: SundayAlbum_Page{XX}_Photo{YY}.{ext}

5. Run the FULL pipeline on ALL 10 test images (5 HEIC + 5 DNG). Generate summary report.

Print a final table:
  input_file | format | photos_extracted | glare_confidence | processing_time | quality_score
```

**Validate:**
- Full pipeline runs without errors on all 10 images?
- Album pages (three_pics, two_pics) produce the correct number of extracted photos?
- Individual prints (cave, harbor, skydiving) produce 1 clean photo each?
- Processing time per HEIC image? (Target: < 15 seconds on M4)
- Processing time per DNG image? (Expect 2-3× slower, that's OK)
- Does the AI quality check agree with your visual assessment?

---

## Phase 10: Iteration & Edge Cases

**Prerequisites — Test Images:**
```bash
# Ensure test images are downloaded before starting this phase
bash scripts/fetch-test-images.sh
ls test-images/  # Should show 10 files (5 HEIC + 5 DNG)
```

At this point you have a working pipeline. Use your real album pages to find and fix issues.

**Suggested Claude Code prompts for common issues:**

```
# If glare detection has false positives on bright photo content:
"The glare detector is flagging the bright sky in IMG_skydiving_normal.HEIC as glare.
Look at the debug overlay. The sky has texture (clouds, gradient) while real glare is
uniform. Increase the local texture threshold in the glare detector to distinguish
bright textured content from actual glare."

# If photo splitting misses a photo:
"IMG_three_pics_normal.HEIC should produce 3 photos but the detector only finds 2.
Look at debug/07_photo_boundaries.jpg. The third photo on the [left/right/bottom]
is being missed because [describe what you see]. Try adjusting the adaptive
threshold parameters or the minimum area ratio."

# If colors look unnatural:
"The color restoration on IMG_harbor is making the water look too blue / too warm /
over-saturated. Make the color adjustments more conservative. Reduce saturation_boost
from 0.15 to 0.08 and reduce clahe_clip_limit from 2.0 to 1.5."

# If DNG and HEIC produce very different results:
"The pipeline produces noticeably different colors for IMG_cave_normal.HEIC vs
IMG_cave_prores.DNG. The DNG version looks [more blue / darker / etc]. This is
likely because rawpy's demosaicing applies different tone curves than the iPhone's
HEIC processing. Normalize the DNG processing to match HEIC output more closely."

# For overall processing speed:
"Profile the full pipeline on IMG_three_pics_normal.HEIC and show me per-step timing.
Which step is the bottleneck? Can we optimize it — maybe by running glare detection
on a downscaled image then applying the mask at full resolution?"
```

---

## Milestone Checklist

```
[ ] Phase 1:  Project scaffold, HEIC + DNG loading, CLI skeleton
[ ] Phase 2:  Page detection, perspective correction
[ ] Phase 3:  Glare detection (detection only — critical, take your time)
[ ] Phase 4:  Single-shot glare removal
[ ] Phase 5:  Multi-shot glare compositing
[ ] Phase 6:  Photo detection & splitting
[ ] Phase 7:  Geometry correction (keystone, rotation, dewarp)
[ ] Phase 8:  Color restoration (white balance, deyellow, restore, enhance)
[ ] Phase 9:  Full pipeline, AI quality check, output naming
[ ] Phase 10: Real-world iteration on your actual album

Quality gates (must pass before moving to web UI / AWS):
[ ] HEIC and DNG files load correctly with proper orientation
[ ] Glare removal works on sleeve glare (three_pics, two_pics)
[ ] Glare removal works on print glare (cave, harbor, skydiving)
[ ] IMG_three_pics correctly splits into 3 individual photos
[ ] IMG_two_pics correctly splits into 2 photos (portrait + landscape)
[ ] Color restoration looks natural on all test images
[ ] Full pipeline runs in < 15 seconds per HEIC image on M4
[ ] You're happy with the output quality on YOUR actual album photos
```

---

## Tips for Working with Claude Code

1. **Show it your images.** When a step isn't working, describe what you see in the debug output. "The glare mask covers the entire top-left photo" is more useful than "glare detection is broken."

2. **Iterate on HEIC first.** The 24MP HEIC files process faster. Once quality is good on HEIC, validate on DNG.

3. **Use --debug on every run.** The debug output is your feedback loop. If you can't see what's happening, you can't direct improvements.

4. **Test both glare types.** The cave/harbor/skydiving images test glossy print glare. The three_pics/two_pics images test plastic sleeve glare. Both must work.

5. **Pin what works.** When a step works well, `git commit`. Run the full pipeline after each phase to catch regressions.

6. **Focus on the album pages.** The three_pics and two_pics images are the critical benchmark — they exercise glare removal AND photo splitting together. If these look great, you have a product.
