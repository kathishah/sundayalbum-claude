# Fix: Rotation Issues — Hough False Corrections + AI Direction Confusion

**Date:** 2026-02-19
**Branch:** `claude/fix-rotation-issue-GLroe`
**Commit:** `fix: disable Hough rotation, improve AI orientation prompt, update CLAUDE.md`

---

## Background

After implementing OpenAI glare removal and AI orientation correction (2026-02-18 and earlier
today), integration testing revealed two distinct rotation bugs. Both caused incorrect output
photos — one adding a false tilt to an already-correct image, the other rotating a photo 180°
in the wrong direction.

---

## Problem 1 — Hough-Line Rotation Fires on Image Content

### Symptom

Running the pipeline on `IMG_harbor_normal.HEIC` (a correctly-oriented single print of a
harbor scene) produced a geometry correction that tilted the image. Debug logs showed:

```
src.geometry.rotation - DEBUG - Detected rotation: -2.05°
```

Running on `IMG_two_pics_vertical_horizontal_normal.HEIC`, Photo 2 (a car photo):

```
src.geometry.rotation - DEBUG - Detected rotation: -5.81°
```

Both images were already correctly oriented. The corrections were false positives.

### Root Cause

`_detect_small_rotation()` in `src/geometry/rotation.py` uses the Hough line transform
(`cv2.HoughLines`) to find dominant lines in the image, then computes their median angle as
the "rotation to correct."

The problem: **Hough finds content edges, not the photo frame.**

- The harbor scene has masts, rigging, and water horizon lines — these dominate Hough and
  produce a -2.05° reading that reflects the composition, not any geometric error.
- The car photo has angled body panels — these produce -5.81°.

The function was structurally incapable of distinguishing between "the photo frame is tilted"
and "the subject of the photo contains diagonal lines." There is no way to fix this without
changing the detection approach entirely.

### Fix

`_detect_small_rotation()` now returns `0.0` unconditionally. A prominent comment explains:

```python
# NOTE: Disabled — returns 0.0 unconditionally. The Hough transform fires on image
# *content* (car bodies, boat rigging, rock edges) rather than the photo frame,
# producing false corrections that tilt already-correct images.
# Gross orientation errors (90°/180°/270°) are handled by the AI orientation step
# (Step 4.5). Fine-drift correction should use the photo's white border or detected
# corner quadrilateral, not content lines.
```

The disabled code remains in the file for reference. `correct_rotation()` remains in place
as the intended home for a future border-based implementation — detecting the white margin of
the physical print and using its angle, not the image content.

**File modified:** `src/geometry/rotation.py`

---

## Problem 2 — AI Orientation Prompt Returns Wrong Direction

### Symptom

`IMG_two_pics_vertical_horizontal_normal.HEIC` Photo 2 (a landscape car photo, inserted into
the album sleeve sideways with the correct top on the right) was being corrected with 270°
instead of 90°. A 270° clockwise rotation = 90° counter-clockwise, which points the car in
the opposite direction from correct.

This matched the previous journal entry's problem table entry for Photo 2, where the expected
correction was documented as 90°. Claude Haiku consistently returned 270° in testing.

### Root Cause

The original prompt told Claude:

> *"Determine the rotation needed to make it correctly oriented ... rotation_degrees is the
> clockwise rotation to apply. When in doubt, use 0."*

This is ambiguous. The model must distinguish between "the top of the scene is on the left"
(needs 90° CW) and "the top of the scene is on the right" (needs 270° CW). Without explicit
spatial anchors, Claude Haiku was confusing the two cases — correctly identifying that the
image needed a 90° rotation, but picking the wrong direction.

The original prompt also lacked any step-by-step reasoning structure. The model was expected
to jump directly to a rotation number without guidance on how to derive it.

### Fix

The prompt was rewritten with two major improvements:

**1. Two-step reasoning guide:** The model is now asked to first identify what is wrong
(upside-down? rotated 90°? already correct?), then choose the rotation value. This mirrors
how a human would reason through the problem.

**2. Explicit spatial anchors for each value:**

```
• 90  — rotate 90° clockwise: the left edge becomes the new top
         (use when the correct top of the scene is currently on the LEFT)
• 270 — rotate 270° clockwise (= 90° counter-clockwise): the right edge becomes the new top
         (use when the correct top of the scene is currently on the RIGHT)
```

These anchors give the model a concrete spatial test to apply rather than guessing from the
abstract label "clockwise."

**File modified:** `src/ai/claude_vision.py`

---

## Test Results

### Unit tests

```
61 passed, 17 skipped
```

Same counts as before — zero regressions. The 17 skipped tests require `test-images/`, which
was not present when unit tests ran.

### CLI integration tests (all three key HEIC images)

| Image | Photos detected | AI rotation (per photo) | Geometry correction time |
|---|---|---|---|
| `IMG_harbor_normal.HEIC` | 1 | 0° (no change) | **0.002 s — nothing applied** |
| `IMG_two_pics_vertical_horizontal_normal.HEIC` | 2 | Photo 1: 270° · Photo 2: **90°** ✅ | **0.004 s — nothing applied** |
| `IMG_three_pics_normal.HEIC` | 3 | 180° each | **0.006 s — nothing applied** |

**Problem 1 confirmed fixed:** Harbor geometry correction time is 0.002 s (the function is
called but immediately returns). No tilt applied.

**Problem 2 confirmed fixed:** `two_pics` Photo 2 now returns 90° CW — the correct value that
rights the car — instead of 270° CW. All three photos in `three_pics` remain correctly at 180°
(page photographed upside down), unchanged from prior working behavior.

---

## Future Work — Fine-Drift Rotation

`correct_rotation()` and the disabled `_detect_small_rotation()` remain as placeholders for
a future implementation. The right approach:

1. Detect the white border of the physical print using edge detection or color-range masking.
2. Fit lines to those four border edges (not the image content).
3. Measure their angle from horizontal/vertical — that angle is the frame tilt.
4. Apply correction only if it's within ±15°, same as the current `max_angle` guard.

This would correctly handle prints that were placed at a slight angle in the scanner or camera
frame, without firing on content edges.
