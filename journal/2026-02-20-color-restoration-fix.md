# 2026-02-20 — Color Restoration: Preventing Scene Color Destruction

## Problem Statement

After Phase 8 color restoration was integrated into the pipeline, two test images came
out significantly worse than their originals:

- **`IMG_cave_normal.HEIC`**: Colors paled and washed out. The warm amber cave lighting
  was neutralized, leaving a flat, desaturated image.
- **`IMG_skydiving_normal.HEIC`**: Strong yellow/orange tint added to the whole image.
  The blue sky was shifted toward orange, making the photo look unnaturally warm.

Confirmed by running the pipeline with `--steps` excluding `white_balance` — both images
looked correct, matching what we saw in the OpenAI-deglared intermediate (`07_photo_*_deglared.jpg`).

## Root Cause Analysis

### Primary cause: Gray-world white balance on scene-dominant colors

`auto_white_balance()` with `method="gray_world"` assumes the average pixel in any photo
should be neutral gray (equal R, G, B). It computes per-channel gains to push the image mean
toward gray, then clips those gains to `[0.5, 2.0]`.

This assumption is reasonable for indoor photos with mixed content. It is catastrophically
wrong for photos dominated by one scene color:

**Cave (warm amber dominant):**
- `mean_r` high → `gain_r` < 1.0 (red suppressed)
- `mean_b` low → `gain_b` up to 2.0× (blue doubled)
- Result: the intentional amber scene lighting is interpreted as a "red color cast to fix",
  and the algorithm pushes the image toward neutral gray, washing out the warmth entirely.

**Skydiving (blue sky dominant):**
- `mean_b` high → `gain_b` < 1.0 (blue suppressed)
- `mean_r` low → `gain_r` up to 2.0× (red boosted)
- Result: the blue sky is interpreted as a "blue color cast to fix",
  reducing blue and boosting red produces a warm orange tint across the entire image.

### Secondary cause: Deyellowing threshold too high

`detect_intentional_warmth()` used `warm_pixel_ratio > 0.3` (30% of pixels must be
warm-hued) to detect intentional scene warmth. The cave photo has warm pixels in the
20-29% range — enough to be visually prominent but below the threshold. This meant the
cave could receive `strength=0.6` deyellowing even when WB had already stripped its
amber tones, further shifting the b* channel toward blue.

### Compounding effect

All four color steps run sequentially, each receiving the previous step's output:

1. WB destroys scene colors (primary damage)
2. Deyellowing shifts b* toward blue (additional damage on warm images)
3. CLAHE redistributes contrast across the already-distorted histogram
4. Sharpening locks in the artifact

A moderate WB error in step 1 becomes a severe color shift by step 4.

## Solutions Applied

### Fix 1 — Gate WB on measured color cast (`pipeline.py`)

Added a pre-flight call to `assess_white_balance_quality()` before applying WB. The
correction is now only applied if `color_cast_score > 0.08`:

```python
wb_quality = assess_white_balance_quality(restored_photo)
if wb_quality['color_cast_score'] > 0.08:
    restored_photo, wb_info = auto_white_balance(...)
else:
    logger.info(f"Photo {photo_idx}: white balance skipped (cast score=...)")
```

`color_cast_score` is `max(|dev_r|, |dev_g|, |dev_b|) / overall_mean` — the maximum
per-channel deviation from neutral, normalized. A score of 0.08 represents an 8% channel
deviation, enough to indicate a real cast from paper aging or mixed lighting, but low
enough to pass through scene-dominant colors (cave: ~0.25 from scene; genuinely
yellowed print: cast from chemical aging superimposed on balanced content).

This is the root cause fix. The others are safety nets.

### Fix 2 — Tighten WB gain clamp (`white_balance.py`)

Even when WB does fire on a borderline cast score, the previous `[0.5, 2.0]` gain range
was far too wide. A 2× channel boost can turn a moderately warm photo orange. Changed to
`[0.75, 1.33]` (approximately ±⅓ stop) across all three WB methods:

```python
# Before
gain_r = np.clip(gain_r, 0.5, 2.0)

# After
gain_r = np.clip(gain_r, 0.75, 1.33)
```

±⅓ stop is conservative and largely reversible. It corrects genuine casts without
catastrophically shifting dominant scene colors.

### Fix 3 — Lower deyellowing threshold and default strength (`deyellow.py`)

Two changes to `remove_yellowing_adaptive()`:

1. **`detect_intentional_warmth()` threshold**: `0.3 → 0.2`
   — Protects scenes where 20-29% of pixels are warm-hued (moderate cave/candlelight
   warmth) not just full-frame sunsets (>30%).

2. **Default strength for non-warm images**: `0.6 → 0.4`
   — Reduces over-correction for images that didn't cross the warmth detection
   threshold but are still subtly warm.

## Validation

- Ran full pipeline on `IMG_cave_normal.HEIC` and `IMG_skydiving_normal.HEIC` without
  the `--steps` workaround — both produced correct-looking output matching the manually
  verified deglared intermediates.
- 78/78 unit and integration tests pass, 0 regressions.
- The WB gate logs its decision at INFO level, making it easy to audit which photos
  received correction and at what cast score.

## What to Watch For

- The `0.08` cast score threshold was chosen empirically. If genuinely color-cast album
  photos (e.g. prints stored in a tinted sleeve) are not being corrected, lower it toward
  `0.05`. If scene-dominant photos are still being altered, raise it toward `0.12`.
- The `[0.75, 1.33]` clamp may be too conservative for genuine heavy casts. A faded
  print with strong yellowing might need more than ⅓ stop of blue boost. If deyellowing
  isn't sufficient to fix heavy casts, revisit both the clamp and the gate threshold.
- The deyellowing `0.2` warmth threshold should be validated on a set of genuinely
  yellowed prints to ensure it still fires when needed.
