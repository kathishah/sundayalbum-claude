# 2026-04-09 — Color Restore: Replace CLAHE with Adaptive Brightness Lift + Vibrance

## Problem Statement

The current `restore_fading()` in `src/color/restore.py` is causing net-negative results on
most real-world photos:

1. **Dimming** — CLAHE redistributes the luminance histogram across small local tiles (8×8 grid).
   For outdoor photos that are already well-exposed, it pulls highlights down to make room for
   shadow detail that doesn't need lifting. The result is a measurably darker image.

2. **Grain** — CLAHE applied to smooth-toned regions (blue sky, water, open walls) amplifies
   microscopic sensor noise. The tile grid finds very low natural variance in those regions and
   aggressively stretches what little fluctuation exists into visible texture grain. This makes
   digitized prints look *more* aged, not less.

Both effects are profoundly visible on outdoor photos. Indoor/low-light scenes (restaurant,
cave interior) exhibit the same grain issue.

## Goal

**Brighten the image by introducing more light and saturating colors — unless it's already
too bright.** Preserve intentionally dark scenes (cave, candlelit dinner, night shot) by
using the scene's own brightest anchor point as a reference rather than targeting a fixed
absolute brightness.

---

## Root Cause

CLAHE is the wrong algorithm for this job. It was designed for medical imaging (CT/MRI) where
the goal is to maximise local contrast for diagnostic detail. For photo restoration, the goal
is to restore *exposure* and *color vibrancy* — which are global properties, not local ones.
CLAHE does not add light; it redistributes it. On an already-well-exposed photo it is net
destructive.

---

## Implementation Plan

### Step 1 — Replace `restore_fading()` core with a three-stage pipeline

**Stage 1: White-point stretch (auto-levels)**

Compute the 99th percentile luminance of the image. Scale the entire image so that percentile
maps to a white-point target (0.96, leaving 4% headroom to avoid clipping).

```python
wp = np.percentile(luminance, 99)
if wp > 0.05:  # avoid dividing near-zero on degenerate images
    scale = 0.96 / wp
    scale = np.clip(scale, 1.0, 2.0)  # never darken; cap at 2× lift
    image = np.clip(image * scale, 0, 1)
```

Effect:
- Faded outdoor photo: bright sky already near 1.0 → scale ≈ 1.0 → barely changes
- Faded indoor/cave: brightest anchor (lamp, face, cave entrance) → lifts proportionally
- Already-bright photo: wp already near 1.0 → scale ≈ 1.0 → near-zero change

No tile grid, no grain, no histogram redistribution.

**Stage 2: Shadow lift via tone curve (protects highlights)**

After white-point stretch, apply a luminance-aware tone curve that lifts the bottom half of
the tonal range while leaving the top 30% mostly untouched. This is what lets a restaurant
dinner photo show visible faces without blowing out the candles or turning the ambiance to
noon daylight.

```
Input luminance → Output luminance mapping (approximate):
  0.0  → 0.05   (black lifted slightly — shadow crush removed)
  0.3  → 0.38   (shadows lifted ~8%)
  0.5  → 0.55   (midtones lifted ~5%)
  0.7  → 0.72   (upper-mids barely touched)
  1.0  → 1.0    (highlights untouched)
```

Implemented as a smooth power-law blend:
```python
# shadow_lift in [0.0, 0.3]; 0 = no lift, 0.15 = moderate
lift = shadow_lift * (1.0 - luminance) ** 2
output = np.clip(luminance + lift, 0, 1)
```

Shadow lift amount is computed adaptively: proportional to how dark the image mean is after
Stage 1. If mean luminance > 0.60 after stretch, shadow lift = 0 (image is already bright).

**Stage 3: Vibrance-style saturation boost**

Replace the current fixed-multiplier saturation boost with a per-pixel vibrance model. Pixels
with low existing saturation get more boost; already-vivid pixels are protected.

```python
# S is the HSV saturation channel [0, 255]
boost_per_pixel = base_boost * (1.0 - S / 255.0)
S_new = np.clip(S + boost_per_pixel * 255, 0, 255)
```

`base_boost` defaults to 0.25 (25% maximum lift on fully desaturated pixels). The effective
boost on a pixel that is already 60% saturated is only 25% × 40% = 10%. Already-vivid
colors barely change. Faded skin tones and muted backgrounds lift the most.

Apply unconditionally — the per-pixel weighting is inherently self-limiting.

### Step 2 — Retire fading detection as a gate for CLAHE

`assess_fading()` and the `is_faded` flag will remain in the module (they may be useful
diagnostics) but will no longer gate the core restoration path. The new algorithm is
self-limiting by design and correct to apply universally.

### Step 3 — Add a brightness guard

After all three stages, check if mean luminance exceeds a ceiling (0.75). If yes, scale
back down proportionally. This is the explicit "unless already too bright" safeguard.

```python
mean_lum = np.mean(to_luminance(output))
if mean_lum > 0.75:
    scale_down = 0.72 / mean_lum
    output = np.clip(output * scale_down, 0, 1)
```

### Step 4 — Config knobs to add to `PipelineConfig`

```python
color_restore_wp_percentile: float = 99.0      # white-point percentile
color_restore_wp_target: float = 0.96          # target for white-point stretch
color_restore_shadow_lift_max: float = 0.15    # max shadow lift amount
color_restore_brightness_ceiling: float = 0.75 # mean lum above this → scale back
color_restore_vibrance_boost: float = 0.25     # base per-pixel saturation boost
```

### Step 5 — Preserve `restore_fading_conservative` and `restore_fading_aggressive` signatures

Both wrapper functions should delegate to the new implementation using appropriate parameter
overrides. Existing callers (pipeline, tests) don't break.

---

## Test Plan

### 0. Setup — clean output and debug directories

```bash
rm -rf debug/* output/*
```

### 1. Run all test-images in debug mode (HEIC only for speed)

Use the existing batch mode to process all HEIC files in one command:

```bash
source .venv/bin/activate

python -m src.cli process test-images/ --batch --filter "*.HEIC" --output ./output/ --debug
```

### 2. What to inspect manually

For each image, compare the debug files **before** and **after** `color_restore`:

| Debug file | What to check |
|---|---|
| `11_photo_NN_wb.jpg` | Baseline — image entering color restore |
| `13_photo_NN_restored.jpg` | After new color restore — should be brighter/more vivid |
| `14_photo_NN_enhanced.jpg` | Final output — check for grain artifacts |

Specific things to verify per image type:

| Test image | Expected result |
|---|---|
| `IMG_cave_normal.HEIC` | Cave stays dark; any lit surfaces brighter; no grain |
| `IMG_harbor_normal.HEIC` | Harbor/sunset colors more vivid; not blown out |
| `IMG_skydiving_normal.HEIC` | Sky more vivid blue; faces brighter; no over-brightening |
| `IMG_three_pics_normal.HEIC` | All 3 extracted photos individually brighter; no grain |
| `IMG_two_pics_vertical_horizontal_normal.HEIC` | Both photos brighter; portrait and landscape both good |

### 3. Regression check — original CLAHE path

Before deleting the old `restore_fading()` code, keep it as `_restore_fading_clahe_legacy()`
(private, unused) so we can run a side-by-side if needed.

### 4. Pass criteria

- No image is measurably darker after color restore than before it (compare `11_*` vs `13_*`)
- No new grain visible in smooth-tone regions (sky, walls, water)
- Cave and restaurant-type images retain their dark ambiance (not blown out to daylight)
- Outdoor images are noticeably more vivid/brighter
- All 5 HEIC test images process without errors

---

## Branch Policy

All implementation work happens exclusively on `feature/color-restore-enhancements`. No
changes to `main`. Commit and push to the remote branch after each logical step (e.g. after
Stage 1 is implemented, after Stage 2, after config knobs are added, after tests pass).

---

## Files to Modify

| File | Change |
|---|---|
| `src/color/restore.py` | Replace CLAHE core with white-point stretch + shadow lift + vibrance |
| `src/pipeline.py` | No change expected — same function signatures |
| `pyproject.toml` / `PipelineConfig` in `src/pipeline.py` | Add 5 new config knobs |

---

## What We Are NOT Changing

- `src/color/white_balance.py` — gray-world WB already fixed in 2026-02-20; leave it
- `src/color/deyellow.py` — deyellowing is independent; leave it
- `src/color/enhance.py` — sharpening step is fine; leave it
- OpenAI glare removal path — out of scope
- Photo detection, page detection — out of scope
