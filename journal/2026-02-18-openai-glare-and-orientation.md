# Plan: OpenAI Glare Removal + AI Orientation Correction

**Date:** 2026-02-18
**Status:** Design — not yet implemented

---

## Background

`scripts/glare_remove.py` is a standalone proof-of-concept that calls OpenAI's `images.edit`
API with a scene description prompt and produces significantly better glare removal than the
current OpenCV inpainting approach. The goal of this work is to promote that script into a
proper pipeline step.

During design, a second problem was identified: extracted photos frequently have incorrect
gross orientation (90°, 180°) that the existing `correct_rotation()` heuristic cannot fix. This
must be solved before glare removal so that photos arrive at the OpenAI API correctly oriented.

---

## Two Problems to Solve

### Problem 1 — Glare removal quality

The current OpenCV inpainting (`src/glare/remover_single.py`) does not produce acceptable
results on either glare type:

- **Plastic sleeve glare** (`IMG_three_pics`, `IMG_two_pics`): broad, flat patches that span
  large areas; inpainting hallucinates texture.
- **Glossy print glare** (`IMG_cave`, `IMG_harbor`, `IMG_skydiving`): contoured highlights
  that follow surface curvature; surrounding-context fill leaves visible halos.

OpenAI `gpt-image-1.5` with a descriptive prompt handles both cases well because it performs
diffusion-based inpainting with semantic understanding of the scene.

### Problem 2 — Gross orientation errors in extracted photos

Observed with real test images after photo splitting:

| Image | Photo | Issue | Correction |
|---|---|---|---|
| `IMG_three_pics_normal.HEIC` | all 3 | Entire album page photographed upside down | 180° each |
| `IMG_two_pics_vertical_horizontal_normal.HEIC` | photo 01 | Portrait print inserted sideways in sleeve | 90° or 270° |
| `IMG_two_pics_vertical_horizontal_normal.HEIC` | photo 02 | Print inserted upside down (car appears inverted) | 180° |

These are not small-angle drift. They are gross orientation errors caused by (a) the album
page being photographed upside down, and (b) individual prints inserted into sleeves at
arbitrary orientations.

The existing `_detect_orientation_error()` in `src/geometry/rotation.py` explicitly returns
`0.0` with the comment: *"This would need face detection or other semantic understanding."*
The Hough line approach normalizes all angles to [-45°, 45°], making it structurally incapable
of distinguishing 0° from 180°. The sky-brightness heuristic fails for indoor photos, caves,
cars, and any shot where bottom isn't darker than top.

Orientation must be solved before glare removal because:
1. `pick_api_size()` uses pixel dimensions (H vs W) to select portrait/landscape API size — a
   sideways portrait photo would request the wrong size.
2. OpenAI inpaints better when the image is semantically upright.

---

## Plan

### Change 1 — New module: `src/glare/remover_openai.py`

Adapt `scripts/glare_remove.py` into a pipeline-compatible module. Key adaptation: the
pipeline works with numpy arrays; the OpenAI API works with file paths.

**Public interface:**

```python
def remove_glare_openai(
    image: np.ndarray,        # float32 RGB [0, 1]
    scene_desc: str,          # 1-sentence description (from Change 2 below)
    api_key: str,
    model: str = "gpt-image-1.5",
    quality: str = "high",
    input_fidelity: str = "high",
) -> np.ndarray:              # float32 RGB [0, 1]
```

**Internally:**

1. Convert numpy array → PNG → `tempfile.NamedTemporaryFile`
2. Call `pick_api_size(w, h, "orient")` to select `1024x1024`, `1536x1024`, or `1024x1536`
   based on the (now correctly oriented) pixel dimensions
3. Call `build_prompt(scene_desc)` for the edit prompt
4. Call `client.images.edit()`
5. Convert returned PIL Image → float32 numpy array
6. Clean up temp file; return result

**Resolution trade-off:** The API returns at most 1536×1024. An extracted photo at 2000×3000
comes back at lower resolution. This is accepted — glare removal quality outweighs the resolution
reduction for this use case. The geometry and color steps that follow work on whatever resolution
they receive.

**Graceful degradation:** On any API failure (network, quota, invalid key) → log a warning and
return the original `image` unchanged. Never fail the pipeline.

---

### Change 2 — AI orientation correction: new Step 4.5

New step between photo splitting (Step 4) and glare removal (Step 5).

**New function in `src/ai/claude_vision.py`:**

```python
@dataclass
class PhotoAnalysis:
    rotation_degrees: int       # 0, 90, 180, or 270 (clockwise to apply)
    flip_horizontal: bool       # True only for true lateral mirror images (rare)
    orientation_confidence: str # "low", "medium", "high"
    scene_description: str      # 1 sentence for OpenAI glare prompt

def analyze_photo_for_processing(
    image: np.ndarray,
    api_key: Optional[str] = None,
    model: str = "claude-haiku-4-5-20251001",
) -> PhotoAnalysis:
    """Single Claude Vision call returning orientation correction + scene description."""
```

Combining both into one call avoids paying double latency when OpenAI glare removal is also
enabled. Even when OpenAI glare is disabled, the orientation correction is still needed and
still uses the same single Claude call (the scene description is simply unused).

**Prompt sent to Claude Haiku:**

> *Look at this photograph. Determine the rotation needed to make it correctly oriented —
> faces right-side up, horizons horizontal, text readable, objects in their natural position.
> Also write a one-sentence description of the scene.*
>
> *Reply with JSON only:*
> `{"rotation_degrees": <0|90|180|270>, "flip_horizontal": <true|false>,`
> `"confidence": <"low"|"medium"|"high">, "scene_description": "<one sentence>"}`
>
> *rotation_degrees is the clockwise rotation to apply. flip_horizontal is only true if the
> image is a lateral mirror (very rare for physical prints). When in doubt, use 0.*

**Application logic:**

- Apply only if confidence is `"medium"` or `"high"`.
- Apply rotation via `_rotate_image()` (already handles 90° multiples with `np.rot90()`, no
  interpolation artifacts).
- Apply horizontal flip via `np.fliplr()` if `flip_horizontal` is True.
- On API failure or `"low"` confidence → pass image through unchanged, log warning.

**Note on page-level vs print-level orientation:** No separate page-level check is needed.
For `IMG_three_pics` (whole page upside down → all 3 extracted photos upside down), Claude
Vision detects 180° for each photo independently. Same correct result, no additional logic.

---

### Change 3 — Updated pipeline step order

```
[4. Photo Detection + Splitting]
    ↓
[4.5. AI Orientation Correction]  ← NEW
     For each extracted photo:
       → analyze_photo_for_processing() → PhotoAnalysis
       → Apply np.rot90() + optional np.fliplr()
       → Photo is now semantically upright
    ↓
[5. Glare Detection]
    ↓
[5b. Glare Removal]
     if use_openai_glare_removal:
       → remove_glare_openai(photo, photo_analysis.scene_description, ...)
     else:
       → remove_glare_single() [existing OpenCV fallback]
    ↓
[6. Geometry: Dewarp + small-angle rotation]
     correct_rotation() still runs here for fine drift (±15°)
     _detect_orientation_error() remains a no-op (superseded by Step 4.5)
    ↓
[7. Color Restoration]
```

The small-angle `correct_rotation()` stays in Step 6. Step 4.5 handles gross orientation
(0°/90°/180°/270°); Step 6 handles fine drift (±3° to ±15°). These are complementary.

---

### Change 4 — `PipelineConfig` additions

```python
# Orientation correction
use_ai_orientation: bool = True
ai_orientation_model: str = "claude-haiku-4-5-20251001"
ai_orientation_min_confidence: str = "medium"  # ignore "low" confidence detections

# OpenAI glare removal
use_openai_glare_removal: bool = False          # opt-in; OpenCV is still the default
openai_model: str = "gpt-image-1.5"
openai_glare_quality: str = "high"
openai_glare_input_fidelity: str = "high"
```

`use_ai_orientation` is `True` by default because orientation errors are visually catastrophic
(upside-down photos in the output). `use_openai_glare_removal` is `False` by default because
it has API cost and latency implications — it's opt-in.

---

### Change 5 — Secrets loading

`scripts/glare_remove.py` reads the OpenAI key from `secrets.json`. The pipeline needs both
the Anthropic and OpenAI keys. Add a unified `load_secrets()` in `src/utils/io.py` (or a new
`src/utils/secrets.py`) that reads both from `secrets.json` and returns them. The Anthropic
key is already loaded somewhere in the pipeline — consolidate both under one call.

---

### Change 6 — CLI flags

```bash
# Enable OpenAI glare removal
python -m src.cli process test-images/IMG_harbor_normal.HEIC --output ./output --openai-glare

# Provide explicit scene description (skips Claude description, still does orientation)
python -m src.cli process test-images/IMG_cave_normal.HEIC --output ./output \
  --openai-glare --scene-desc "A cave interior with warm amber light"

# Disable AI orientation if needed (e.g., images known to be correctly oriented)
python -m src.cli process test-images/IMG_harbor_normal.HEIC --output ./output --no-ai-orientation
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Orientation before glare | Glare removal quality and API size selection both depend on the image being upright |
| Claude Haiku for orientation | Fast, cheap; a single-sentence description + rotation label is well within Haiku's capability |
| Combined orientation + description call | One API round-trip per photo instead of two; saves ~1-2s latency |
| `use_ai_orientation: True` by default | Upside-down output photos are a showstopper; the cost (one Haiku call per photo) is acceptable |
| `use_openai_glare_removal: False` by default | API cost and 5-15s latency per photo; user must opt in explicitly |
| Accept resolution reduction from OpenAI | Max 1536×1024 return. Glare-free 1536px photo > glare-ridden 3000px photo for most uses |
| Graceful degradation at every AI step | If orientation call fails → pass through. If glare call fails → OpenCV fallback. Pipeline never hard-fails on API issues |

---

## Estimated API Calls Per Album Page

For a 3-photo album page with OpenAI glare enabled:

| Step | API | Calls | Approx latency |
|---|---|---|---|
| Orientation + description | Claude Haiku | 3 (one per photo) | ~1-2s each |
| Glare removal | OpenAI `gpt-image-1.5` | 3 (one per photo) | ~5-15s each |
| **Total** | | **6 calls** | **~20-50s** |

For orientation-only mode (no OpenAI glare):

| Step | API | Calls | Approx latency |
|---|---|---|---|
| Orientation + description | Claude Haiku | 3 | ~1-2s each |
| **Total** | | **3 calls** | **~3-6s** |

---

## Files to Create/Modify

| File | Change |
|---|---|
| `src/glare/remover_openai.py` | New — pipeline-compatible OpenAI glare removal |
| `src/ai/claude_vision.py` | Add `PhotoAnalysis` dataclass + `analyze_photo_for_processing()` |
| `src/pipeline.py` | New Step 4.5; updated glare step; new config fields |
| `src/utils/io.py` or `src/utils/secrets.py` | Unified `load_secrets()` for both API keys |
| `src/cli.py` | Add `--openai-glare`, `--scene-desc`, `--no-ai-orientation` flags |

`src/geometry/rotation.py` is not modified — `_detect_orientation_error()` remains a no-op,
now permanently superseded by the AI orientation step.
