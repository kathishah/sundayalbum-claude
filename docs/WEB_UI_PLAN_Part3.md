# Sunday Album Web UI — Implementation Plan (Part 3 of 3)
# Phases 4–7: macOS UI Parity, Step Detail, Re-processing, Prod Deployment

**Version:** 1.5
**Date:** March 2026
**Status:** Phase 4 up next — Phase 3 complete (dev.sundayalbum.com live)
**See also:** WEB_UI_PLAN_Part1.md (Phases 0–2: completed), WEB_UI_PLAN_Part2.md (Phase 3: ✅ complete)

---

## Phase 4: Library UI — Match macOS App

The current library page is functional but visually different from the macOS app. This phase brings full visual and interaction parity with `mac-app/SundayAlbum/Views/LibraryView.swift` and `AlbumPageCard.swift`.

**Reference files:**
- `mac-app/SundayAlbum/Views/LibraryView.swift` — grid layout, expanded overlay, drop target
- `mac-app/SundayAlbum/Views/AlbumPageCard.swift` — card layout, ThumbBox, AfterSection, PipelineProgressWheel, ExpandedAlbumCard
- `web/src/app/(app)/library/page.tsx` — current library page (replace)
- `web/src/components/library/AlbumPageCard.tsx` — current card (rewrite)
- `web/src/components/library/ProgressWheel.tsx` — current arc wheel (replace with pie segments)

---

### 4.1 Adaptive Grid Layout

**macOS reference:** `LibraryView.swift` line 9 — `[GridItem(.adaptive(minimum: 280, maximum: 400), spacing: 16)]`

**Current state:** Library page uses a fixed horizontal list, not an adaptive grid.

**What to build:**
- CSS Grid with `grid-template-columns: repeat(auto-fill, minmax(280px, 400px))`
- 16px gap between cards (matches macOS `spacing: 16`)
- 32px horizontal padding on the grid container (matches macOS `.padding(.horizontal, 32)`)
- 40px bottom padding (matches macOS `.padding(.bottom, 40)`)
- `DropZone` shown only when `jobs.length === 0` (consistent with macOS: `if appState.jobs.isEmpty`)
- When jobs exist, drop target is still active — overlay amber border on the whole page when dragging (matches macOS `.onDrop` + `isDropTargeted` border overlay)

**Header row:**
- "Library" in Fraunces 32px semibold (matches `.fraunces(32, weight: .semibold)`)
- "Add Photos" amber button top-right (matches macOS `Label("Add Photos", systemImage: "plus")`)
- 32px padding top, 24px padding bottom between header and grid

---

### 4.2 AlbumPageCard — Correct Layout

**macOS reference:** `AlbumPageCard.swift` — `VStack` with a thumbnail row + filename below.

**Card structure (top to bottom):**
```
┌─────────────────────────────────────────────────────┐
│  [ThumbBox]  →  [AfterSection]                      │  height: 112px (88px thumb + 24px pad)
│  before thumb   progress wheel OR output thumbs     │
├─────────────────────────────────────────────────────┤
│  filename.HEIC                                      │  centered, 12px DM Sans semibold
└─────────────────────────────────────────────────────┘
```

**Geometry-aware layout (matches macOS `GeometryReader`):**
- `ThumbBox`: fixed 60px wide × 88px tall
- Arrow icon: `→` in amber, 11px semibold (matches `saAmber400`)
- `AfterSection`: fills remaining width = `card_width - 24px_hpad - 60px_before - 31px_arrow_chrome`
- Use CSS flexbox with `flex: 1` on AfterSection so it grows naturally

**Card chrome:**
- Background: white (light) / `sa-stone-900` (dark)
- Border radius: 12px
- Box shadow: `0 3px 6px rgba(0,0,0,0.08)` (matches `shadow(color: saShadow, radius: 6, y: 3)`)
- No border — shadow only

**Hover × delete button:**
- Absolutely positioned top-right, 8px inset
- Shows on `mouseenter`, hides on `mouseleave` with 200ms fade + scale(0.8) transition
- Icon: `×` circle, 16px
- Color: amber/red if job is running/queued; stone-400 if complete/failed (matches macOS logic)
- Clicking calls DELETE `/jobs/{jobId}` and removes from Zustand store

---

### 4.3 ThumbBox — Before Thumbnail from debug_urls

**macOS reference:** `AlbumPageCard.swift` `loadBeforeImage()` — uses `debug/01_loaded.jpg` first, falls back to original HEIC.

**Current state:** No before thumbnail is shown on the web card.

**What to build:**
- On job creation (optimistic), use a `URL.createObjectURL()` from the uploaded `File` object — this gives an immediate preview before the job has any debug images
- Once `GET /jobs/{jobId}` returns, use `debug_urls['load']` (the `01_loaded.jpg` presigned URL) as the before thumbnail — this is the JPEG version of the loaded image, much faster to display than the raw HEIC
- `ThumbBox` is a 60×88px container:
  - If image available: `object-fit: cover`, `border-radius: 8px`
  - If loading: stone-100 background with a small spinner (matches macOS `ProgressView().controlSize(.small)`)
- The `File` object URL is stored in Zustand job state at upload time and kept until replaced by the debug URL

**API note:** `debug_urls` is a dict returned by `GET /jobs/{jobId}`. Key `'load'` maps to the presigned URL for `01_loaded.jpg`. The frontend should request this URL and display it — it's a normal JPEG, not HEIC.

---

### 4.4 PipelineProgressWheel — Pie Segments

**macOS reference:** `AlbumPageCard.swift` `PipelineProgressWheel` + `PieSegment`.

**Current state:** `ProgressWheel.tsx` uses a continuous SVG arc (single stroke-dasharray animation).

**What to replace it with:**

A pie-chart with discrete segments — one per pipeline step:

```
totalSteps = 6  (load, page_detect, photo_detect, ai_orient, glare_remove, color_restore)
gapDegrees = 3° between each segment
completedCount = number of steps where status === 'complete'

Each segment i:
  sliceDegrees = 360 / totalSteps = 60°
  startAngle = i * 60 - 90 + 1.5  (start from top, leave 3° gap)
  endAngle   = (i+1) * 60 - 90 - 1.5
  fill: saAmber500 if i < completedCount, else saStone200
```

**SVG implementation:**
- Draw 6 pie slices using SVG `<path>` with arc commands (same as macOS `PieSegment.path(in:)`)
- Each path: move to center → arc → close
- Donut hole: white `<circle>` at center, radius = 22% of total size (matches `.padding(size * 0.22)`)
- Center label: `"{completedCount}"` in bold + `"of {totalSteps}"` in smaller text below
- Text uses DM Sans: count in `size * 0.22` equivalent (bold), label in `size * 0.12` equivalent

**Size:** 88px (matches `thumbHeight` in compact card). In expanded card: 160px.

**Animation:** On each `completedCount` change, the new segment fades in (200ms, `saStandard`).

**Step mapping** (must match backend `step` names in WebSocket `step_update` messages):
```ts
const PIPELINE_STEPS = [
  'load',
  'page_detect',
  'photo_detect',
  'ai_orient',
  'glare_remove',
  'color_restore',
] as const  // 6 steps, indices 0–5
```

---

### 4.5 AfterSection — Output Thumbnails

**macOS reference:** `AlbumPageCard.swift` `AfterSection`.

**What to build:**

When `job.status === 'complete'` and `job.output_urls` is non-empty:

**Compact card (equal-slot mode):**
- Divide `sectionWidth` equally among all output photos with 4px gaps
- `slotW = (sectionWidth - 4px * (n-1)) / n`
- Each thumbnail: `object-fit: cover`, `border-radius: 6px`, fixed height 88px, width = `slotW`

**Expanded card overlay (natural aspect ratio mode):**
- Show up to 3 photos at natural aspect ratio, height 160px
- Width = `height * min(aspectRatio, 1.5)` (matches macOS `thumbWidth` calculation)
- If more than 3 photos: show `+{overflow}` badge (stone-200 bg, stone-500 text) for remaining count

When job is processing or queued: show `PipelineProgressWheel` instead (same slot, same width as the section).

---

### 4.6 Expanded Card Overlay (Single-Click)

**macOS reference:** `LibraryView.swift` lines 78–94 + `ExpandedAlbumCard.swift`.

**What to build:**

Single-click on any card → show `ExpandedAlbumCard` as a centered overlay:

- Dim backdrop (black at 50% opacity) behind the expanded card, click to dismiss
- Background grid cards: `opacity: 0.3`, `scale: 0.95`, `blur: 3px` (matches macOS)
- `ExpandedAlbumCard` max-width: 640px, centered in viewport
- Transition: scale from 0.94 + fade (matches `.scale(scale: 0.94).combined(with: .opacity)`)

**ExpandedAlbumCard layout:**
```
┌──────────────────────────────────────────────────────────┐
│  [ThumbBox 120×160]  →  [AfterSection expanded mode]     │  padding: 20px
├──────────────────────────────────────────────────────────┤
│  filename.HEIC                    [View Step Details →]  │  padding: 20px
│  Step 3 of 6: glare_remove                               │
└──────────────────────────────────────────────────────────┘
```

- ThumbBox: 120×160px (matches macOS `frame(width: 120, height: 160)`)
- Arrow icon: amber, 16px (matches `.font(.system(size: 16, weight: .semibold))`)
- Divider between thumbnail row and footer
- Footer left: filename (14px DM Sans semibold) + `JobStatusLine` below
- Footer right: "View Step Details" amber button → navigate to `/library/{jobId}`
- Border: 1px `sa-stone-200` (light) / `sa-stone-800` (dark) at 16px radius
- Shadow: `0 12px 32px rgba(0,0,0,0.22)` (matches macOS shadow)

**JobStatusLine** (shown in footer of expanded card):
- `queued`: "Queued" with clock icon, stone-400
- `running`: slim progress bar (52px wide, 3px tall, amber fill) + "Step N of 6: {stepName}" text
- `complete`: "N photos extracted · X.Xs" with green checkmark icon
- `failed`: error message with red icon

**Double-click** on a card (or clicking "View Step Details" in expanded overlay) → navigate to `/library/{jobId}` (Phase 5).

---

### 4.7 Debug Image Strip in Expanded Card

**macOS reference:** `ExpandedAlbumCard` → `StepDetailView` (full step detail is Phase 5). In Phase 4 we show a lightweight debug strip.

**What to build (Phase 4 scope — not full StepDetailView):**

In the `ExpandedAlbumCard`, below the thumbnail row (before the footer divider), show a horizontal scrollable strip of the pipeline's debug images:

```
[01_loaded] → [02_page_detected] → [03_warped] → [04_photo_boundaries] → [05b_oriented] → [07_deglared] → [14_enhanced]
```

- Each debug image: 64px tall, natural width (aspect-fit), `border-radius: 6px`
- Label below each: step name in 9px stone-400 text
- Horizontally scrollable with `overflow-x: auto`
- Images loaded from `debug_urls` dict (presigned S3 URLs from `GET /jobs/{jobId}`)
- Shown only for completed or in-progress jobs (not queued/optimistic)
- Only show debug images that exist (some steps may be missing for certain inputs)

**`debug_urls` key mapping:**
```ts
const DEBUG_STEP_LABELS: Record<string, string> = {
  load:              '1. Load',
  page_detect:       '2. Page',
  photo_detect:      '3. Photos',
  ai_orient:         '4. Orient',
  glare_remove:      '5. Glare',
  color_restore:     '6. Color',
}
```

---

### 4.8 Per-Photo Step Tree (Multiple Photos)

**macOS reference:** `StepDetailView.swift` — StepTree sidebar with per-photo breakdown.

**Phase 4 scope:** Show a simplified step tree in the expanded card when there are multiple extracted photos, so the user can see per-photo progress without navigating to the full step detail view.

**What to build:**

When `job.photos_count > 1` and job is running or complete:

```
Photo 1  ● ● ● ● ● ○   (pie-style dots, one per step)
Photo 2  ● ● ○ ○ ○ ○
Photo 3  ● ○ ○ ○ ○ ○
```

- Shown in the expanded card below the debug strip
- Each row: photo index + 6 step dots (amber filled = complete, stone = pending)
- Step dot size: 8px
- Font: 11px DM Sans

**Data source:** `GET /jobs/{jobId}` returns per-photo step status in `photos[i].steps` dict. If not available (single-photo jobs or no per-photo breakdown), this section is hidden.

---

### 4.9 Verification Checklist

- [ ] Library page shows adaptive grid (min 280px columns, fills available width)
- [ ] "Add Photos" amber button in header (top-right)
- [ ] DropZone shown only when no jobs
- [ ] Amber border overlay appears on drag-over (even when grid is populated)
- [ ] AlbumPageCard: ThumbBox shows before thumbnail (from object URL or `debug_urls['load']`)
- [ ] AlbumPageCard: Arrow icon in amber between before and after
- [ ] AlbumPageCard: PipelineProgressWheel shows during processing (pie segments, not arc)
- [ ] PipelineProgressWheel: correct segment count (6), correct filled count, "X of 6" center label
- [ ] AlbumPageCard: Output thumbnails appear equal-slot when complete
- [ ] AlbumPageCard: Filename below card, centered, truncated in middle
- [ ] AlbumPageCard: Hover shows × delete button (top-right), fades in/out
- [ ] × button: amber/red for running jobs, stone for complete
- [ ] Single-click → expanded overlay; click backdrop → dismiss
- [ ] Background cards dim + scale + blur when overlay is open
- [ ] ExpandedAlbumCard: 120×160 ThumbBox, natural-ratio output thumbs (up to 3 + overflow badge)
- [ ] ExpandedAlbumCard: JobStatusLine correct for each state (queued/running/complete/failed)
- [ ] ExpandedAlbumCard: "View Step Details" button navigates to `/library/{jobId}`
- [ ] ExpandedAlbumCard: Debug image strip scrollable, labeled, images load from presigned URLs
- [ ] Per-photo step tree shown for multi-photo jobs

---

## Phase 5: Step Detail Views

Replicate `mac-app/SundayAlbum/Views/StepDetailView.swift` and all step-specific views.

### 5.1 StepDetailLayout

3-pane layout: breadcrumb (top), StepTree (left 196px), StepCanvas (right).

- **Breadcrumb:** "Library / {filename} / {stepName}"
- **StepTree (left sidebar, 196px):** ordered list of all pipeline steps + photos as expandable nodes
  - Click a step → StepCanvas shows that step's debug image
  - Per-photo sub-items when multiple photos
  - Active item: amber left border
- **StepCanvas (right):** shows the debug image for the selected step, with step-specific controls below

### 5.2 Step-Specific Views

| View | macOS Reference | Key Interactions |
|------|----------------|------------------|
| PageDetectionView | `Views/Steps/PageDetectionStepView.swift` | SVG overlay with draggable corner handles |
| PhotoSplitView | `Views/Steps/PhotoSplitStepView.swift` | Colored region rectangles |
| OrientationView | `Views/Steps/OrientationStepView.swift` | Rotation picker (0/90/180/270) + scene desc editor |
| GlareRemovalView | `Views/Steps/GlareRemovalStepView.swift` | Before/after with `saReveal` animation |
| ColorCorrectionView | `Views/Steps/ColorCorrectionStepView.swift` | Sliders (brightness, saturation, warmth, sharpness) |
| ResultsView | `Views/Steps/ResultsStepView.swift` | Photo grid + ComparisonView + export/download |

All images loaded from S3 via presigned URLs from `GET /jobs/{jobId}` response (`debug_urls` map + `output_urls` list).

### 5.3 Verification

- Navigate to step detail for a completed job (`/library/{jobId}`)
- Click through all steps in the tree, see corresponding debug images in canvas
- Verify animations match macOS app timing (200ms standard, 350ms slide)

---

## Phase 6: Re-processing + Polish

### 6.1 Re-process from Step

- `POST /jobs/{jobId}/reprocess { from_step, photo_index, overrides }`
- Backend starts a new Step Functions execution with `start_from` parameter
- State machine uses Choice states to skip steps before `start_from`
- Overrides (e.g., adjusted corners, rotation change) saved to `debug/{stem}_overrides.json`

### 6.2 Wire Up Interactive Controls

- PageDetectionView corner drag → reprocess from perspective
- OrientationView rotation change → reprocess from ai_orient
- ColorCorrectionView slider change → reprocess from color_restore

### 6.3 Polish

- Error handling: retry buttons, error messages per step
- Mobile responsive: library + step detail adapt to smaller screens
- Loading states and skeleton placeholders for images

### 6.4 Verification

- Adjust corners in PageDetectionView, see pipeline re-run from perspective step
- Change rotation, see glare removal re-run with correct orientation

---

## Phase 7: Production Hardening

- CloudFront distribution for frontend (S3 static hosting)
- CORS configuration on API Gateway
- Per-user rate limiting (3 concurrent jobs, 50 photos/day)
- CloudWatch dashboards + alarms
- "Delete my data" endpoint
- Provisioned concurrency on critical Lambdas (optional, for cold start mitigation)

---

## Prod Deployment: app.sundayalbum.com

**Prerequisite:** Phase 4 library UI complete + Phase 3 dev verification done (both ✅).

### Prod App Runner Service

- Service name: `sundayalbum-web-prod`
- URL: `https://yjb7t3tngm.us-west-2.awsapprunner.com` (deployed, awaiting custom domain)
- Environment variables: `NEXT_PUBLIC_API_URL=https://e1f6o1ah49.execute-api.us-west-2.amazonaws.com`

### Custom Domain Setup

1. Associate `app.sundayalbum.com` on `sundayalbum-web-prod` App Runner service
2. Add ACM cert CNAME validation record to Route 53 hosted zone
3. Add Route 53 ALIAS record: `app.sundayalbum.com` → prod App Runner domain

### Branch Strategy (for prod releases)

- Feature branches off `web-ui-implementation` → PR to `web-ui-implementation` (deploys to dev, integration test)
- When dev is verified → PR `web-ui-implementation` → `main` (deploys to prod)
- Backend (CDK) deployments remain manual

### Prod CDK Stack

- Stack name: `SundayAlbumStack` (no suffix — existing prod resources)
- API Gateway: `https://e1f6o1ah49.execute-api.us-west-2.amazonaws.com`
- WebSocket: `wss://l7t59cvhyh.execute-api.us-west-2.amazonaws.com/$default`
- S3 bucket: `sundayalbum-data-680073251743-us-west-2`
