# Sunday Album Web UI — Implementation Plan (Part 3 of 3)
# Phases 4–7: macOS UI Parity, Step Detail, Re-processing, Prod Deployment

**Version:** 1.6
**Date:** March 2026
**Status:** Phase 4 up next — Phase 3 complete (dev.sundayalbum.com live)
**See also:** WEB_UI_PLAN_Part1.md (Phases 0–2: completed), WEB_UI_PLAN_Part2.md (Phase 3: ✅ complete)

---

## Phase 4: Library UI — Match macOS App

The current library page is functional but visually different from the macOS app. This phase brings full visual and interaction parity with `mac-app/SundayAlbum/Views/LibraryView.swift` and `AlbumPageCard.swift`.

### Animation & Color Reference

All animations share the same easing curve — only duration differs (source: `DesignSystem.swift`):

| Token | Duration | CSS |
|-------|----------|-----|
| `saStandard` | 200ms | `cubic-bezier(0.16, 1, 0.3, 1)` |
| `saSlide` | 350ms | `cubic-bezier(0.16, 1, 0.3, 1)` |
| `saReveal` | 600ms | `cubic-bezier(0.16, 1, 0.3, 1)` |
| `saSpring` | physics | `spring(response: 0.4s, damping: 0.6)` — use Framer Motion `type: "spring"` |

Key semantic color tokens (light / dark):

| Token | Light | Dark | Used for |
|-------|-------|------|----------|
| `saBackground` | `#ffffff` | `#000000` | Page background |
| `saCard` | `rgb(240,239,236)` | `rgb(35,31,29)` | Card surfaces, expanded card, donut hole |
| `saSurface` | `rgb(245,245,244)` | `rgb(24,21,19)` | ThumbBox loading state, inset panels |
| `saBorder` | `rgb(219,217,215)` | `rgb(55,50,47)` | Expanded card border |
| `saShadow` | `rgba(28,25,23,0.45)` | `rgba(255,255,255,0.12)` | Card drop shadow |
| `saTextPrimary` | `rgb(68,64,60)` | `rgb(245,245,244)` | Filenames, counts |
| `saTextSecondary` | `rgb(120,113,108)` | `rgb(168,162,158)` | Step label text |
| `saTextTertiary` | `rgb(168,162,158)` | `rgb(120,113,108)` | "of N" label in wheel |
| `saAmber400` | `rgb(251,191,36)` | same | Compact card arrow |
| `saAmber500` | `rgb(217,119,6)` | same | CTAs, pie fill, expanded card arrow |
| `saSuccess` | `rgb(22,163,74)` | same | Complete checkmark |
| `saError` | `rgb(220,38,38)` | same | Delete button (running), failed state |

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

**DropZone (empty state):**
- Fixed height: 320px (matches macOS `.frame(height: 320)`)
- Horizontal padding: 32px (same as grid)

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
- HStack gap between ThumbBox, arrow, and AfterSection: 10px (matches macOS `spacing: 10`)
- Arrow icon: `→` in `saAmber400` (`rgb(251,191,36)`), 11px semibold — note: compact card uses `saAmber400`, expanded card uses `saAmber500`
- `AfterSection`: fills remaining width. macOS formula: `card_width - hPad*2 - beforeW - fixedChrome` where `hPad=12, beforeW=60, fixedChrome=31` (10px gap + 11px icon + 10px gap). Use CSS `flex: 1` on AfterSection.

**Card chrome:**
- Background: `saCard` — warm grey `rgb(240,239,236)` light / `rgb(35,31,29)` dark (NOT white — matches macOS `Color.saCard`)
- Page background (`saBackground`) is white/black; card surface is the warmer grey
- Border radius: 12px
- Shadow: `saShadow` with `radius: 6, y: 3` — in CSS: `0 3px 12px rgba(28,25,23,0.45)` light, `0 3px 12px rgba(255,255,255,0.12)` dark
- No border — shadow only

**Hover × delete button:**
- Absolutely positioned top-right, 8px inset
- Shows on `mouseenter`, hides on `mouseleave`
- Transition: opacity + `scale(0.8)` with `saStandard` (200ms `cubic-bezier(0.16,1,0.3,1)`)
- Icon: `×` circle, 16px
- Color: `saError` (`rgb(220,38,38)`) if job is running/queued; `saStone400` if complete/failed
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
  - If loading: `saSurface` background (`rgb(245,245,244)` light / `rgb(24,21,19)` dark) with a small spinner (matches macOS `Color.saSurface` + `ProgressView().controlSize(.small)`)
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
- Donut hole: `saCard` color `<circle>` at center, radius = 22% of total size (matches `.padding(size * 0.22)`) — must use `saCard` (warm grey), not white, so it blends with the card background
- Center label: `"{completedCount}"` in bold + `"of {totalSteps}"` in smaller text below
- Text uses DM Sans: count in `size * 0.22` equivalent (bold), label in `size * 0.12` equivalent
- Count text: `saTextPrimary`; "of N" text: `saTextTertiary`

**Size:** 88px (matches `thumbHeight` in compact card). In expanded card: 160px.

**Animation:** On each `completedCount` change, the new segment fades in (200ms, `saStandard`).

**Step mapping** — The backend has 10 Lambda steps; the UI shows 6 user-visible steps (matching the macOS `PipelineStep` enum). Map backend step names → visual step index:

```ts
// 6 visual steps (indices 0–5), shown as pie segments and in debug strip
const VISUAL_STEPS = ['load', 'page_detect', 'photo_detect', 'ai_orient', 'glare_remove', 'color_restore'] as const

// Backend Lambda step → visual step index
// Steps not listed here (normalize, perspective, photo_split, geometry) are collapsed
// into their adjacent visual step and do not advance the counter independently.
const BACKEND_TO_VISUAL: Record<string, number> = {
  load:          0,  // sa-load
  normalize:     0,  // sa-normalize → still "load" visual step
  page_detect:   1,  // sa-page-detect
  perspective:   1,  // sa-perspective → still "page_detect" visual step
  photo_detect:  2,  // sa-photo-detect
  photo_split:   2,  // sa-photo-split → still "photo_detect" visual step
  ai_orient:     3,  // sa-ai-orient
  glare_remove:  4,  // sa-glare-remove
  geometry:      4,  // sa-geometry → still "glare_remove" visual step (pass-through)
  color_restore: 5,  // sa-color-restore
}
// completedCount = BACKEND_TO_VISUAL[current_step] when step completes,
// or BACKEND_TO_VISUAL[current_step] + 1 when current_step === 'color_restore' and status === 'complete'
```

WebSocket `step_update` messages use the backend step names (`load`, `page_detect`, etc.). Use `BACKEND_TO_VISUAL` to compute `completedCount` for the pie wheel.

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
- 8px gap between thumbnails in expanded mode (matches macOS `HStack(spacing: 8)`)
- If more than 3 photos: show `+{overflow}` badge alongside them
  - Background: `saStone200`; text: `saStone500`; border-radius: 6px
  - Badge width = `thumbHeight * 0.65` (matches macOS `frame(width: thumbHeight * 0.65, height: thumbHeight)`)

When job is processing or queued: show `PipelineProgressWheel` instead (same slot, same width as the section).

---

### 4.6 Expanded Card Overlay (Single-Click)

**macOS reference:** `LibraryView.swift` lines 78–94 + `ExpandedAlbumCard.swift`.

**What to build:**

Single-click on any card → show `ExpandedAlbumCard` as a centered overlay:

- Dim backdrop (black at 50% opacity) behind the expanded card, click to dismiss
- Background grid cards: `opacity: 0.3`, `scale: 0.95` — animated with `saSpring` (spring: response 0.4s, damping 0.6)
- Blur: apply `blur(3px)` to the **scroll container** (not per-card) — animated with `saStandard` (200ms `cubic-bezier(0.16,1,0.3,1)`)
- Two animations run simultaneously on open/close: spring on card opacity/scale, standard on container blur
- `ExpandedAlbumCard` max-width: 640px, outer padding 48px from overlay edges, centered in viewport
- Transition: scale from 0.94 + fade (matches `.scale(scale: 0.94).combined(with: .opacity)`) — use `saSpring` for the card entry/exit

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
- Arrow icon: `saAmber500` (`rgb(217,119,6)`), 16px semibold — expanded card uses `saAmber500`, unlike compact card which uses `saAmber400`
- HStack gap: 16px (matches macOS `HStack(spacing: 16)`)
- Thumbnail row padding: 20px all sides (matches macOS `.padding(20)`)
- Divider between thumbnail row and footer
- Footer padding: 20px all sides (matches macOS `.padding(20)`)
- Footer left: filename (14px DM Sans semibold) + `JobStatusLine` below (5px gap)
- Footer right: "View Step Details" amber button → navigate to `/library/{jobId}`
- Background: `saCard` (warm grey, same as compact card — NOT white)
- Border: 1px `saBorder` — `rgb(219,217,215)` light / `rgb(55,50,47)` dark (matches macOS `Color.saBorder`)
- Border radius: 16px
- Shadow: `0 12px 32px rgba(28,25,23,0.22)` (matches macOS `Color.saStone900.opacity(0.22), radius: 32, y: 12`)

**JobStatusLine** (shown in footer of expanded card):
- `queued`: "Queued" with clock icon, `saStone400`
- `running`: slim progress bar (52px wide, 3px tall, capsule shape, `saAmber500` fill on `saStone200` track) + "Step N of 6: {stepName}" text (11px DM Sans, `saTextSecondary`)
  - Step number: `min(BACKEND_TO_VISUAL[current_step] + 1, 6)` using the mapping from 4.4
  - Step name: use `VISUAL_STEPS[BACKEND_TO_VISUAL[current_step]]` (human label)
- `complete`: "N photo(s) extracted · X.Xs" with `saSuccess` checkmark icon (11px DM Sans)
  - Pluralise correctly: "1 photo extracted" vs "3 photos extracted"
- `failed`: error message with `saError` exclamation icon (11px DM Sans)

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
- [ ] DropZone shown only when no jobs, fixed height 320px
- [ ] Amber border overlay appears on drag-over (even when grid is populated)
- [ ] AlbumPageCard: background is `saCard` warm grey (not white)
- [ ] AlbumPageCard: ThumbBox shows before thumbnail (from object URL or `debug_urls['load']`)
- [ ] AlbumPageCard: Arrow icon is `saAmber400` (bright yellow-gold) in compact card
- [ ] AlbumPageCard: PipelineProgressWheel shows during processing (pie segments, not arc)
- [ ] PipelineProgressWheel: correct segment count (6), correct filled count, "X of 6" center label
- [ ] PipelineProgressWheel: donut hole color matches `saCard` (warm grey, not white)
- [ ] AlbumPageCard: Output thumbnails appear equal-slot when complete
- [ ] AlbumPageCard: Filename below card, centered, truncated in middle
- [ ] AlbumPageCard: Hover shows × delete button (top-right), fades in/out
- [ ] × button: `saError` red for running/queued jobs, `saStone400` for complete/failed
- [ ] Single-click → expanded overlay with `saSpring` animation; click backdrop → dismiss
- [ ] Background cards dim (0.3 opacity) + scale (0.95) with spring; scroll container blurs (3px) with saStandard
- [ ] ExpandedAlbumCard: background is `saCard`, border is `saBorder`, radius 16px
- [ ] ExpandedAlbumCard: 120×160 ThumbBox, arrow in `saAmber500`, 16px HStack gap
- [ ] ExpandedAlbumCard: natural-ratio output thumbs (up to 3 + overflow badge width = thumbHeight * 0.65)
- [ ] ExpandedAlbumCard: JobStatusLine correct for each state (queued/running/complete/failed)
- [ ] ExpandedAlbumCard: JobStatusLine step number uses BACKEND_TO_VISUAL mapping
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

**Typography note for Phase 5:** The macOS app uses **JetBrains Mono** (monospace) for filenames and metadata inside `ComparisonView` and `ResultsStepView` (11px). This font is NOT used in `AlbumPageCard` or `ExpandedAlbumCard` (those use DM Sans). Load JetBrains Mono as a web font and apply it to filename/metadata displays in the step detail views only.

**GlareRemovalView reveal animation:** The "after" image uses `saReveal` (600ms `cubic-bezier(0.16,1,0.3,1)`). The amber glow shadow on the after image fades in with the same duration but a 400ms delay. Trigger this sequence on component mount and again when the photo index changes.

### 5.3 Verification

- Navigate to step detail for a completed job (`/library/{jobId}`)
- Click through all steps in the tree, see corresponding debug images in canvas
- Verify all animation timings: saStandard 200ms, saSlide 350ms, saReveal 600ms — all use `cubic-bezier(0.16,1,0.3,1)`
- Verify spring on card interactions (response 0.4s, damping 0.6)
- GlareRemovalView: after image fades in over 600ms; glow starts at 400ms

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
