# Public Marketing Site — Implementation Log

**Date:** 2026-04-10  
**Branch:** `feature/public-website`  
**Status:** Complete — pending DNS cutover of `www.sundayalbum.com`

---

## What Was Built

A full public marketing site served as a Next.js route group `(public)` inside the existing
web app. No new deployment or infrastructure — `www.sundayalbum.com` and `app.sundayalbum.com`
will both route to the same App Runner service; the `(public)` group intercepts the root and
the marketing paths, while all authenticated app routes remain unchanged.

### New pages

| Route | File | Purpose |
|-------|------|---------|
| `/` | `(public)/page.tsx` | Landing page — hero, how it works, demo sliders, pipeline teaser |
| `/pricing` | `(public)/pricing/page.tsx` | Free Mac app, free web tier, BYOK model |
| `/download` | `(public)/download/page.tsx` | macOS download + web app CTA |
| `/about` | `(public)/about/page.tsx` | Project motivation and contact |
| `/pipeline` | `(public)/pipeline/page.tsx` | Step-by-step pipeline walkthrough with real images |

The previous root `app/page.tsx` (which redirected to `/login`) was deleted; the new
`(public)/page.tsx` now owns the root route.

### New components

| Component | Purpose |
|-----------|---------|
| `MarketingNav` | Top nav with logo, page links, "Get started" CTA |
| `MarketingFooter` | Footer with links grouped by category |
| `BeforeAfterSlider` | Draggable reveal slider — used on the landing page demo |

### Layout

`(public)/layout.tsx` wraps all marketing pages with `MarketingNav` + `MarketingFooter` and
exports shared OpenGraph `metadata`. The app's existing `(app)/layout.tsx` is unchanged.

---

## Landing Page Design Decisions

### Hero

Two-column layout: copy on the left, animated image sequence on the right. A `HeroCycler`
component cross-fades through three stages of the cave pipeline output (raw crop →
deglared → color restored) on a 2.8-second loop using Framer Motion `AnimatePresence`.

Source images: `debug/IMG_cave_normal_{05_photo_01_raw, 07_deglared, 14_enhanced}.jpg`,
resized to 900×600 and saved to `web/public/demo/cave-stage-{0,1,2}.jpg`.

### "See the results" section

Two `BeforeAfterSlider` instances with real pipeline output:

- **Pair A** (landscape, `h-72` × full column width): a single print — drag right to reveal
  the restored image. `initialPosition={0}` so it starts showing the "before" raw extract.
- **Pair B** (portrait): an album page with three prints — drag right to reveal the three
  individually extracted and restored photos stacked vertically. Dimensions are a precise
  90° rotation of Pair A: Pair B width = Pair A height (`w-72`); Pair B height = Pair A
  rendered width (measured via `ResizeObserver`).

Features list (previously a separate card grid) is displayed as inline text columns on the
left and right of the two sliders — no separate "Features" section.

### CTA routing

All CTAs use relative paths (`/login`, `/settings`) rather than absolute
`https://app.sundayalbum.com` URLs. This means the same build works correctly on both
`dev.sundayalbum.com` (dev) and `app.sundayalbum.com` / `www.sundayalbum.com` (prod) without
any environment-specific logic.

---

## Pipeline Page

`/pipeline` shows the 8 visually distinct steps from `IMG_1268.HEIC` with real debug output
images. The page opens with a full-bleed hero of the final output. Steps alternate image-left /
image-right. The page splits into two labeled sections: "page-level steps" and "per-photo steps."

For authenticated users arriving via `?jobId=...`, each step exposes config toggles and a
"Re-run from here" button that calls `POST /jobs/{jobId}/reprocess` with `from_step` and
`config` overrides. This is linked from the main nav and from the "Under the hood" section on
the landing page.

Source images saved to `web/public/demo/pipeline/`:

| File | Source | Dimensions |
|------|--------|------------|
| `01_loaded.jpg` | `debug/IMG_1268_01_loaded.jpg` | 600×800 |
| `02_page_detected.jpg` | `debug/IMG_1268_02_page_detected.jpg` | 600×800 |
| `03_page_warped.jpg` | `debug/IMG_1268_03_page_warped.jpg` | 900×596 |
| `04_photo_bounds.jpg` | `debug/IMG_1268_04_photo_boundaries.jpg` | 900×596 |
| `05_raw.jpg` | `debug/IMG_1268_05_photo_01_raw.jpg` | 900×596 |
| `06_oriented.jpg` | `debug/IMG_1268_05b_photo_01_oriented.jpg` | 900×596 |
| `07_deglared.jpg` | `debug/IMG_1268_07_photo_01_deglared.jpg` | 900×600 |
| `08_enhanced.jpg` | `debug/IMG_1268_14_photo_01_enhanced.jpg` | 900×600 |
| `09_final.jpg` | `output/SundayAlbum_IMG_1268_Photo01.jpg` | 900×600 |

---

## Files Changed / Added

```
web/src/app/(public)/layout.tsx            — new
web/src/app/(public)/page.tsx              — new (replaces root redirect)
web/src/app/(public)/about/page.tsx        — new
web/src/app/(public)/download/page.tsx     — new
web/src/app/(public)/pricing/page.tsx      — new
web/src/app/(public)/pipeline/page.tsx     — new
web/src/app/page.tsx                       — deleted (was a redirect to /login)
web/src/components/MarketingNav.tsx        — new
web/src/components/MarketingFooter.tsx     — new
web/src/components/BeforeAfterSlider.tsx   — new
web/public/demo/cave-stage-{0,1,2}.jpg    — new (hero animation frames)
web/public/demo/pair-a-{before,after}.jpg — new (landing page slider, Pair A)
web/public/demo/pair-b-{before,after}.jpg — new (landing page slider, Pair B)
web/public/demo/pair-b-after-{1,2,3}.jpg  — new (Pair B after slot: 3 individual photos)
web/public/demo/pipeline/*.jpg             — new (9 pipeline step images)
.gitignore                                 — added mac-app build dirs + web tsbuildinfo
```

---

## Pending

- **DNS:** Point `www.sundayalbum.com` to the same App Runner service as `app.sundayalbum.com`.
  Update Route 53 in the `sundayalbum.com` zone. The Next.js app handles both hostnames without
  any config change.
- **PR to `dev`:** `feature/public-website` → `dev` for review and deploy to dev environment.
