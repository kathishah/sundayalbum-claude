# Album Digitizer — UI Design Specification

**Version:** 1.0  
**Date:** February 2026  
**Status:** Draft  
**Companion Documents:** PRD_Album_Digitizer.md, Implementation_Album_Digitizer.md

---

## 1. Design Philosophy

### 1.1 Aesthetic Direction: "Warm Archival"

The design evokes the feeling of opening a cherished family photo album — warm, nostalgic, and personal — while the interface itself is unmistakably modern, clean, and confident. The product handles something deeply sentimental (family memories), and the design should honor that emotional weight without being saccharine.

**Core tension:** Nostalgic warmth meets precision engineering. The photos are old; the tool is not.

**Memorable signature:** The before/after glare reveal moment. When processing completes, the glare melts away in an animated transition — the single "wow" moment that makes users want to share the product.

### 1.2 Design Principles

1. **Trust through transparency.** Users are handing over irreplaceable family memories. Every step shows what's happening — no black boxes. Processing stages are visible, confidence is surfaced, originals are always accessible.
2. **The photo is the hero.** The UI recedes. Backgrounds are muted, controls are peripheral, chrome is minimal. The user's photos dominate every screen.
3. **Dark during capture, light during review.** Dark mode during camera capture (reduces screen glare reflecting onto album pages — a functional choice). Light mode during review and editing (accurate color judgment requires a neutral background).
4. **Progressive disclosure.** Show the simple path first. Advanced controls (manual crop, color sliders, warp correction) are available but tucked away. A first-time user should complete the full flow without opening a single settings panel.
5. **Batch-aware, not batch-only.** Every screen works for one photo or one hundred. Progress indicators, queue management, and batch actions scale gracefully.

---

## 2. Design System

### 2.1 Color Palette

#### Primary Palette

```
--color-amber-50:    #FFF8F0     Background (light mode — warm white, not stark)
--color-amber-100:   #FEF0DC     Card backgrounds, subtle fills
--color-amber-200:   #FBD FA2    Borders, dividers
--color-amber-500:   #D97706     Primary accent (warm amber — evokes old photo tones)
--color-amber-600:   #B45309     Primary accent hover/active
--color-amber-800:   #78350F     Dark accent text
```

#### Neutral Palette

```
--color-stone-50:    #FAFAF9     Secondary background
--color-stone-100:   #F5F5F4     Input fields, inactive areas
--color-stone-200:   #E7E5E4     Borders, dividers
--color-stone-400:   #A8A29E     Placeholder text, disabled states
--color-stone-500:   #78716C     Secondary text
--color-stone-700:   #44403C     Primary text
--color-stone-900:   #1C1917     Headings, high-emphasis text
```

#### Dark Mode Palette (Capture Screens)

```
--color-dark-bg:     #0C0A09     Primary background
--color-dark-surface:#1C1917     Cards, panels
--color-dark-border: #292524     Borders
--color-dark-text:   #F5F5F4     Primary text
--color-dark-muted:  #A8A29E     Secondary text
```

#### Semantic Colors

```
--color-success:     #16A34A     Processing complete, good confidence
--color-warning:     #EA580C     Low confidence, attention needed
--color-error:       #DC2626     Processing error, critical issues
--color-info:        #2563EB     Informational states
--color-glare-overlay: rgba(234, 88, 12, 0.4)   Glare detection heat map overlay
--color-page-detect:   rgba(22, 163, 74, 0.6)    Page boundary detection overlay
--color-page-partial:  rgba(220, 38, 38, 0.6)    Partial page detection overlay
```

### 2.2 Typography

#### Font Stack

```
--font-display:  'Fraunces', serif          Headings, hero text, emotional moments
--font-body:     'DM Sans', sans-serif      Body text, UI labels, controls
--font-mono:     'JetBrains Mono', monospace File names, technical info, counters
```

**Fraunces** is a variable display serif with an "old style" warmth — soft, organic curves that feel handcrafted. It pairs beautifully with the archival theme without looking dated. Used sparingly for headings and hero moments.

**DM Sans** is geometric and clean — the modern counterpoint. Highly legible at small sizes for UI controls and body text.

#### Type Scale

```
--text-hero:     48px / 1.1  (Fraunces, weight 600)     Landing page hero
--text-h1:       32px / 1.2  (Fraunces, weight 500)     Page titles
--text-h2:       24px / 1.3  (Fraunces, weight 500)     Section headings
--text-h3:       18px / 1.4  (DM Sans, weight 600)      Subsection headings
--text-body:     16px / 1.5  (DM Sans, weight 400)      Body text
--text-body-sm:  14px / 1.5  (DM Sans, weight 400)      Secondary body, captions
--text-caption:  12px / 1.4  (DM Sans, weight 500)      Labels, metadata
--text-mono:     13px / 1.4  (JetBrains Mono, weight 400)  File names, counters
```

### 2.3 Spacing & Layout

#### Spacing Scale (8px base)

```
--space-1:   4px       Inline padding, tight gaps
--space-2:   8px       Component internal padding
--space-3:   12px      Small gaps between related elements
--space-4:   16px      Standard gap
--space-5:   20px      Medium gap
--space-6:   24px      Section padding
--space-8:   32px      Large section gaps
--space-10:  40px      Page section separation
--space-12:  48px      Hero section padding
--space-16:  64px      Major layout breaks
```

#### Layout Grid

```
Desktop:   12-column grid, 1200px max-width, 24px gutters, 48px side margins
Tablet:    8-column grid, 24px gutters, 32px side margins
Mobile:    4-column grid, 16px gutters, 16px side margins
```

#### Breakpoints

```
--bp-mobile:    0–639px
--bp-tablet:    640–1023px
--bp-desktop:   1024–1439px
--bp-wide:      1440px+
```

### 2.4 Elevation & Depth

```
--shadow-sm:     0 1px 2px rgba(28, 25, 23, 0.05)                    Subtle lift (cards)
--shadow-md:     0 4px 6px -1px rgba(28, 25, 23, 0.07),
                 0 2px 4px -2px rgba(28, 25, 23, 0.05)               Floating panels
--shadow-lg:     0 10px 15px -3px rgba(28, 25, 23, 0.08),
                 0 4px 6px -4px rgba(28, 25, 23, 0.04)               Modals, dropdowns
--shadow-photo:  0 2px 8px rgba(28, 25, 23, 0.12),
                 0 0 0 1px rgba(28, 25, 23, 0.04)                    Photo thumbnails (subtle frame)
--shadow-glow:   0 0 20px rgba(217, 119, 6, 0.15)                    Active/selected states
```

### 2.5 Border Radius

```
--radius-sm:     6px       Buttons, inputs, small cards
--radius-md:     10px      Cards, panels
--radius-lg:     16px      Modals, large containers
--radius-xl:     24px      Hero cards, featured elements
--radius-full:   9999px    Pills, avatars, round buttons
```

### 2.6 Motion & Animation

#### Timing Functions

```
--ease-out-expo:    cubic-bezier(0.16, 1, 0.3, 1)       Primary ease (smooth deceleration)
--ease-in-out:      cubic-bezier(0.45, 0, 0.55, 1)      Symmetric transitions
--ease-spring:      cubic-bezier(0.34, 1.56, 0.64, 1)   Playful overshoot (capture button)
```

#### Duration Scale

```
--duration-fast:    120ms    Hover states, micro-interactions
--duration-normal:  200ms    Standard transitions
--duration-slow:    350ms    Panel slides, reveals
--duration-reveal:  600ms    Before/after glare reveal
--duration-page:    500ms    Page transitions
```

#### Key Animations

**Glare reveal** (signature animation): When processing completes, the glare on the before image dissolves away over 600ms using a radial wipe emanating from the glare's center point. The glare mask drives the animation — bright areas dissolve first, creating the appearance of glare "melting" off the photo. Accompanied by a subtle warm glow pulse on the photo border.

**Capture flash:** On shutter press, a white overlay flashes at 80% opacity for 100ms, then fades over 200ms. Simultaneously, the captured thumbnail scales from 0 to 1 with spring easing and slides into the filmstrip.

**Processing progress:** A soft amber shimmer passes across the thumbnail in a loop (left-to-right gradient sweep, 2s cycle) while processing. On completion, the shimmer resolves into a green checkmark with a scale-up spring animation.

**Queue entry stagger:** When multiple items enter the processing queue, they stagger in with 60ms delay between each, sliding up with fade-in (ease-out-expo, 350ms).

---

## 3. Component Library

### 3.1 Buttons

#### Primary Button

```
Background:     --color-amber-500
Text:           white, --text-body, weight 600
Padding:        12px 24px
Border radius:  --radius-sm
Hover:          --color-amber-600, translate-y -1px, shadow-md
Active:         --color-amber-700, translate-y 0
Disabled:       opacity 0.5, no pointer events
Transition:     all --duration-fast --ease-out-expo
```

Large variant (CTAs): padding 16px 32px, --text-h3, border-radius --radius-md.

#### Secondary Button

```
Background:     transparent
Border:         1.5px solid --color-stone-300
Text:           --color-stone-700, --text-body, weight 500
Hover:          background --color-stone-50, border-color --color-stone-400
```

#### Ghost Button

```
Background:     transparent
Text:           --color-stone-500, --text-body-sm, weight 500
Hover:          text --color-stone-700, background --color-stone-100
```
Used for inline actions (rotate, flip, revert).

#### Icon Button

```
Size:           40px × 40px (touch target: 44px minimum)
Background:     transparent
Icon:           20px, --color-stone-500
Hover:          background --color-stone-100, icon --color-stone-700
Border radius:  --radius-sm
```

#### Capture Button (Special)

```
Outer ring:     64px diameter, 3px border, white (dark mode)
Inner circle:   52px diameter, white fill
Hover:          inner circle scales to 56px (spring ease)
Active (press): inner circle scales to 48px, ring border 4px
Optimal state:  outer ring pulses with amber glow (1.5s cycle)
Disabled:       opacity 0.4
```

### 3.2 Cards

#### Photo Card

```
Background:     white
Border:         1px solid --color-stone-200
Border radius:  --radius-md
Shadow:         --shadow-photo
Overflow:       hidden (image bleeds to edges)
Hover:          shadow-md, translate-y -2px
Selected:       border 2px solid --color-amber-500, shadow-glow
```

Content structure:
- Photo thumbnail (fills card width, aspect-ratio preserved).
- Below thumbnail: thin status bar (4px height, color-coded).
- On hover: semi-transparent action bar slides up from bottom (rotate, download, edit icons).

#### Queue Card

```
Background:     white
Border:         1px solid --color-stone-200
Border radius:  --radius-md
Padding:        --space-3
Layout:         horizontal — thumbnail (60px) | info column | status
```

Info column: page number (caption weight 600), photo count (caption), time estimate (caption, muted).
Status column: spinner (processing), checkmark (done), warning icon (needs attention), error icon (failed).

### 3.3 Inputs

#### Text Input

```
Background:     --color-stone-100
Border:         1.5px solid --color-stone-200
Border radius:  --radius-sm
Padding:        10px 14px
Text:           --text-body, --color-stone-900
Placeholder:    --color-stone-400
Focus:          border --color-amber-500, ring 3px --color-amber-500/20
```

#### Slider

```
Track:          4px height, --color-stone-200, --radius-full
Fill:           --color-amber-500
Thumb:          20px diameter, white, shadow-md, border 2px --color-amber-500
Hover:          thumb scales to 24px
Active:         thumb --color-amber-600
```

#### Toggle Switch

```
Track:          44px × 24px, --color-stone-300, --radius-full
Thumb:          20px diameter, white, shadow-sm
Active track:   --color-amber-500
Thumb travel:   20px horizontal slide, --duration-fast, --ease-out-expo
```

#### Radio & Checkbox

```
Radio:          20px diameter, 2px border --color-stone-300
Selected:       border --color-amber-500, 6px inner circle --color-amber-500
Checkbox:       20px square, --radius-sm, 2px border --color-stone-300
Checked:        background --color-amber-500, white checkmark icon
```

### 3.4 Overlays & Modals

#### Modal

```
Backdrop:       rgba(28, 25, 23, 0.5), backdrop-filter blur(4px)
Panel:          white, --radius-lg, --shadow-lg, max-width 560px
Padding:        --space-8
Entry:          fade in backdrop (200ms), scale panel from 0.95 (350ms, ease-out-expo)
Exit:           reverse at 200ms
```

#### Tooltip

```
Background:     --color-stone-900
Text:           white, --text-caption
Padding:        6px 10px
Border radius:  --radius-sm
Arrow:          6px CSS triangle
Delay:          400ms hover delay before showing
```

#### Toast / Notification

```
Background:     white
Border-left:    4px solid (semantic color)
Shadow:         --shadow-lg
Padding:        --space-3 --space-4
Border radius:  --radius-md
Entry:          slide in from right (350ms, ease-out-expo)
Auto-dismiss:   5 seconds, fade out (200ms)
Position:       bottom-right on desktop, bottom-center on mobile
```

### 3.5 Progress Indicators

#### Processing Shimmer

```
Gradient:       linear-gradient(90deg, transparent 0%, --color-amber-100 50%, transparent 100%)
Animation:      translate-x from -100% to 100%, 2s linear infinite
Applied to:     thumbnail container during processing
```

#### Progress Bar

```
Track:          4px height, --color-stone-200, --radius-full
Fill:           --color-amber-500, --radius-full
Animation:      width transition --duration-normal --ease-out-expo
Indeterminate:  shimmer gradient (same as above)
```

#### Stability Ring (Camera)

```
Size:           48px diameter
Stroke:         3px, white with 40% opacity (incomplete portion)
Fill stroke:    3px, white (completed portion)
Animation:      stroke-dashoffset driven by accelerometer stability (0–100%)
Full state:     pulse glow (amber) for 500ms, then auto-capture triggers
```

### 3.6 Before/After Slider

```
Container:      photo dimensions, --radius-md, overflow hidden
Divider:        2px vertical line, white, full height
Handle:         40px diameter circle, white, --shadow-lg
Handle icon:    left/right arrows (◀ ▶), --color-stone-500
Drag behavior:  constrained to container width, real-time clip-path update
Touch target:   handle itself + 20px invisible padding
Default pos:    50% (centered)
```

The "before" image is on the left, "after" on the right. Dragging the handle clips the "after" layer to reveal the "before" underneath (or vice versa). On mobile, swipe gesture works anywhere on the image (no need to grab the handle precisely).

---

## 4. Screen Designs — Detailed Specifications

### 4.1 Landing Page

#### Layout (Desktop)

```
┌────────────────────────────────────────────────────────────────────┐
│  NAVBAR                                                            │
│  [Logo: "Album Digitizer" in Fraunces]         [How It Works] [FAQ]│
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     HERO SECTION                            │   │
│  │                                                             │   │
│  │  "Digitize your photo albums.                               │   │
│  │   No glare. No hassle. Free."                               │   │
│  │          (Fraunces, 48px, --color-stone-900)                │   │
│  │                                                             │   │
│  │  Subtitle: "Remove reflections from glossy sleeves,         │   │
│  │  split multi-photo pages, restore faded colors —            │   │
│  │  all automatically."                                        │   │
│  │          (DM Sans, 18px, --color-stone-500)                 │   │
│  │                                                             │   │
│  │  [  Start with Camera  ]  [  Upload Photos  ]               │   │
│  │  (primary button, large)  (secondary button, large)         │   │
│  │                                                             │   │
│  │  ┌─────────────────────────────────────────────┐            │   │
│  │  │  INTERACTIVE BEFORE/AFTER                    │            │   │
│  │  │                                              │            │   │
│  │  │  [Album page with glare] ◀──▶ [Clean photos] │            │   │
│  │  │  (draggable slider comparison)               │            │   │
│  │  │  Max width: 700px, centered                  │            │   │
│  │  └─────────────────────────────────────────────┘            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  HOW IT WORKS                               │   │
│  │                                                             │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │   │
│  │  │  ① 📸    │    │  ② ✨    │    │  ③ 💾    │              │   │
│  │  │ Capture  │───▶│ Process  │───▶│ Download │              │   │
│  │  │          │    │          │    │          │              │   │
│  │  │ Photo or │    │ Glare,   │    │ Clean    │              │   │
│  │  │ upload   │    │ split,   │    │ photos,  │              │   │
│  │  │ album    │    │ color,   │    │ ready to │              │   │
│  │  │ pages    │    │ geometry │    │ share    │              │   │
│  │  └──────────┘    └──────────┘    └──────────┘              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               FEATURE HIGHLIGHTS                            │   │
│  │                                                             │   │
│  │  ┌─────────────────────┐  ┌─────────────────────┐          │   │
│  │  │ Glare Removal       │  │ Auto Photo Splitting │          │   │
│  │  │ [before/after mini] │  │ [animation: page     │          │   │
│  │  │                     │  │  splits into photos]  │          │   │
│  │  └─────────────────────┘  └─────────────────────┘          │   │
│  │  ┌─────────────────────┐  ┌─────────────────────┐          │   │
│  │  │ Perspective Fix     │  │ Color Restoration    │          │   │
│  │  │ [trapezoid →        │  │ [faded → vibrant     │          │   │
│  │  │  rectangle anim]    │  │  comparison]         │          │   │
│  │  └─────────────────────┘  └─────────────────────┘          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  FOOTER: Privacy Policy | FAQ | GitHub | "Made with care for your  │
│  family memories"                                                  │
└────────────────────────────────────────────────────────────────────┘
```

#### Hero Before/After Details

The hero comparison is the most important element on the landing page. It shows a real album page with severe glare on the left transforming into clean, split individual photos on the right.

- The before side shows: a glossy album page with visible glare hotspot, 3 photos behind the sleeve.
- The after side shows: the same 3 photos, individually extracted, perspective-corrected, colors restored, and glare-free.
- The slider auto-animates slowly on page load (slides from 30% to 70% over 3 seconds, then settles at 50%) to draw attention. After the initial animation, it becomes user-draggable.
- On mobile, the comparison is stacked vertically with a swipe gesture instead of a slider.

#### Feature Highlight Cards

Each card contains a looping micro-animation (5–8 seconds, no sound):
- **Glare Removal:** The glare "melts" off a photo (the signature animation).
- **Photo Splitting:** Detected boundaries animate in, then photos slide apart into a grid.
- **Perspective Fix:** A trapezoid morphs into a rectangle.
- **Color Restoration:** A faded sepia photo blooms into full color.

These animations use CSS/JS — not videos — for instant loading and crisp rendering.

#### Mobile Layout

- Stacked single-column layout.
- Hero text reduces to 32px.
- CTAs stack vertically (camera on top, upload below).
- Before/after becomes swipe-based (tap to toggle or swipe).
- Feature cards stack in a 1-column scrollable list.

---

### 4.2 Capture Screen — Phone Camera

#### Layout (Full-Screen, Dark Mode)

```
┌────────────────────────────────────────┐
│  STATUS BAR (translucent dark)         │
│  Page 5        ⚙️ Settings      ✕ Close│
├────────────────────────────────────────┤
│                                        │
│                                        │
│     ┌────────────────────────┐         │
│     │                        │         │
│     │    CAMERA VIEWFINDER   │         │
│     │                        │         │
│     │   ╔════════════════╗   │         │
│     │   ║  Album Page    ║   │         │
│     │   ║  (green        ║   │         │
│     │   ║   boundary)    ║   │         │
│     │   ║                ║   │         │
│     │   ║  ▓▓ glare      ║   │         │
│     │   ║  ▓▓ overlay    ║   │         │
│     │   ║  (orange)      ║   │         │
│     │   ╚════════════════╝   │         │
│     │                        │         │
│     └────────────────────────┘         │
│                                        │
│  "Tilt slightly right to reduce glare" │
│  (guidance text, amber, --text-body-sm)│
│                                        │
├────────────────────────────────────────┤
│  BOTTOM CONTROLS                       │
│                                        │
│  [📷 last]     ◉ CAPTURE      [mode]  │
│  (thumbnail)  (capture btn)   toggle   │
│               (64px)                   │
│                                        │
│  ●───────── Single ──── Multi-Shot     │
│  (mode toggle, pill selector)          │
│                                        │
│  ┌──┐┌──┐┌──┐┌──┐┌──┐                │
│  │t1││t2││t3││t4││t5│ ← filmstrip     │
│  └──┘└──┘└──┘└──┘└──┘   of captures   │
└────────────────────────────────────────┘
```

#### Overlay System (Layered on Camera Feed)

**Layer 1 — Page boundary detection:**
- A rounded-corner quadrilateral drawn as a 2px stroke overlaying the detected album page edges.
- Color: green (`--color-success`, 60% opacity) when all four edges are in frame.
- Color: red (`--color-error`, 60% opacity) when any edge extends beyond the frame.
- Color: yellow (`--color-warning`, 60% opacity) when detection is uncertain.
- Outside the quadrilateral: subtle dark vignette (10% opacity black) to focus attention on the album page.

**Layer 2 — Glare heat map:**
- Semi-transparent orange-red gradient overlay on regions with detected glare.
- Intensity maps to glare severity: light orange (minor) → deep red (severe).
- Togglable via a small eye icon in the top-right corner.
- Animates smoothly as the user tilts the phone (glare regions shift).

**Layer 3 — Guidance text:**
- Positioned below the viewfinder area (above the capture controls).
- Context-aware messages:
  - "Hold steady..." (stability < 50%)
  - "Tilt slightly [direction] to reduce glare" (glare detected, with directional arrow icon)
  - "Move closer — album page should fill the frame" (page too small)
  - "Ready! Tap to capture" (optimal conditions met)
  - "Great capture!" (post-capture confirmation, shown for 1 second)
- Text is DM Sans, 14px, white on dark mode, with a subtle text-shadow for readability over any camera content.

#### Capture Button Behavior

**Resting state:** White circle (52px) inside white ring (64px, 3px border). Subtle breathing animation (ring opacity pulses 80%–100%, 2s cycle).

**Optimal conditions met:** Ring gains an amber glow (box-shadow pulse). If auto-capture is enabled, a 1.5-second countdown appears inside the ring as a circular fill animation, then auto-fires.

**Press:** Inner circle shrinks to 48px (spring ease). Simultaneously, the screen flashes white (80% opacity, 100ms). The captured frame freezes for 200ms, then a thumbnail of the capture slides (ease-out-expo, 300ms) from the center into the filmstrip at the bottom.

**Multi-shot mode active:** The ring shows a progress arc (e.g., "2/4" with the arc 50% filled). Each subsequent capture adds to the arc. After the final shot, a "compositing..." message appears briefly.

#### Filmstrip (Bottom)

- Horizontal scrollable strip of captured page thumbnails (48px × 48px, --radius-sm).
- Most recent capture on the left with a green border pulse (300ms) on entry.
- Tap any thumbnail to preview the full capture in a quick-look overlay.
- Long-press to delete (with confirmation).
- Counter badge on the last thumbnail: "12 pages."

#### Mode Toggle

- Pill-shaped selector between "Single" and "Multi-Shot (Glare)."
- "Multi-Shot" has a small sparkle icon to indicate it's the premium quality option.
- When switching to multi-shot, a brief tooltip animates in: "Take 3–5 shots at slightly different angles. We'll combine them for the best glare removal."

#### Settings Panel (Gear Icon)

Slides in as a half-sheet from the bottom (mobile) or a dropdown panel (desktop).

Contents:
- **Auto-capture:** toggle (on/off). When on, captures automatically when stability + framing conditions are met.
- **Camera selector:** front/rear toggle (defaults to rear).
- **Flash:** off / auto / on toggle.
- **Resolution:** "Standard" / "Maximum" radio. Standard uses the camera's default; Maximum forces highest available resolution (may be slower).
- **Grid overlay:** toggle to show a rule-of-thirds grid (for alignment aid).

---

### 4.3 Capture Screen — Desktop Webcam

#### Layout (Desktop, Dark Mode)

```
┌──────────────────────────────────────────────────────────────────┐
│  NAVBAR (dark)                                                    │
│  [← Back]   Album Digitizer   [Settings ⚙️]                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────┐  ┌────────────────────────┐ │
│  │                                  │  │  CAPTURE QUEUE         │ │
│  │       WEBCAM VIEWFINDER          │  │                        │ │
│  │                                  │  │  Page 1  ✅  0.8s      │ │
│  │   ╔══════════════════════╗       │  │  ┌────┐               │ │
│  │   ║                      ║       │  │  │ t1 │ 3 photos      │ │
│  │   ║    Album Page        ║       │  │  └────┘               │ │
│  │   ║    (green boundary)  ║       │  │                        │ │
│  │   ║                      ║       │  │  Page 2  ✅  1.2s      │ │
│  │   ║    ▓▓ glare areas    ║       │  │  ┌────┐               │ │
│  │   ║                      ║       │  │  │ t2 │ 2 photos      │ │
│  │   ╚══════════════════════╝       │  │  └────┘               │ │
│  │                                  │  │                        │ │
│  │  "Press SPACE to capture"        │  │  Page 3  ⏳ processing │ │
│  │  (guidance text, centered below) │  │  ┌────┐               │ │
│  │                                  │  │  │ t3 │ shimmer...    │ │
│  │  ○ Stability     ◉ CAPTURE      │  │  └────┘               │ │
│  │    ring          (space bar)     │  │                        │ │
│  │                                  │  │  ────────────────────  │ │
│  │  Camera: [Razer Kiyo Pro ▼]      │  │  12 pages captured    │ │
│  │  Mode:   ● Single ○ Multi-Shot   │  │  9 processed          │ │
│  │                                  │  │  [Process Remaining]   │ │
│  └──────────────────────────────────┘  └────────────────────────┘ │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

Key differences from phone capture:
- Side panel shows the scrollable capture queue (instead of a bottom filmstrip).
- Camera selector dropdown is always visible (users may have multiple webcams).
- Keyboard shortcut (SPACE) prominently displayed — faster than clicking.
- Wider viewport means the viewfinder and queue panel coexist side by side.

---

### 4.4 Upload Screen

#### Layout (Desktop, Light Mode)

```
┌──────────────────────────────────────────────────────────────────┐
│  NAVBAR                                                           │
│  [← Back]   Album Digitizer                                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                                                              │ │
│  │           ┌──────────────────────────────┐                   │ │
│  │           │                              │                   │ │
│  │           │    ☁️ (upload cloud icon)      │                   │ │
│  │           │                              │                   │ │
│  │           │  Drop your album page        │                   │ │
│  │           │  photos here                 │                   │ │
│  │           │                              │                   │ │
│  │           │  JPEG, PNG, HEIF up to 50 MB │                   │ │
│  │           │                              │                   │ │
│  │           │  [ Browse Files ]            │                   │ │
│  │           │  (secondary button)          │                   │ │
│  │           └──────────────────────────────┘                   │ │
│  │           Dashed border, 2px, --color-stone-300              │ │
│  │           On drag-over: border --color-amber-500,            │ │
│  │           background --color-amber-50                        │ │
│  │                                                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  UPLOAD STRIP (appears after files are added)                │ │
│  │                                                              │ │
│  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐  +12 more      │ │
│  │  │ f1 │ │ f2 │ │ f3 │ │ f4 │ │ f5 │ │ f6 │               │ │
│  │  │ ✕  │ │ ✕  │ │ ✕  │ │ ✕  │ │ ✕  │ │ ✕  │               │ │
│  │  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘               │ │
│  │  18 files selected (247 MB total)          [+ Add More]     │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌────────────────────────────────────────┐                       │
│  │  BATCH SETTINGS                        │                       │
│  │                                        │                       │
│  │  Album name:  [ My Family Album  ]     │                       │
│  │  (optional, used in file naming)       │                       │
│  │                                        │                       │
│  │  Output format:                        │                       │
│  │  ● JPEG (90%)  ○ PNG  ○ TIFF           │                       │
│  │  Quality: [━━━━━━━━●━━] 90%            │                       │
│  │                                        │                       │
│  │  ☐ Also export uncorrected versions    │                       │
│  └────────────────────────────────────────┘                       │
│                                                                    │
│          [   Process All (18 pages)   ]                           │
│          (primary button, large, centered)                        │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

#### Drop Zone Behavior

**Default:** Dashed border (2px, --color-stone-300), light stone background. Upload icon and text centered.

**Drag over (files hovering):** Border animates to solid --color-amber-500. Background shifts to --color-amber-50. Icon bounces gently (spring animation). Text changes to "Drop to add."

**Files added:** Drop zone shrinks to a smaller "Add more" target. The upload strip expands below showing thumbnails. Each thumbnail has a small "×" button to remove it.

**Validation errors:** If an unsupported file format is dropped, a toast notification appears: "Some files were skipped (unsupported format). We accept JPEG, PNG, and HEIF."

#### Mobile Upload Layout

- Drop zone simplified to a single "Select Photos" button (drag-and-drop not reliable on mobile).
- Opens native photo picker with multi-select.
- Upload strip becomes a vertical scrollable list of thumbnails.
- Batch settings collapse into an expandable section.

---

### 4.5 Processing Queue Screen

#### Layout (Desktop, Light Mode)

```
┌──────────────────────────────────────────────────────────────────┐
│  NAVBAR                                                           │
│  [← Back]   Processing — 47 pages                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  PROGRESS SUMMARY BAR                                        │ │
│  │                                                              │ │
│  │  [━━━━━━━━━━━━━━━━━●━━━━━━━━━━] 62%                          │ │
│  │  29 of 47 pages complete · ~2 min remaining                  │ │
│  │                                     [Pause All] [Cancel All] │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  QUEUE LIST                                                  │ │
│  │                                                              │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │ ✅ Page 1 · 3 photos extracted · 0.8s                   │ │ │
│  │  │ ┌────┐ ┌────┐ ┌────┐                                   │ │ │
│  │  │ │ p1 │ │ p2 │ │ p3 │  (extracted photo thumbnails)     │ │ │
│  │  │ └────┘ └────┘ └────┘                    [View Results]  │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                                                              │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │ ⏳ Page 2 · Processing: Color Restoration               │ │ │
│  │  │ [━━━━━━━━━━━━━━━●━━━] 75%                               │ │ │
│  │  │ (shimmer animation on thumbnail)                        │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                                                              │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │ ⚠️ Page 3 · 2 photos · Glare confidence: Low            │ │ │
│  │  │ ┌────┐ ┌────┐                                          │ │ │
│  │  │ │ p1 │ │ p2 │  "Consider re-shooting this page"        │ │ │
│  │  │ └────┘ └────┘               [View Results] [Re-shoot]  │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                                                              │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │ ⬚ Page 4 · Queued                                      │ │ │
│  │  │ (muted appearance, waiting)                             │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │  ...                                                        │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  BOTTOM ACTION BAR                                           │ │
│  │  29 pages complete (87 photos)                               │ │
│  │                          [View All Photos]  [Export All ↓]   │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

#### Queue Item States

**Queued (waiting):** Muted appearance (--color-stone-400 text). Dotted border. Page number and "Queued" label.

**Processing (active):** Normal text color. Amber shimmer animation on the row background. Step indicator showing which pipeline step is active (e.g., "Glare Removal" → "Photo Detection" → "Color Restoration"). Per-page progress bar.

**Complete (success):** Green checkmark. Extracted photo thumbnails appear inline (animated in with stagger). Processing time shown. "View Results" link.

**Attention needed:** Orange warning icon. Specific message (e.g., "Low glare removal confidence" or "Unusual photo layout detected — please review"). "View Results" and "Re-shoot" buttons.

**Error:** Red error icon. Error message (e.g., "Could not detect any photos on this page"). "Retry" and "Delete" buttons.

#### Real-Time Updates

As processing completes, rows animate from "processing" to "complete" state. The extracted thumbnails fade in with a 60ms stagger between each. The progress summary bar at the top updates in real-time. A subtle chime sound (optional, off by default) plays when a batch completes.

---

### 4.6 Page Results View

#### Layout (Desktop, Light Mode)

```
┌──────────────────────────────────────────────────────────────────┐
│  NAVBAR                                                           │
│  [← Queue]   Page 5 of 47   [◀ Prev]  [Next ▶]                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌────────────────────────────┐  ┌─────────────────────────────┐ │
│  │                            │  │  EXTRACTED PHOTOS            │ │
│  │  ORIGINAL ALBUM PAGE       │  │                             │ │
│  │                            │  │  ┌───────────────────────┐  │ │
│  │  ┌──────────────────────┐  │  │  │                       │  │ │
│  │  │                      │  │  │  │   Photo 1 (large)     │  │ │
│  │  │  (full resolution    │  │  │  │   Before/After slider │  │ │
│  │  │   original with      │  │  │  │                       │  │ │
│  │  │   detected photo     │  │  │  │   ✅ High confidence   │  │ │
│  │  │   boundaries drawn   │  │  │  │   [↻] [⤓] [✎]        │  │ │
│  │  │   as colored          │  │  │  └───────────────────────┘  │ │
│  │  │   rectangles)        │  │  │                             │ │
│  │  │                      │  │  │  ┌───────────┐ ┌─────────┐ │ │
│  │  │  Photo boundaries:   │  │  │  │ Photo 2   │ │ Photo 3 │ │ │
│  │  │  #1 = blue           │  │  │  │ B/A slider│ │ B/A     │ │ │
│  │  │  #2 = green          │  │  │  │           │ │ slider  │ │ │
│  │  │  #3 = purple         │  │  │  │ ⚠️ Low    │ │ ✅ Good  │ │ │
│  │  │                      │  │  │  │ [↻][⤓][✎]│ │[↻][⤓][✎]│ │ │
│  │  └──────────────────────┘  │  │  └───────────┘ └─────────┘ │ │
│  │                            │  │                             │ │
│  │  Zoomable (scroll to zoom, │  │  Photos sized proportional │ │
│  │  click-drag to pan)        │  │  to their actual dimensions│ │
│  └────────────────────────────┘  └─────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    [Accept & Next →]                          │ │
│  │                    (primary button, centered)                │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

#### Photo Boundary Overlay

On the original album page (left panel), the detected photo boundaries are drawn as colored rectangles with matching numbers. Colors are assigned in sequence (blue, green, purple, orange) to distinguish multiple photos. The boundary lines are 2px with a subtle glow. Clicking a boundary on the original highlights the corresponding extracted photo on the right (and vice versa).

#### Per-Photo Actions

Each extracted photo card shows:
- **Before/after slider:** Compact version (the user can drag to compare).
- **Confidence badge:** ✅ green "High confidence" or ⚠️ orange "Low confidence — review recommended."
- **Quick action icons:**
  - ↻ Rotate 90° (cycles through 0°, 90°, 180°, 270° on each click).
  - ⤓ Download this photo individually.
  - ✎ Open in Photo Editor (navigates to the detailed editor view).

#### Confidence Overlay

For photos with low glare-removal confidence, the photo shows a subtle pulsing orange border. Tapping the ⚠️ badge reveals a tooltip: "Some glare may remain in this photo. Open the editor for manual adjustments, or re-shoot this page with multi-shot mode."

---

### 4.7 Photo Grid View

#### Layout (Desktop, Light Mode)

```
┌──────────────────────────────────────────────────────────────────┐
│  NAVBAR                                                           │
│  [← Queue]   All Photos                                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  TOOLBAR                                                     │ │
│  │  142 photos from 47 pages                                    │ │
│  │                                                              │ │
│  │  Sort: [Page Order ▼]   Filter: [All ▼]   [☐ Select Mode]   │ │
│  │                                                              │ │
│  │  Sort options: Page Order, Quality (best first),             │ │
│  │                Needs Attention                               │ │
│  │  Filter options: All, Needs Review, High Confidence Only     │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  PHOTO GRID (masonry layout)                                 │ │
│  │                                                              │ │
│  │  ┌─────┐ ┌────────┐ ┌─────┐ ┌─────┐ ┌────────┐             │ │
│  │  │     │ │        │ │     │ │     │ │        │             │ │
│  │  │     │ │  land- │ │     │ │     │ │  land- │             │ │
│  │  │port-│ │  scape │ │port-│ │port-│ │  scape │             │ │
│  │  │rait │ │        │ │rait │ │rait │ │        │             │ │
│  │  │     │ └────────┘ │     │ │     │ └────────┘             │ │
│  │  │     │ ┌─────┐    │     │ │     │ ┌─────┐               │ │
│  │  └─────┘ │     │    └─────┘ └─────┘ │     │               │ │
│  │  ┌────────┐    │    ┌────────┐       │     │               │ │
│  │  │  land- │    │    │  land- │       └─────┘               │ │
│  │  │  scape │    │    │  scape │       ┌────────┐            │ │
│  │  └────────┘    │    └────────┘       │  land- │            │ │
│  │          └─────┘                     └────────┘            │ │
│  │                                                              │ │
│  │  Each photo card: thumbnail, page label ("P3-2"),            │ │
│  │  confidence dot (green/orange), hover → action icons         │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  BOTTOM BAR                                                  │ │
│  │  142 photos ready                [Export All (ZIP) ↓]        │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

#### Grid Behavior

- **Masonry layout:** Photos maintain their original aspect ratio. Columns adapt to viewport width (4 columns desktop, 3 tablet, 2 mobile).
- **Photo labels:** Each thumbnail shows a small label in the bottom-left corner: "P3-2" means Page 3, Photo 2.
- **Hover state:** Photo scales up 2% with shadow-md. Action icons (rotate, download, edit) appear as a semi-transparent bar across the bottom of the thumbnail.
- **Click:** Opens the Photo Editor for that image.

#### Select Mode

When "Select Mode" is toggled on:
- Each photo shows a checkbox in the top-left corner.
- Clicking a photo toggles selection (instead of opening the editor).
- A floating action bar appears at the bottom: "X selected — [Download] [Re-process] [Delete]."
- "Select all" and "Deselect all" links appear in the toolbar.

#### Mobile Layout

- 2-column grid with uniform aspect ratio (square crops for grid consistency; full aspect ratio on tap).
- Swipe left on a photo card to reveal actions (download, edit, delete).
- Bottom bar becomes sticky and scrolls with the grid.

---

### 4.8 Individual Photo Editor

#### Layout (Desktop, Light Mode)

```
┌──────────────────────────────────────────────────────────────────┐
│  NAVBAR                                                           │
│  [← Back to Grid]   Page 5, Photo 2                [Before/After]│
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌────────────────────────────────────┐ ┌──────────────────────┐ │
│  │                                    │ │  ADJUSTMENTS         │ │
│  │                                    │ │                      │ │
│  │                                    │ │  ▸ Crop & Geometry   │ │
│  │                                    │ │  ┌──────────────────┐│ │
│  │         PHOTO CANVAS               │ │  │ [Auto-detect]    ││ │
│  │                                    │ │  │                  ││ │
│  │     (full-resolution photo         │ │  │ Rotation:        ││ │
│  │      with current corrections      │ │  │ [━━━●━━━] 0.0°   ││ │
│  │      applied)                      │ │  │                  ││ │
│  │                                    │ │  │ [90° ↻] [90° ↺]  ││ │
│  │     Zoom: scroll wheel             │ │  │ [Flip H] [Flip V]││ │
│  │     Pan: click-drag                │ │  │                  ││ │
│  │     Crop handles visible when      │ │  │ Perspective:     ││ │
│  │     Crop section is expanded       │ │  │ [Fix] [Manual 4pt]│ │
│  │                                    │ │  └──────────────────┘│ │
│  │                                    │ │                      │ │
│  │                                    │ │  ▸ Color             │ │
│  │                                    │ │  ┌──────────────────┐│ │
│  │                                    │ │  │ Auto enhance: [●]││ │
│  │                                    │ │  │                  ││ │
│  │                                    │ │  │ Exposure:        ││ │
│  │                                    │ │  │ [━━━●━━━] +0.0   ││ │
│  │                                    │ │  │                  ││ │
│  │                                    │ │  │ White balance:   ││ │
│  │                                    │ │  │ [━━━●━━━] 0      ││ │
│  │                                    │ │  │ (cool ← → warm)  ││ │
│  │                                    │ │  │                  ││ │
│  │                                    │ │  │ Saturation:      ││ │
│  │                                    │ │  │ [━━━●━━━] +0     ││ │
│  │                                    │ │  │                  ││ │
│  │                                    │ │  │ Restore fading:  ││ │
│  │                                    │ │  │ [●] on           ││ │
│  │                                    │ │  └──────────────────┘│ │
│  │                                    │ │                      │ │
│  │                                    │ │  ▸ Glare             │ │
│  │                                    │ │  ┌──────────────────┐│ │
│  │                                    │ │  │ Confidence: 72%  ││ │
│  │                                    │ │  │ Show map: [○]    ││ │
│  │                                    │ │  │ [Re-process]     ││ │
│  │                                    │ │  │ [Paint glare ✎]  ││ │
│  │                                    │ │  └──────────────────┘│ │
│  │                                    │ │                      │ │
│  └────────────────────────────────────┘ │──────────────────────│ │
│                                         │ [Revert to Auto]     │ │
│                                         │ [Save] [Download ⤓]  │ │
│                                         └──────────────────────┘ │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

#### Adjustment Panel Sections

Each section is collapsible (accordion style). Only one section is expanded at a time to keep the panel from getting too long. Sections are:

**Crop & Geometry:**
- "Auto-detect edges" button — re-runs the boundary detection and perspective correction.
- Rotation slider: -15° to +15°, with 0.1° precision. Dragging the slider shows a grid overlay on the photo for alignment reference.
- 90° rotation buttons (clockwise, counter-clockwise).
- Flip buttons (horizontal, vertical).
- "Fix Perspective" button — applies automatic perspective correction.
- "Manual 4-point" button — switches to a mode where the user drags four corner handles on the photo to define the perspective transform.

**Color:**
- "Auto enhance" master toggle — when on, all color corrections are applied. When off, shows the uncorrected (but geometry-fixed) version.
- Exposure slider: -2.0 to +2.0 EV, default 0.
- White balance slider: cool (-100) to warm (+100), default 0.
- Saturation slider: -100 to +100, default 0.
- "Restore fading" toggle — specifically targets the histogram compression and desaturation common in aged prints.
- All sliders update the photo canvas in real-time (debounced at 50ms).

**Glare:**
- Confidence percentage — how confident the system is that glare was fully removed.
- "Show confidence map" toggle — overlays a heat map on the photo showing areas of remaining glare.
- "Re-process" button — re-runs glare removal with different parameters (useful if multi-shot data is available).
- "Paint glare mask" button — enters a mode where the user can paint over areas they identify as remaining glare. On save, those specific regions are re-processed.

#### Before/After Toggle

The "[Before/After]" button in the navbar toggles a full-canvas before/after comparison. Clicking it activates the draggable slider across the entire photo canvas. The "before" is the original unprocessed crop; the "after" is the current corrected version.

#### Mobile Editor Layout

- Photo canvas takes the full viewport width.
- Adjustment panel becomes a bottom sheet (swipe up to expand, swipe down to minimize).
- Bottom sheet shows only the section titles in minimized state. Tap a section to expand it.
- Before/after toggle becomes a tap-and-hold gesture on the photo (hold to see before, release for after).

---

### 4.9 Export Screen

#### Layout (Desktop, Light Mode)

```
┌──────────────────────────────────────────────────────────────────┐
│  NAVBAR                                                           │
│  [← Back]   Export                                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  EXPORT SUMMARY                                              │ │
│  │                                                              │ │
│  │  Ready to export 142 photos from 47 pages                    │ │
│  │                                                              │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │ │
│  │  │    142      │  │     47      │  │   ~850 MB   │          │ │
│  │  │   photos    │  │    pages    │  │  estimated  │          │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘          │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  OUTPUT SETTINGS                                             │ │
│  │                                                              │ │
│  │  Format:        ● JPEG    ○ PNG    ○ TIFF                    │ │
│  │                                                              │ │
│  │  JPEG Quality:  [━━━━━━━━━━━━●━━━] 90%                      │ │
│  │                 (70%=smaller files ← → 100%=highest quality) │ │
│  │                                                              │ │
│  │  AI Upscaling:  [○] Off                                     │ │
│  │                 [ ] 2× (doubles resolution)                  │ │
│  │                 [ ] 4× (quadruples resolution)               │ │
│  │                                                              │ │
│  │  Include originals:  [○]                                     │ │
│  │  (Also export uncorrected versions alongside enhanced)       │ │
│  │                                                              │ │
│  │  ────────────────────────────────────────────                │ │
│  │                                                              │ │
│  │  File naming:                                                │ │
│  │  Album name: [ My Family Album ]                             │ │
│  │  Preview:    My_Family_Album_Page03_Photo02.jpg              │ │
│  │  (album_name + page number + photo number + format)          │ │
│  │                                                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│                                                                    │
│          ┌─────────────────────────────────────────┐              │
│          │     Download All (ZIP) — ~850 MB         │              │
│          │     (primary button, extra-large)        │              │
│          └─────────────────────────────────────────┘              │
│                                                                    │
│  Or download by page:                                             │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  Page 1 (3 photos, 18 MB)                        [Download] │ │
│  │  Page 2 (2 photos, 12 MB)                        [Download] │ │
│  │  Page 3 (4 photos, 24 MB)                        [Download] │ │
│  │  ...                                                        │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

#### Download Progress

When "Download All" is clicked:
- The button transforms into a progress bar (same dimensions, same position).
- Progress bar fills left-to-right as the ZIP is generated server-side.
- Text updates: "Preparing your photos..." → "Generating ZIP (62%)..." → "Download starting..."
- Once ready, the browser's native download dialog appears.
- Button returns to its original state with text: "Download Again."

---

## 5. Navigation & Flow

### 5.1 Primary User Flow

```
Landing Page
    │
    ├── "Start with Camera" ──▶ Capture Screen (phone/webcam)
    │                              │
    │                              ├── Capture pages (loop)
    │                              │
    │                              └── "Done Capturing" ──▶ Processing Queue
    │
    ├── "Upload Photos" ──▶ Upload Screen
    │                          │
    │                          └── "Process All" ──▶ Processing Queue
    │
    └── Processing Queue
            │
            ├── Page complete ──▶ Page Results View
            │                        │
            │                        ├── Click photo ──▶ Photo Editor
            │                        │                      │
            │                        │                      └── Save ──▶ Back to Results
            │                        │
            │                        └── "Accept & Next" ──▶ Next Page Results
            │
            ├── "View All Photos" ──▶ Photo Grid View
            │                            │
            │                            └── Click photo ──▶ Photo Editor
            │
            └── "Export All" ──▶ Export Screen
                                    │
                                    └── Download ──▶ Done
```

### 5.2 Navigation Bar

The navbar is persistent across all app screens (not on the landing page).

**Desktop navbar:**
```
[← Back]   Album Digitizer   [Capture] [Upload] [Queue (12)] [Photos (87)] [Export]
```
- "Back" returns to the previous screen (browser history based).
- Tabs show counts where relevant (queue count, photo count).
- Active tab has an amber underline.

**Mobile navbar:**
- Simplified to: [← Back] and a hamburger menu (☰) for screen navigation.
- The hamburger opens a bottom sheet with the navigation options.

### 5.3 Session Indicator

A small, persistent element in the navbar corner (desktop) or at the top of the hamburger menu (mobile):

```
Session: active · 6 days remaining · [Delete Session]
```

This reminds users their data is temporary and shows time until auto-deletion.

---

## 6. Responsive Behavior Summary

### 6.1 Breakpoint Adaptations

| Element | Desktop (1024+) | Tablet (640–1023) | Mobile (< 640) |
|---------|-----------------|-------------------|----------------|
| Landing hero | Side-by-side text + comparison | Stacked | Stacked, smaller type |
| Capture viewfinder | 60% width + side queue panel | Full width + bottom filmstrip | Full screen |
| Upload drop zone | Large centered zone | Full width | "Select" button only |
| Processing queue | List with inline thumbnails | List with smaller thumbnails | Compact list, tap to expand |
| Page results | Side-by-side original + extracted | Stacked (original on top) | Stacked, swipeable |
| Photo grid | 4-column masonry | 3-column | 2-column |
| Photo editor | Side-by-side canvas + panel | Canvas on top, panel below | Full-width canvas + bottom sheet |
| Export | Centered card layout | Full-width | Full-width, stacked |

### 6.2 Touch Targets

All interactive elements have a minimum touch target of 44×44px on mobile, per WCAG guidelines. This applies to buttons, checkboxes, sliders, and icon buttons. Where the visual element is smaller than 44px (e.g., a 20px checkbox), invisible padding extends the touch target.

### 6.3 Gestures (Mobile)

| Gesture | Context | Action |
|---------|---------|--------|
| Tap | Capture button | Take photo |
| Long press | Filmstrip thumbnail | Delete captured page |
| Swipe left | Photo grid card | Reveal action buttons |
| Swipe left/right | Page results | Navigate between pages |
| Pinch to zoom | Photo editor, page results | Zoom into photo |
| Two-finger drag | Photo editor (zoomed) | Pan photo |
| Tap and hold | Photo editor | Show "before" version |

---

## 7. Accessibility

### 7.1 Keyboard Navigation

All interactive elements are reachable via Tab key. Focus order follows visual layout (top-to-bottom, left-to-right). Focus indicator: 2px amber outline with 3px offset (clearly visible on all backgrounds).

Key shortcuts (desktop):
- **Space:** Capture photo (on capture screen).
- **←/→ arrows:** Navigate between pages (on results view).
- **B:** Toggle before/after comparison.
- **Esc:** Close modal, exit editor, dismiss notification.

### 7.2 Screen Reader Support

- All images have descriptive alt text generated during processing (e.g., "Extracted photo 2 of 3 from album page 5").
- Processing status updates are announced via ARIA live regions.
- The before/after slider is labeled: "Comparison slider. Drag to compare original and processed photo."
- Confidence indicators are announced: "Glare removal confidence: 72 percent. Review recommended."

### 7.3 Color Contrast

All text meets WCAG AA contrast ratios (4.5:1 for normal text, 3:1 for large text). Semantic colors (success, warning, error) are paired with icons so information is not conveyed by color alone. The glare overlay heat map uses both color and opacity variation.

### 7.4 Reduced Motion

When the user's system preference is `prefers-reduced-motion: reduce`, all animations are disabled or replaced with instant transitions. The signature "glare reveal" animation becomes a simple crossfade. The processing shimmer becomes a static progress bar.

---

## 8. Error States & Empty States

### 8.1 Error States

**Camera permission denied:**
- Full-screen message: "Camera access is needed to capture album pages."
- "How to enable camera access" expandable instructions (per browser).
- "Upload photos instead" fallback button.

**Upload failed:**
- Toast notification: "Upload failed for [filename]. Tap to retry."
- Failed file shows a red border in the upload strip with a retry icon.

**Processing failed:**
- Queue item shows red error state with specific message.
- "Retry" button attempts reprocessing.
- "Skip" button removes the page and continues with the rest of the batch.
- If repeated failures: "Something's not working right. Try uploading a different photo or contact us."

**No photos detected:**
- Page results show the original with a message: "We couldn't detect any individual photos on this page. This might be a full-page photo, or the layout is unusual."
- Options: "Treat entire page as one photo" or "Draw photo boundaries manually."

**Network error:**
- Overlay banner: "Connection lost. Your captures are saved locally and will be uploaded when you're back online." (Phase 2, PWA mode.)

### 8.2 Empty States

**No captures yet (capture screen):**
- Filmstrip shows: "Your captured pages will appear here."

**No photos yet (grid view):**
- Centered illustration (a simple line drawing of an open photo album) with text: "No photos yet. Start by capturing or uploading your album pages."

**Empty queue:**
- Centered text: "Nothing to process. Capture or upload album pages to get started."

---

## 9. Iconography

Use **Lucide Icons** (open-source, consistent style) throughout the application. 20px default size, 1.5px stroke weight.

Key icons and their usage:

| Icon | Lucide Name | Usage |
|------|-------------|-------|
| Camera | `camera` | Capture mode navigation, capture CTA |
| Upload | `upload-cloud` | Upload mode navigation, drop zone |
| Image | `image` | Photo grid navigation, photo placeholder |
| Download | `download` | Download buttons, export |
| Rotate CW | `rotate-cw` | Rotate photo clockwise |
| Rotate CCW | `rotate-ccw` | Rotate photo counter-clockwise |
| Edit | `pencil` | Open photo editor |
| Check | `check` | Processing complete, confirmed state |
| Alert triangle | `alert-triangle` | Warning, low confidence |
| X circle | `x-circle` | Error, remove, close |
| Settings | `settings` | Settings panel |
| Eye | `eye` / `eye-off` | Toggle overlay visibility |
| Maximize | `maximize-2` | Full-screen / zoom |
| Grid | `grid-3x3` | Photo grid view |
| Layers | `layers` | Page results view |
| Sparkles | `sparkles` | Multi-shot mode, AI enhancement |
| Trash | `trash-2` | Delete |
| Undo | `undo-2` | Revert changes |
| Flip horizontal | `flip-horizontal-2` | Flip photo |
| Flip vertical | `flip-vertical-2` | Flip photo |
| Pause | `pause` | Pause processing |
| Play | `play` | Resume processing |
| Folder | `folder-down` | Batch download |
