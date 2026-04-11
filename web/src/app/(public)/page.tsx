'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { isAuthenticated } from '@/lib/auth'
import BeforeAfterSlider from '@/components/BeforeAfterSlider'

// ─── Animation helpers ────────────────────────────────────────────────────────

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.16, 1, 0.3, 1] as [number, number, number, number] } },
}

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.12 } },
}

// ─── Feature grid ─────────────────────────────────────────────────────────────

const features = [
  {
    icon: (
      <svg viewBox="0 0 24 24" className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="3" y="5" width="7" height="9" rx="1" />
        <rect x="13" y="5" width="8" height="5" rx="1" />
        <rect x="13" y="13" width="8" height="5" rx="1" />
        <rect x="3" y="17" width="7" height="3" rx="1" />
      </svg>
    ),
    title: 'Multi-photo detection',
    description: 'Finds every photo on the album page — 1, 2, 3, or more, automatically.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="12" r="9" />
        <path d="M9 9c1.5-2 5.5-2 6 1.5-1 2-4 2-5 4 0 1 1 2 2 2" strokeLinecap="round" />
        <circle cx="12" cy="19" r="0.5" fill="currentColor" />
      </svg>
    ),
    title: 'AI glare removal',
    description: 'Diffusion-based inpainting removes sleeve and print glare intelligently.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="12" r="9" />
        <path d="M8 12a4 4 0 0 1 8 0" />
        <path d="M7 15l3-3 2 2 3-4 2 2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    title: 'Color restoration',
    description: 'Reverses yellowing and fading, restoring natural tones to old prints.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M12 3v3M12 18v3M3 12h3M18 12h3" strokeLinecap="round" />
        <circle cx="12" cy="12" r="5" />
        <path d="M7 17l2-2M17 7l-2 2M7 7l2 2M17 17l-2-2" strokeLinecap="round" />
      </svg>
    ),
    title: 'Auto-orientation',
    description: 'Detects and corrects rotation using AI scene understanding per photo.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M4 8L8 4h12v12l-4 4H4V8z" />
        <path d="M4 8h12v12" />
        <path d="M16 4l4 4" />
      </svg>
    ),
    title: 'Perspective correction',
    description: 'Straightens keystoned shots taken at an angle for a flat, clean crop.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="3" y="3" width="8" height="18" rx="2" />
        <rect x="13" y="3" width="8" height="11" rx="2" />
        <path d="M13 18h8" strokeLinecap="round" />
        <path d="M17 15v6" strokeLinecap="round" />
      </svg>
    ),
    title: 'Web + Mac',
    description: 'Process in the browser or download the free native Mac app.',
  },
]

// ─── Pipeline steps ────────────────────────────────────────────────────────────

const pipelineSteps = [
  { id: 'snap', label: 'Snap', detail: 'Take a photo of your album page with any phone camera.' },
  { id: 'page', label: 'Page Detect', detail: 'Finds and perspective-corrects the album page boundary.' },
  { id: 'split', label: 'Split Photos', detail: 'Detects each individual photo on the page and extracts it.' },
  { id: 'glare', label: 'Remove Glare', detail: 'AI inpainting erases sleeve and print glare completely.' },
  { id: 'color', label: 'Restore Color', detail: 'Reverses yellowing, lifts shadows, and sharpens detail.' },
  { id: 'done', label: 'Done', detail: 'Download your clean, individually restored digital photos.' },
]

// ─── Demo image slots (CSS placeholders until real pipeline images are added) ─

function DemoBeforeSlot() {
  return (
    <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-amber-100 via-yellow-200 to-amber-300 dark:from-amber-900/60 dark:via-yellow-800/50 dark:to-amber-700/40">
      <div className="relative w-4/5 h-4/5 rounded border-2 border-amber-300/60 overflow-hidden">
        {/* Simulated glare patch */}
        <div className="absolute inset-0 bg-gradient-to-br from-white/80 via-white/20 to-transparent" />
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 opacity-40">
          <div className="w-3/4 h-2 rounded bg-amber-600" />
          <div className="w-1/2 h-2 rounded bg-amber-600" />
          <div className="w-2/3 h-2 rounded bg-amber-500" />
        </div>
        <p className="absolute bottom-2 left-0 right-0 text-center text-xs text-amber-700 font-medium opacity-70">
          Yellowed · Glare
        </p>
      </div>
    </div>
  )
}

function DemoAfterSlot() {
  return (
    <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-stone-50 via-orange-50 to-amber-50 dark:from-stone-800 dark:via-stone-700 dark:to-stone-800">
      <div className="relative w-4/5 h-4/5 rounded border-2 border-stone-200/60 overflow-hidden shadow-md">
        <div className="absolute inset-0 bg-gradient-to-br from-orange-100/30 via-transparent to-amber-50/20" />
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 opacity-40">
          <div className="w-3/4 h-2 rounded bg-stone-500" />
          <div className="w-1/2 h-2 rounded bg-stone-400" />
          <div className="w-2/3 h-2 rounded bg-stone-500" />
        </div>
        <p className="absolute bottom-2 left-0 right-0 text-center text-xs text-stone-600 font-medium opacity-70">
          Restored · Crisp
        </p>
      </div>
    </div>
  )
}

function DemoMultiBeforeSlot() {
  return (
    <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-amber-100 via-yellow-200 to-amber-200 dark:from-amber-900/60 dark:via-yellow-800/50 dark:to-amber-700/40 p-6">
      <div className="relative w-full h-full rounded border-2 border-amber-300/60 bg-amber-50/50 dark:bg-amber-900/30 overflow-hidden">
        {/* Glare overlay */}
        <div className="absolute inset-0 bg-gradient-to-tr from-white/50 via-white/10 to-transparent pointer-events-none" />
        {/* Three photo placeholders arranged like an album page */}
        <div className="absolute inset-3 grid grid-cols-2 gap-2">
          <div className="rounded bg-amber-300/40 border border-amber-400/30" />
          <div className="rounded bg-amber-300/40 border border-amber-400/30" />
          <div className="col-span-2 rounded bg-amber-300/40 border border-amber-400/30" />
        </div>
        <p className="absolute bottom-2 left-0 right-0 text-center text-xs text-amber-700 font-medium opacity-70">
          Album page · 3 photos
        </p>
      </div>
    </div>
  )
}

function DemoMultiAfterSlot() {
  return (
    <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-stone-50 via-orange-50 to-amber-50 dark:from-stone-800 dark:via-stone-700 dark:to-stone-800 p-4">
      <div className="w-full h-full grid grid-cols-2 gap-3">
        <div className="rounded-lg bg-gradient-to-br from-orange-100 to-amber-100 dark:from-stone-600 dark:to-stone-700 border border-stone-200/50 shadow-sm" />
        <div className="rounded-lg bg-gradient-to-br from-blue-50 to-stone-100 dark:from-stone-600 dark:to-stone-700 border border-stone-200/50 shadow-sm" />
        <div className="col-span-2 rounded-lg bg-gradient-to-br from-green-50 to-stone-100 dark:from-stone-600 dark:to-stone-700 border border-stone-200/50 shadow-sm flex items-center justify-center">
          <p className="text-xs text-stone-400 font-medium">3 individual photos</p>
        </div>
      </div>
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function HomePage() {
  const [authed, setAuthed] = useState(false)
  const [activeStep, setActiveStep] = useState<string | null>(null)

  useEffect(() => {
    setAuthed(isAuthenticated())
  }, [])

  const primaryCta = authed
    ? { href: '/library', label: 'Go to Library' }
    : { href: '/login', label: 'Try it free' }

  return (
    <div className="overflow-x-hidden">
      {/* ── 1. Hero ─────────────────────────────────────────────────────────── */}
      <section className="relative py-20 md:py-32 px-4 overflow-hidden">
        {/* Warm background gradient */}
        <div className="absolute inset-0 bg-gradient-to-b from-sa-amber-50 to-white dark:from-sa-stone-900 dark:to-sa-stone-950 pointer-events-none" />
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-sa-amber-100/60 dark:bg-sa-amber-900/20 rounded-full blur-3xl pointer-events-none" />

        <div className="relative max-w-3xl mx-auto text-center">
          <motion.div
            initial="hidden"
            animate="show"
            variants={stagger}
            className="flex flex-col items-center gap-6"
          >
            <motion.div variants={fadeUp}>
              <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-sa-amber-100 dark:bg-sa-amber-900/40 text-sa-amber-700 dark:text-sa-amber-300 text-xs font-medium border border-sa-amber-200 dark:border-sa-amber-800">
                <span className="w-1.5 h-1.5 rounded-full bg-sa-amber-500 animate-pulse" />
                Free for Mac · Free tier for web
              </span>
            </motion.div>

            <motion.h1
              variants={fadeUp}
              className="font-display text-4xl md:text-6xl font-bold text-sa-stone-900 dark:text-sa-stone-50 leading-[1.1] tracking-tight"
            >
              Your print photos,{' '}
              <span className="text-sa-amber-600 dark:text-sa-amber-400">beautifully</span>{' '}
              digitized.
            </motion.h1>

            <motion.p
              variants={fadeUp}
              className="text-lg md:text-xl text-sa-stone-600 dark:text-sa-stone-300 max-w-2xl leading-relaxed"
            >
              Snap a photo of a single print or an entire album page. Sunday Album finds
              every photo, removes glare, restores faded colors, and delivers clean digital
              photos — automatically.
            </motion.p>

            <motion.div variants={fadeUp} className="flex flex-wrap justify-center gap-3 pt-2">
              <Link
                href={primaryCta.href}
                className="px-6 py-3 rounded-xl text-base font-medium bg-sa-amber-500 hover:bg-sa-amber-600 text-white transition-colors duration-[200ms] shadow-sm"
              >
                {primaryCta.label}
              </Link>
              <Link
                href="/download"
                className="px-6 py-3 rounded-xl text-base font-medium border border-sa-stone-200 dark:border-sa-stone-700 text-sa-stone-700 dark:text-sa-stone-200 hover:bg-sa-stone-50 dark:hover:bg-sa-stone-800 transition-colors duration-[200ms]"
              >
                Download for Mac
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ── 2. How It Works ─────────────────────────────────────────────────── */}
      <section className="py-20 px-4 bg-white dark:bg-sa-stone-950">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: '-80px' }}
            variants={stagger}
            className="text-center mb-14"
          >
            <motion.p variants={fadeUp} className="text-sm font-semibold uppercase tracking-wider text-sa-amber-600 dark:text-sa-amber-400 mb-3">
              How it works
            </motion.p>
            <motion.h2 variants={fadeUp} className="font-display text-3xl md:text-4xl font-bold text-sa-stone-900 dark:text-sa-stone-50">
              Three steps to digital memories
            </motion.h2>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: '-60px' }}
            variants={stagger}
            className="grid md:grid-cols-3 gap-8"
          >
            {[
              {
                step: '01',
                title: 'Snap',
                desc: 'Take a photo of your album page with any phone. One photo captures 1, 2, 3 or more prints at once.',
                icon: (
                  <svg viewBox="0 0 40 40" className="w-10 h-10" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <rect x="6" y="11" width="28" height="22" rx="3" />
                    <circle cx="20" cy="22" r="6" />
                    <circle cx="20" cy="22" r="3" fill="currentColor" stroke="none" />
                    <path d="M15 11V9h10v2" strokeLinecap="round" />
                  </svg>
                ),
              },
              {
                step: '02',
                title: 'Process',
                desc: 'Sunday Album detects every photo on the page, removes glare, restores colors, and corrects orientation.',
                icon: (
                  <svg viewBox="0 0 40 40" className="w-10 h-10" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <circle cx="20" cy="20" r="13" />
                    <path d="M20 10v4M20 26v4M10 20h4M26 20h4" strokeLinecap="round" />
                    <circle cx="20" cy="20" r="4" />
                  </svg>
                ),
              },
              {
                step: '03',
                title: 'Download',
                desc: 'Get clean, individually restored digital photos ready to share, print, or archive forever.',
                icon: (
                  <svg viewBox="0 0 40 40" className="w-10 h-10" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M20 8v16M13 18l7 7 7-7" strokeLinecap="round" strokeLinejoin="round" />
                    <path d="M8 30h24" strokeLinecap="round" />
                  </svg>
                ),
              },
            ].map((item) => (
              <motion.div
                key={item.step}
                variants={fadeUp}
                className="flex flex-col items-center text-center p-6 rounded-2xl bg-sa-stone-50 dark:bg-sa-stone-900 border border-sa-stone-100 dark:border-sa-stone-800"
              >
                <div className="mb-4 p-3 rounded-xl bg-sa-amber-50 dark:bg-sa-amber-900/30 text-sa-amber-600 dark:text-sa-amber-400">
                  {item.icon}
                </div>
                <span className="text-xs font-bold text-sa-amber-500 tracking-widest mb-2">{item.step}</span>
                <h3 className="font-display text-xl font-bold text-sa-stone-900 dark:text-sa-stone-50 mb-2">{item.title}</h3>
                <p className="text-sm text-sa-stone-500 dark:text-sa-stone-400 leading-relaxed">{item.desc}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ── 3. Before / After ───────────────────────────────────────────────── */}
      <section className="py-20 px-4 bg-sa-stone-50 dark:bg-sa-stone-900">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: '-80px' }}
            variants={stagger}
            className="text-center mb-14"
          >
            <motion.p variants={fadeUp} className="text-sm font-semibold uppercase tracking-wider text-sa-amber-600 dark:text-sa-amber-400 mb-3">
              See the difference
            </motion.p>
            <motion.h2 variants={fadeUp} className="font-display text-3xl md:text-4xl font-bold text-sa-stone-900 dark:text-sa-stone-50">
              Drag to compare
            </motion.h2>
            <motion.p variants={fadeUp} className="mt-3 text-sa-stone-500 dark:text-sa-stone-400 max-w-xl mx-auto">
              Move the slider to see the transformation. Every photo is processed individually.
            </motion.p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: '-60px' }}
            variants={stagger}
            className="grid md:grid-cols-2 gap-8"
          >
            {/* Pair A — Single print */}
            <motion.div variants={fadeUp} className="flex flex-col gap-3">
              <BeforeAfterSlider
                beforeLabel="Original scan"
                afterLabel="Restored"
                beforeSlot={<DemoBeforeSlot />}
                afterSlot={<DemoAfterSlot />}
                className="h-64 md:h-72"
              />
              <p className="text-xs text-center text-sa-stone-400 dark:text-sa-stone-500">
                Single print — glare removed, colors restored
              </p>
            </motion.div>

            {/* Pair B — Album page with 3 photos */}
            <motion.div variants={fadeUp} className="flex flex-col gap-3">
              <BeforeAfterSlider
                beforeLabel="Album page"
                afterLabel="3 photos extracted"
                beforeSlot={<DemoMultiBeforeSlot />}
                afterSlot={<DemoMultiAfterSlot />}
                className="h-64 md:h-72"
              />
              <p className="text-xs text-center text-sa-stone-400 dark:text-sa-stone-500">
                Album page → 3 individual restored photos
              </p>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ── 4. Feature Grid ─────────────────────────────────────────────────── */}
      <section className="py-20 px-4 bg-white dark:bg-sa-stone-950">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: '-80px' }}
            variants={stagger}
            className="text-center mb-14"
          >
            <motion.p variants={fadeUp} className="text-sm font-semibold uppercase tracking-wider text-sa-amber-600 dark:text-sa-amber-400 mb-3">
              Features
            </motion.p>
            <motion.h2 variants={fadeUp} className="font-display text-3xl md:text-4xl font-bold text-sa-stone-900 dark:text-sa-stone-50">
              Everything your memories deserve
            </motion.h2>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: '-60px' }}
            variants={stagger}
            className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5"
          >
            {features.map((f) => (
              <motion.div
                key={f.title}
                variants={fadeUp}
                className="p-5 rounded-2xl bg-sa-stone-50 dark:bg-sa-stone-900 border border-sa-stone-100 dark:border-sa-stone-800 hover:border-sa-amber-200 dark:hover:border-sa-amber-800 transition-colors duration-[200ms]"
              >
                <div className="mb-3 text-sa-amber-600 dark:text-sa-amber-400">{f.icon}</div>
                <h3 className="font-semibold text-sa-stone-900 dark:text-sa-stone-50 mb-1">{f.title}</h3>
                <p className="text-sm text-sa-stone-500 dark:text-sa-stone-400 leading-relaxed">{f.description}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ── 5. Pipeline Visualization ───────────────────────────────────────── */}
      <section className="py-20 px-4 bg-sa-stone-50 dark:bg-sa-stone-900">
        <div className="max-w-5xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: '-80px' }}
            variants={stagger}
            className="text-center mb-14"
          >
            <motion.p variants={fadeUp} className="text-sm font-semibold uppercase tracking-wider text-sa-amber-600 dark:text-sa-amber-400 mb-3">
              Under the hood
            </motion.p>
            <motion.h2 variants={fadeUp} className="font-display text-3xl md:text-4xl font-bold text-sa-stone-900 dark:text-sa-stone-50">
              A full processing pipeline
            </motion.h2>
            <motion.p variants={fadeUp} className="mt-3 text-sa-stone-500 dark:text-sa-stone-400">
              Hover any step to learn what it does.
            </motion.p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: '-60px' }}
            variants={stagger}
            className="flex flex-wrap justify-center gap-3 md:gap-0 items-center"
          >
            {pipelineSteps.map((step, i) => (
              <motion.div key={step.id} variants={fadeUp} className="flex items-center">
                <div
                  className="relative group"
                  onMouseEnter={() => setActiveStep(step.id)}
                  onMouseLeave={() => setActiveStep(null)}
                  onFocus={() => setActiveStep(step.id)}
                  onBlur={() => setActiveStep(null)}
                  tabIndex={0}
                >
                  <div
                    className={`
                      px-4 py-2.5 rounded-xl text-sm font-medium cursor-default
                      transition-all duration-[200ms]
                      ${
                        activeStep === step.id
                          ? 'bg-sa-amber-500 text-white shadow-md scale-105'
                          : 'bg-white dark:bg-sa-stone-800 text-sa-stone-700 dark:text-sa-stone-200 border border-sa-stone-200 dark:border-sa-stone-700 hover:border-sa-amber-300 dark:hover:border-sa-amber-700'
                      }
                    `}
                  >
                    {step.label}
                  </div>
                  {/* Tooltip */}
                  <div
                    className={`
                      absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-52 p-3
                      rounded-xl bg-sa-stone-900 dark:bg-sa-stone-50 text-sa-stone-50 dark:text-sa-stone-900
                      text-xs leading-relaxed shadow-lg z-10
                      transition-all duration-[200ms]
                      ${activeStep === step.id ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-1 pointer-events-none'}
                    `}
                  >
                    {step.detail}
                    <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-sa-stone-900 dark:border-t-sa-stone-50" />
                  </div>
                </div>

                {/* Arrow between steps */}
                {i < pipelineSteps.length - 1 && (
                  <svg
                    viewBox="0 0 24 24"
                    className="w-5 h-5 text-sa-stone-300 dark:text-sa-stone-600 mx-1 flex-shrink-0 hidden md:block"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M5 12h14M14 7l5 5-5 5" />
                  </svg>
                )}
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* ── 6. Pricing Summary ──────────────────────────────────────────────── */}
      <section className="py-20 px-4 bg-white dark:bg-sa-stone-950">
        <div className="max-w-3xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: '-80px' }}
            variants={stagger}
            className="text-center mb-14"
          >
            <motion.p variants={fadeUp} className="text-sm font-semibold uppercase tracking-wider text-sa-amber-600 dark:text-sa-amber-400 mb-3">
              Pricing
            </motion.p>
            <motion.h2 variants={fadeUp} className="font-display text-3xl md:text-4xl font-bold text-sa-stone-900 dark:text-sa-stone-50">
              Simple, honest pricing
            </motion.h2>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: '-60px' }}
            variants={stagger}
            className="grid sm:grid-cols-2 gap-6"
          >
            {/* Mac App */}
            <motion.div
              variants={fadeUp}
              className="p-6 rounded-2xl border-2 border-sa-stone-200 dark:border-sa-stone-700 bg-white dark:bg-sa-stone-900 flex flex-col"
            >
              <div className="mb-4">
                <span className="text-xs font-bold uppercase tracking-wider text-sa-stone-400 dark:text-sa-stone-500">Mac App</span>
                <div className="mt-2 flex items-end gap-1">
                  <span className="font-display text-4xl font-bold text-sa-stone-900 dark:text-sa-stone-50">Free</span>
                  <span className="text-sa-stone-400 mb-1">forever</span>
                </div>
              </div>
              <ul className="flex flex-col gap-2 text-sm text-sa-stone-600 dark:text-sa-stone-300 mb-6 flex-1">
                {['Full 10-step pipeline', 'Runs locally on your Mac', 'No account, no login', 'No internet required'].map((item) => (
                  <li key={item} className="flex items-center gap-2">
                    <svg viewBox="0 0 16 16" className="w-4 h-4 text-sa-success flex-shrink-0" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M3 8l3.5 3.5L13 4" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    {item}
                  </li>
                ))}
              </ul>
              <Link
                href="/download"
                className="block w-full px-4 py-2.5 rounded-xl text-sm font-medium text-center border border-sa-stone-200 dark:border-sa-stone-700 text-sa-stone-700 dark:text-sa-stone-200 hover:bg-sa-stone-50 dark:hover:bg-sa-stone-800 transition-colors duration-[200ms]"
              >
                Download for Mac
              </Link>
            </motion.div>

            {/* Web App */}
            <motion.div
              variants={fadeUp}
              className="p-6 rounded-2xl border-2 border-sa-amber-400 dark:border-sa-amber-600 bg-sa-amber-50 dark:bg-sa-amber-950/30 flex flex-col relative overflow-hidden"
            >
              <div className="absolute top-4 right-4 px-2 py-0.5 rounded-full bg-sa-amber-500 text-white text-xs font-medium">
                Most popular
              </div>
              <div className="mb-4">
                <span className="text-xs font-bold uppercase tracking-wider text-sa-amber-600 dark:text-sa-amber-400">Web App</span>
                <div className="mt-2 flex items-end gap-1">
                  <span className="font-display text-4xl font-bold text-sa-stone-900 dark:text-sa-stone-50">Free</span>
                  <span className="text-sa-stone-400 mb-1">20 pages/day</span>
                </div>
              </div>
              <ul className="flex flex-col gap-2 text-sm text-sa-stone-600 dark:text-sa-stone-300 mb-6 flex-1">
                {['Email login, no password', 'Live processing progress', 'Bring your own keys to remove limits', 'Download photos instantly'].map((item) => (
                  <li key={item} className="flex items-center gap-2">
                    <svg viewBox="0 0 16 16" className="w-4 h-4 text-sa-success flex-shrink-0" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M3 8l3.5 3.5L13 4" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    {item}
                  </li>
                ))}
              </ul>
              <Link
                href="/login"
                className="block w-full px-4 py-2.5 rounded-xl text-sm font-medium text-center bg-sa-amber-500 hover:bg-sa-amber-600 text-white transition-colors duration-[200ms]"
              >
                Get started free
              </Link>
            </motion.div>
          </motion.div>

          <motion.p
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center text-sm text-sa-stone-400 dark:text-sa-stone-500 mt-5"
          >
            <Link href="/pricing" className="hover:text-sa-amber-600 dark:hover:text-sa-amber-400 transition-colors duration-[200ms]">
              See full pricing details →
            </Link>
          </motion.p>
        </div>
      </section>

      {/* ── 7. Final CTA ────────────────────────────────────────────────────── */}
      <section className="py-24 px-4 bg-gradient-to-b from-sa-amber-50 to-sa-amber-100 dark:from-sa-stone-900 dark:to-sa-stone-800 relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-sa-amber-200/40 via-transparent to-transparent dark:from-sa-amber-900/20 pointer-events-none" />
        <div className="relative max-w-2xl mx-auto text-center">
          <motion.div
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: '-80px' }}
            variants={stagger}
            className="flex flex-col items-center gap-6"
          >
            <motion.h2 variants={fadeUp} className="font-display text-3xl md:text-5xl font-bold text-sa-stone-900 dark:text-sa-stone-50 leading-tight">
              Give your memories a second life.
            </motion.h2>
            <motion.p variants={fadeUp} className="text-lg text-sa-stone-600 dark:text-sa-stone-300 max-w-lg leading-relaxed">
              Your family's best moments deserve better than a faded album. Start digitizing today.
            </motion.p>
            <motion.div variants={fadeUp} className="flex flex-wrap justify-center gap-3">
              <Link
                href={primaryCta.href}
                className="px-7 py-3.5 rounded-xl text-base font-medium bg-sa-amber-500 hover:bg-sa-amber-600 text-white transition-colors duration-[200ms] shadow-sm"
              >
                {primaryCta.label}
              </Link>
              <Link
                href="/download"
                className="px-7 py-3.5 rounded-xl text-base font-medium border border-sa-stone-300 dark:border-sa-stone-600 text-sa-stone-700 dark:text-sa-stone-200 hover:bg-white/60 dark:hover:bg-sa-stone-800 transition-colors duration-[200ms]"
              >
                Download for Mac
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}
