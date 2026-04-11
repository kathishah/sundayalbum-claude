'use client'

import { useEffect, useState, useCallback } from 'react'
import { useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { isAuthenticated } from '@/lib/auth'
import { getJob, reprocessJob } from '@/lib/api'
import type { Job } from '@/lib/types'

// ── Step definitions ──────────────────────────────────────────────────────────

interface ConfigOption {
  key: string
  label: string
  description: string
  type: 'boolean'
  defaultValue: boolean
}

interface PipelineStepDef {
  id: string
  label: string
  phase: 'page' | 'photo'
  icon: React.ReactNode
  description: string
  detail: string
  output: string
  configOptions: ConfigOption[]
}

function StepIcon({ d }: { d: string }) {
  return (
    <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d={d} />
    </svg>
  )
}

const STEPS: PipelineStepDef[] = [
  {
    id: 'load',
    label: 'Load',
    phase: 'page',
    icon: <StepIcon d="M4 16l4-4 4 4 4-8 4 8M3 20h18" />,
    description: 'Decode the uploaded image and apply EXIF orientation.',
    detail:
      'Supports HEIC (iPhone), JPEG, PNG, and DNG (ProRAW). Converts to float32 RGB [0,1] for consistent downstream processing. EXIF rotation metadata is applied so the image is always right-side up entering the pipeline.',
    output: 'Loaded image at native resolution, EXIF-corrected.',
    configOptions: [],
  },
  {
    id: 'normalize',
    label: 'Normalize',
    phase: 'page',
    icon: <StepIcon d="M4 12h16M12 4v16" />,
    description: 'Resize to working resolution and generate a thumbnail.',
    detail:
      'Downscales to a 2048px maximum on the longest side (balancing quality and processing speed). Generates a 400px thumbnail for the library card view. The working resolution is stored for all subsequent page-level steps.',
    output: 'Normalized image at 2048px max, 400px thumbnail.',
    configOptions: [],
  },
  {
    id: 'page_detect',
    label: 'Page Detect',
    phase: 'page',
    icon: <StepIcon d="M3 7l4-4h10l4 4v10l-4 4H7l-4-4V7z" />,
    description: 'Find the album page boundary using GrabCut segmentation.',
    detail:
      'Uses OpenCV GrabCut with an inset seed rectangle to segment the album page from the background. Finds the largest quadrilateral contour. Falls back to the full image if no clear page boundary is found. The output quad defines the four corners of the album page.',
    output: 'Four-point quad describing the album page boundary.',
    configOptions: [],
  },
  {
    id: 'perspective',
    label: 'Perspective',
    phase: 'page',
    icon: <StepIcon d="M5 19L3 3l7 4 4-4 4 4 4-4-2 16H5z" />,
    description: 'Warp the page to a fronto-parallel (flat-on) view.',
    detail:
      'Computes a homography from the four detected page corners to a rectangle of equivalent area. The result is a perspective-corrected, top-down view of the album page — as if you had photographed it flat on a table.',
    output: 'Perspective-corrected album page image.',
    configOptions: [],
  },
  {
    id: 'photo_detect',
    label: 'Photo Detect',
    phase: 'page',
    icon: <StepIcon d="M3 5h5v5H3zM3 14h5v5H3zM11 5h10M11 9h10M11 14h10M11 18h10" />,
    description: 'Find individual photo boundaries within the page.',
    detail:
      'Converts to LAB color space and applies edge detection + contour finding to locate the individual prints on the album page. Filters by aspect ratio and minimum area. Sorts detected photos top-to-bottom, left-to-right.',
    output: 'List of bounding quads, one per detected photo.',
    configOptions: [],
  },
  {
    id: 'photo_split',
    label: 'Photo Split',
    phase: 'page',
    icon: <StepIcon d="M12 3v18M3 12h18" />,
    description: 'Extract each photo as its own image crop.',
    detail:
      'Applies a homography per detected photo quad to extract a clean, rectangular crop. Each photo is saved as an independent image and enters the per-photo stages of the pipeline. Order matches the photo_detect sort order.',
    output: 'N individual photo crop images.',
    configOptions: [],
  },
  {
    id: 'ai_orient',
    label: 'AI Orient',
    phase: 'photo',
    icon: <StepIcon d="M4 12a8 8 0 1 1 16 0 8 8 0 0 1-16 0M12 8v4l3 3" />,
    description: 'Detect and correct photo rotation using Claude AI.',
    detail:
      'Sends the photo to Claude Haiku with a prompt asking for rotation (0/90/180/270°), horizontal flip, and a one-sentence scene description. The scene description is reused in the glare removal step as context for the inpainting model. Runs before glare removal so the model inpaints a correctly-oriented image.',
    output: 'Rotated photo + scene description string.',
    configOptions: [
      {
        key: 'skip_ai_orient',
        label: 'Skip AI orientation',
        description: 'Pass the photo through without rotation correction. Use if Claude is unavailable or the photo is already correctly oriented.',
        type: 'boolean',
        defaultValue: false,
      },
    ],
  },
  {
    id: 'glare_remove',
    label: 'Glare Remove',
    phase: 'photo',
    icon: <StepIcon d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />,
    description: 'Remove glare using AI diffusion inpainting or OpenCV fallback.',
    detail:
      'Default path: detects glare regions (bright saturated highlights), creates a mask, and calls OpenAI gpt-image-1 images.edit with the photo + mask + scene description. The model reconstructs the masked area using the surrounding content as context — it understands what the photo should look like. Fallback path (OpenCV): uses classical inpainting (Telea algorithm). Quality is significantly worse, especially for sleeve glare.',
    output: 'Photo with glare regions reconstructed.',
    configOptions: [
      {
        key: 'use_openai',
        label: 'Use AI inpainting (OpenAI)',
        description:
          'Enabled: use OpenAI diffusion inpainting for high-quality glare removal. Disabled: fall back to OpenCV classical inpainting (faster but lower quality, especially for large glare patches).',
        type: 'boolean',
        defaultValue: true,
      },
    ],
  },
  {
    id: 'geometry',
    label: 'Geometry',
    phase: 'photo',
    icon: <StepIcon d="M3 12h18M12 3v18M5 5l14 14M19 5L5 19" />,
    description: 'Small-angle rotation and dewarp correction (currently pass-through).',
    detail:
      'Intended to correct small residual rotation (from tilted camera) and lens distortion. Both sub-steps are currently disabled — the Hough-line rotation detector fires on image content rather than the photo frame, producing false corrections. The dewarp detector similarly fires on curved scene content. A future version will detect the white border of the physical print to determine the correct angle.',
    output: 'Photo (unchanged in current build).',
    configOptions: [],
  },
  {
    id: 'color_restore',
    label: 'Color Restore',
    phase: 'photo',
    icon: <StepIcon d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20M8 12c0-2.2 1.8-4 4-4s4 1.8 4 4-1.8 4-4 4-4-1.8-4-4" />,
    description: 'Restore natural colors: white balance, deyellow, brightness, sharpen.',
    detail:
      'Four-stage pipeline: (1) White-point stretch — finds the brightest neutral pixels and rescales channels so they land at white. (2) Deyellow — reduces the blue-channel deficit that causes warm/amber casts in aged prints. (3) Adaptive brightness lift — raises shadows and midtones proportionally to the image\'s median luminance, avoids over-brightening well-exposed photos. (4) Vibrance — adds saturation selectively to undersaturated pixels (preserves already-vivid areas). Finalized with unsharp-mask sharpening.',
    output: 'Color-restored, sharpened photo.',
    configOptions: [],
  },
]

// ── Helpers ───────────────────────────────────────────────────────────────────

const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] as [number, number, number, number] } },
}

const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.07 } } }

const STEP_LABELS_MAP: Record<string, string> = {
  load: 'load',
  normalize: 'normalize',
  page_detect: 'page_detect',
  perspective: 'perspective',
  photo_detect: 'photo_detect',
  photo_split: 'photo_split',
  ai_orient: 'ai_orient',
  glare_remove: 'glare_remove',
  geometry: 'geometry',
  color_restore: 'color_restore',
}

// ── Step card ─────────────────────────────────────────────────────────────────

interface StepCardProps {
  step: PipelineStepDef
  index: number
  job: Job | null
  onRerun: (stepId: string, config: Record<string, unknown>) => void
  rerunning: string | null
}

function StepCard({ step, index, job, onRerun, rerunning }: StepCardProps) {
  const [expanded, setExpanded] = useState(false)
  const [config, setConfig] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(step.configOptions.map((o) => [o.key, o.defaultValue])),
  )

  const canRerun = !!job && job.status !== 'processing'

  return (
    <motion.div
      variants={fadeUp}
      className="rounded-2xl border border-sa-stone-200 dark:border-sa-stone-700 bg-white dark:bg-sa-stone-900 overflow-hidden"
    >
      <button
        className="w-full text-left p-5 flex items-start gap-4 hover:bg-sa-stone-50 dark:hover:bg-sa-stone-800 transition-colors duration-[200ms]"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
      >
        {/* Step number */}
        <span className="flex-shrink-0 w-7 h-7 rounded-full bg-sa-amber-100 dark:bg-sa-amber-900/40 text-sa-amber-600 dark:text-sa-amber-400 text-xs font-bold flex items-center justify-center mt-0.5">
          {index + 1}
        </span>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sa-amber-600 dark:text-sa-amber-400">{step.icon}</span>
            <h3 className="font-semibold text-sa-stone-900 dark:text-sa-stone-50">{step.label}</h3>
            <span className={`ml-auto text-xs px-2 py-0.5 rounded-full ${
              step.phase === 'page'
                ? 'bg-sa-stone-100 dark:bg-sa-stone-700 text-sa-stone-500 dark:text-sa-stone-400'
                : 'bg-sa-amber-100 dark:bg-sa-amber-900/40 text-sa-amber-600 dark:text-sa-amber-400'
            }`}>
              {step.phase === 'page' ? 'page-level' : 'per-photo'}
            </span>
          </div>
          <p className="text-sm text-sa-stone-500 dark:text-sa-stone-400">{step.description}</p>
        </div>

        <svg
          viewBox="0 0 16 16"
          className={`w-4 h-4 text-sa-stone-400 flex-shrink-0 mt-1 transition-transform duration-[200ms] ${expanded ? 'rotate-180' : ''}`}
          fill="none" stroke="currentColor" strokeWidth="1.5"
        >
          <path d="M3 6l5 5 5-5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {expanded && (
        <div className="px-5 pb-5 pt-0 border-t border-sa-stone-100 dark:border-sa-stone-800">
          <div className="grid md:grid-cols-2 gap-5 mt-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-sa-stone-400 dark:text-sa-stone-500 mb-2">How it works</p>
              <p className="text-sm text-sa-stone-600 dark:text-sa-stone-300 leading-relaxed">{step.detail}</p>
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-sa-stone-400 dark:text-sa-stone-500 mb-2">Output</p>
              <p className="text-sm text-sa-stone-600 dark:text-sa-stone-300 leading-relaxed">{step.output}</p>
            </div>
          </div>

          {/* Config options + re-run */}
          {(step.configOptions.length > 0 || canRerun) && (
            <div className="mt-5 pt-4 border-t border-sa-stone-100 dark:border-sa-stone-800">
              {step.configOptions.length > 0 && (
                <div className="mb-4">
                  <p className="text-xs font-semibold uppercase tracking-wider text-sa-stone-400 dark:text-sa-stone-500 mb-3">
                    Configuration
                  </p>
                  <div className="flex flex-col gap-3">
                    {step.configOptions.map((opt) => (
                      <label key={opt.key} className="flex items-start gap-3 cursor-pointer group">
                        <div className="relative mt-0.5 flex-shrink-0">
                          <input
                            type="checkbox"
                            className="sr-only peer"
                            checked={config[opt.key]}
                            onChange={(e) => setConfig((c) => ({ ...c, [opt.key]: e.target.checked }))}
                            disabled={!canRerun}
                          />
                          <div className="w-9 h-5 rounded-full bg-sa-stone-200 dark:bg-sa-stone-700 peer-checked:bg-sa-amber-500 transition-colors duration-[200ms] peer-disabled:opacity-50" />
                          <div className="absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-[200ms] peer-checked:translate-x-4" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-sa-stone-800 dark:text-sa-stone-200 group-hover:text-sa-stone-900 dark:group-hover:text-sa-stone-50 transition-colors duration-[200ms]">
                            {opt.label}
                          </p>
                          <p className="text-xs text-sa-stone-500 dark:text-sa-stone-400 leading-relaxed mt-0.5">
                            {opt.description}
                          </p>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>
              )}

              {canRerun && (
                <button
                  onClick={() => onRerun(STEP_LABELS_MAP[step.id] ?? step.id, config)}
                  disabled={rerunning === step.id}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium bg-sa-amber-500 hover:bg-sa-amber-600 disabled:opacity-60 text-white transition-colors duration-[200ms]"
                >
                  {rerunning === step.id ? (
                    <>
                      <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="9" strokeOpacity="0.25" />
                        <path d="M12 3a9 9 0 0 1 9 9" />
                      </svg>
                      Re-running…
                    </>
                  ) : (
                    <>
                      <svg viewBox="0 0 16 16" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M3 8a5 5 0 1 1 10 0" strokeLinecap="round" />
                        <path d="M13 5v3h-3" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      Re-run from this step
                    </>
                  )}
                </button>
              )}

              {!canRerun && !job && (
                <p className="text-xs text-sa-stone-400 dark:text-sa-stone-500 italic">
                  Open a job to enable re-run controls.{' '}
                  <Link href="/library" className="text-sa-amber-600 dark:text-sa-amber-400 hover:underline">
                    Go to Library →
                  </Link>
                </p>
              )}

              {!canRerun && job && job.status === 'processing' && (
                <p className="text-xs text-sa-stone-400 dark:text-sa-stone-500 italic">
                  Job is currently processing — wait for it to complete before re-running.
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </motion.div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function PipelinePage() {
  const searchParams = useSearchParams()
  const jobId = searchParams.get('jobId')

  const [authed, setAuthed] = useState(false)
  const [job, setJob] = useState<Job | null>(null)
  const [jobError, setJobError] = useState<string | null>(null)
  const [rerunning, setRerunning] = useState<string | null>(null)
  const [rerunMsg, setRerunMsg] = useState<string | null>(null)

  useEffect(() => {
    setAuthed(isAuthenticated())
  }, [])

  useEffect(() => {
    if (!jobId || !isAuthenticated()) return
    getJob(jobId)
      .then(setJob)
      .catch((e: Error) => setJobError(e.message))
  }, [jobId])

  const handleRerun = useCallback(
    async (fromStep: string, config: Record<string, unknown>) => {
      if (!job) return
      setRerunning(fromStep)
      setRerunMsg(null)
      try {
        await reprocessJob(job.job_id, { from_step: fromStep, config })
        setRerunMsg(`Re-running from "${fromStep}" — check your Library for progress.`)
        setJob((j) => (j ? { ...j, status: 'processing' } : j))
      } catch (e: unknown) {
        setRerunMsg(`Error: ${e instanceof Error ? e.message : String(e)}`)
      } finally {
        setRerunning(null)
      }
    },
    [job],
  )

  return (
    <div className="py-16 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <motion.div initial="hidden" animate="show" variants={stagger} className="mb-12">
          <motion.p variants={fadeUp} className="text-sm font-semibold uppercase tracking-wider text-sa-amber-600 dark:text-sa-amber-400 mb-3">
            Pipeline
          </motion.p>
          <motion.h1 variants={fadeUp} className="font-display text-4xl md:text-5xl font-bold text-sa-stone-900 dark:text-sa-stone-50 mb-4">
            How Sunday Album processes your photos
          </motion.h1>
          <motion.p variants={fadeUp} className="text-lg text-sa-stone-500 dark:text-sa-stone-400 leading-relaxed">
            Ten sequential steps transform a phone photo of an album page into clean,
            individually restored digital photos. Expand any step to learn what it does —
            and re-run individual steps from your job if you want to tweak the result.
          </motion.p>
        </motion.div>

        {/* Job context bar */}
        {jobId && (
          <div className={`mb-8 p-4 rounded-xl border text-sm ${
            job
              ? 'bg-sa-amber-50 dark:bg-sa-amber-950/30 border-sa-amber-200 dark:border-sa-amber-800 text-sa-amber-800 dark:text-sa-amber-200'
              : jobError
                ? 'bg-red-50 dark:bg-red-950/30 border-red-200 dark:border-red-800 text-red-700 dark:text-red-300'
                : 'bg-sa-stone-50 dark:bg-sa-stone-800 border-sa-stone-200 dark:border-sa-stone-700 text-sa-stone-500'
          }`}>
            {job ? (
              <span>
                Job <code className="font-mono text-xs bg-black/10 px-1 py-0.5 rounded">{job.job_id.slice(0, 8)}…</code>{' '}
                loaded — <strong>{job.input_filename}</strong> · {job.photo_count} photo{job.photo_count !== 1 ? 's' : ''} · status: <strong>{job.status}</strong>.{' '}
                Expand any step below to re-run from that point.
              </span>
            ) : jobError ? (
              <span>Could not load job: {jobError}</span>
            ) : (
              <span>Loading job…</span>
            )}
          </div>
        )}

        {rerunMsg && (
          <div className="mb-6 p-4 rounded-xl bg-sa-stone-50 dark:bg-sa-stone-800 border border-sa-stone-200 dark:border-sa-stone-700 text-sm text-sa-stone-700 dark:text-sa-stone-200">
            {rerunMsg}{' '}
            {rerunMsg.startsWith('Re-running') && (
              <Link href="/library" className="text-sa-amber-600 dark:text-sa-amber-400 hover:underline">
                View in Library →
              </Link>
            )}
          </div>
        )}

        {/* Not authenticated / no jobId prompt */}
        {!authed && (
          <div className="mb-8 p-4 rounded-xl bg-sa-stone-50 dark:bg-sa-stone-800 border border-sa-stone-200 dark:border-sa-stone-700 text-sm text-sa-stone-500 dark:text-sa-stone-400">
            <strong className="text-sa-stone-700 dark:text-sa-stone-200">Want to re-run steps on your own images?</strong>{' '}
            <Link href="/login" className="text-sa-amber-600 dark:text-sa-amber-400 hover:underline">Sign in</Link>{' '}
            then open a completed job and click "View in Pipeline" to load it here.
          </div>
        )}

        {authed && !jobId && (
          <div className="mb-8 p-4 rounded-xl bg-sa-stone-50 dark:bg-sa-stone-800 border border-sa-stone-200 dark:border-sa-stone-700 text-sm text-sa-stone-500 dark:text-sa-stone-400">
            <strong className="text-sa-stone-700 dark:text-sa-stone-200">Re-run mode:</strong>{' '}
            open a completed job from your{' '}
            <Link href="/library" className="text-sa-amber-600 dark:text-sa-amber-400 hover:underline">Library</Link>{' '}
            and click "View in Pipeline" to load it here with re-run controls.
          </div>
        )}

        {/* Phase labels */}
        <div className="flex gap-4 mb-6 text-xs">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-sa-stone-300 dark:bg-sa-stone-600 inline-block" />
            <span className="text-sa-stone-500 dark:text-sa-stone-400">Page-level (runs once per image)</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-sa-amber-400 inline-block" />
            <span className="text-sa-stone-500 dark:text-sa-stone-400">Per-photo (runs once per detected photo)</span>
          </span>
        </div>

        {/* Steps */}
        <motion.div initial="hidden" animate="show" variants={stagger} className="flex flex-col gap-3">
          {STEPS.map((step, i) => (
            <StepCard
              key={step.id}
              step={step}
              index={i}
              job={job}
              onRerun={handleRerun}
              rerunning={rerunning}
            />
          ))}
        </motion.div>

        {/* Footer CTA */}
        <div className="mt-12 p-6 rounded-2xl bg-sa-amber-50 dark:bg-sa-amber-950/30 border border-sa-amber-200 dark:border-sa-amber-800 text-center">
          <h2 className="font-display text-xl font-bold text-sa-stone-900 dark:text-sa-stone-50 mb-2">
            Try the full pipeline on your photos
          </h2>
          <p className="text-sm text-sa-stone-500 dark:text-sa-stone-400 mb-4">
            Upload a photo of an album page and watch every step run in real time.
          </p>
          <div className="flex justify-center gap-3">
            <Link href="/login" className="px-5 py-2.5 rounded-xl text-sm font-medium bg-sa-amber-500 hover:bg-sa-amber-600 text-white transition-colors duration-[200ms]">
              Try it free
            </Link>
            <Link href="/download" className="px-5 py-2.5 rounded-xl text-sm font-medium border border-sa-stone-200 dark:border-sa-stone-700 text-sa-stone-700 dark:text-sa-stone-200 hover:bg-white dark:hover:bg-sa-stone-800 transition-colors duration-[200ms]">
              Download for Mac
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
