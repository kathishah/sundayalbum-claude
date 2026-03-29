'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { motion, AnimatePresence } from 'framer-motion'
import { getJob } from '@/lib/api'
import type { Job } from '@/lib/types'
import { JOB_STEP_TREE, PHOTO_STEP_TREE } from '@/lib/constants'
import GlareRemovalView from '@/components/step-detail/GlareRemovalView'
import ResultsView from '@/components/step-detail/ResultsView'

// ── Types ──────────────────────────────────────────────────────────────────────

type StepSelection =
  | { kind: 'job'; stepKey: string }
  | { kind: 'photo'; photoIdx: number; stepKey: string }
  | { kind: 'results' }

function photoNum(idx: number): string {
  return String(idx).padStart(2, '0')
}

function getDebugUrl(job: Job, sel: StepSelection): string | undefined {
  const urls = job.debug_urls ?? {}
  if (sel.kind === 'job') {
    const cfg = JOB_STEP_TREE.find((s) => s.stepKey === sel.stepKey)
    return cfg ? urls[cfg.debugKey] : undefined
  }
  if (sel.kind === 'photo') {
    const cfg = PHOTO_STEP_TREE.find((s) => s.stepKey === sel.stepKey)
    const n = photoNum(sel.photoIdx)
    return cfg ? urls[cfg.debugKeyFn(n)] : undefined
  }
  return undefined
}

function selectionLabel(sel: StepSelection): string {
  if (sel.kind === 'results') return 'Results'
  if (sel.kind === 'job') {
    return JOB_STEP_TREE.find((s) => s.stepKey === sel.stepKey)?.label ?? sel.stepKey
  }
  return PHOTO_STEP_TREE.find((s) => s.stepKey === sel.stepKey)?.label ?? sel.stepKey
}

// ── StepTree ───────────────────────────────────────────────────────────────────

function TreeRow({
  label,
  active,
  available,
  indented = false,
  onClick,
}: {
  label: string
  active: boolean
  available: boolean
  indented?: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      disabled={!available}
      className={[
        'w-full text-left py-2 text-[13px] transition-colors duration-[200ms] border-l-2',
        indented ? 'pl-8 pr-3' : 'px-4',
        active
          ? 'border-sa-amber-500 text-sa-amber-600 dark:text-sa-amber-400 font-semibold bg-sa-amber-50 dark:bg-sa-amber-950/20'
          : available
          ? 'border-transparent text-sa-stone-700 dark:text-sa-stone-300 hover:text-sa-stone-900 dark:hover:text-sa-stone-100 hover:bg-sa-stone-100 dark:hover:bg-sa-stone-800'
          : 'border-transparent text-sa-stone-300 dark:text-sa-stone-600 cursor-not-allowed',
      ].join(' ')}
    >
      {label}
    </button>
  )
}

function StepTree({
  job,
  selection,
  onSelect,
}: {
  job: Job
  selection: StepSelection
  onSelect: (s: StepSelection) => void
}) {
  const urls = job.debug_urls ?? {}

  const isJobActive = (key: string) =>
    selection.kind === 'job' && selection.stepKey === key
  const isPhotoActive = (idx: number, key: string) =>
    selection.kind === 'photo' && selection.photoIdx === idx && selection.stepKey === key
  const isResultsActive = selection.kind === 'results'

  return (
    <nav className="w-[196px] flex-shrink-0 flex flex-col py-2 overflow-y-auto">
      {/* Pre-split job steps */}
      {JOB_STEP_TREE.map(({ stepKey, label, debugKey }) => (
        <TreeRow
          key={stepKey}
          label={label}
          active={isJobActive(stepKey)}
          available={!!urls[debugKey]}
          onClick={() => onSelect({ kind: 'job', stepKey })}
        />
      ))}

      {/* Per-photo sections */}
      {Array.from({ length: Math.max(job.photo_count, 1) }, (_, i) => i + 1).map((photoIdx) => (
        <div key={photoIdx}>
          {job.photo_count > 1 && (
            <div className="px-4 pt-3 pb-1">
              <span className="text-[11px] font-semibold text-sa-stone-400 dark:text-sa-stone-500 uppercase tracking-wider">
                Photo {photoIdx}
              </span>
            </div>
          )}
          {PHOTO_STEP_TREE.map(({ stepKey, label, debugKeyFn }) => {
            const n = photoNum(photoIdx)
            return (
              <TreeRow
                key={`${photoIdx}-${stepKey}`}
                label={label}
                active={isPhotoActive(photoIdx, stepKey)}
                available={!!urls[debugKeyFn(n)]}
                indented={job.photo_count > 1}
                onClick={() => onSelect({ kind: 'photo', photoIdx, stepKey })}
              />
            )
          })}
        </div>
      ))}

      {/* Results — available when output_urls present */}
      {(job.output_urls?.length ?? 0) > 0 && (
        <TreeRow
          label="Results"
          active={isResultsActive}
          available={true}
          onClick={() => onSelect({ kind: 'results' })}
        />
      )}
    </nav>
  )
}

// ── Step canvas views ──────────────────────────────────────────────────────────

function DebugImageCanvas({ url, label }: { url: string | undefined; label: string }) {
  if (!url) {
    return (
      <div className="flex-1 flex items-center justify-center text-sa-stone-400 dark:text-sa-stone-500 text-sm">
        No debug image for this step
      </div>
    )
  }
  return (
    <div className="flex-1 flex justify-center p-6 overflow-auto">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={url}
        alt={label}
        className="max-w-full h-auto rounded-xl shadow-md object-contain self-start"
      />
    </div>
  )
}

function OrientationView({ job, photoIdx }: { job: Job; photoIdx: number }) {
  const n = photoNum(photoIdx)
  const url = job.debug_urls?.[`05b_photo_${n}_oriented`]

  return (
    <div className="flex-1 flex items-start gap-6 p-6 overflow-auto">
      <div className="flex-1 flex justify-center">
        {url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={url}
            alt={`Photo ${photoIdx} — oriented`}
            className="max-w-full h-auto rounded-xl shadow-md object-contain self-start"
          />
        ) : (
          <div className="flex items-center justify-center text-sa-stone-400 dark:text-sa-stone-500 text-sm">
            No image available
          </div>
        )}
      </div>

      {/* Info panel — matches macOS right-side control panel */}
      <div className="w-56 flex-shrink-0 flex flex-col gap-4 pt-1">
        <h3 className="text-[13px] font-semibold text-sa-stone-700 dark:text-sa-stone-200">
          Orientation
        </h3>
        <p className="text-[12px] text-sa-stone-500 dark:text-sa-stone-400 leading-relaxed">
          AI-corrected rotation (0°/90°/180°/270°) applied before glare removal so the
          model receives a semantically upright image.
        </p>
        <p className="text-[11px] text-sa-stone-400 dark:text-sa-stone-500">
          Rotation editing will be available in a future update.
        </p>
      </div>
    </div>
  )
}

function ColorRestoreView({ job, photoIdx }: { job: Job; photoIdx: number }) {
  const n = photoNum(photoIdx)
  const url = job.debug_urls?.[`14_photo_${n}_enhanced`]

  return (
    <div className="flex-1 flex items-start gap-6 p-6 overflow-auto">
      <div className="flex-1 flex justify-center">
        {url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={url}
            alt={`Photo ${photoIdx} — color restored`}
            className="max-w-full h-auto rounded-xl shadow-md object-contain self-start"
          />
        ) : (
          <div className="flex items-center justify-center text-sa-stone-400 dark:text-sa-stone-500 text-sm">
            No image available
          </div>
        )}
      </div>

      {/* Slider panel — display only, Phase 6 makes them interactive */}
      <div className="w-56 flex-shrink-0 flex flex-col gap-4 pt-1">
        <h3 className="text-[13px] font-semibold text-sa-stone-700 dark:text-sa-stone-200">
          Color Restore
        </h3>
        {[
          { label: 'Brightness',  value: 0   },
          { label: 'Saturation',  value: 15  },
          { label: 'Warmth',      value: 0   },
          { label: 'Sharpness',   value: 50  },
        ].map(({ label, value }) => (
          <div key={label} className="flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <span className="text-[12px] text-sa-stone-600 dark:text-sa-stone-300">{label}</span>
              <span className="text-[11px] text-sa-stone-400 dark:text-sa-stone-500">{value > 0 ? `+${value}` : value}%</span>
            </div>
            <div className="relative h-1.5 rounded-full bg-sa-stone-200 dark:bg-sa-stone-700 overflow-hidden">
              <div
                className="absolute inset-y-0 left-0 rounded-full bg-sa-amber-500"
                style={{ width: `${50 + value / 2}%` }}
              />
            </div>
          </div>
        ))}
        <p className="text-[11px] text-sa-stone-400 dark:text-sa-stone-500 pt-1">
          Adjustment controls will be interactive in a future update.
        </p>
      </div>
    </div>
  )
}

// ── Main page ──────────────────────────────────────────────────────────────────

export default function JobDetailPage({ params }: { params: { jobId: string } }) {
  const [job, setJob] = useState<Job | null>(null)
  const [loading, setLoading] = useState(true)
  const [fetchError, setFetchError] = useState<string | null>(null)
  const [selection, setSelection] = useState<StepSelection>({ kind: 'job', stepKey: 'load' })

  useEffect(() => {
    getJob(params.jobId)
      .then((j) => {
        setJob(j)
        // Auto-select first step that has a debug image
        for (const { stepKey, debugKey } of JOB_STEP_TREE) {
          if (j.debug_urls?.[debugKey]) {
            setSelection({ kind: 'job', stepKey })
            return
          }
        }
        // If no pre-split steps, try first per-photo step for photo 1
        for (const { stepKey, debugKeyFn } of PHOTO_STEP_TREE) {
          if (j.debug_urls?.[debugKeyFn('01')]) {
            setSelection({ kind: 'photo', photoIdx: 1, stepKey })
            return
          }
        }
        // Fall back to results if available
        if (j.output_urls?.length) {
          setSelection({ kind: 'results' })
        }
      })
      .catch(() => setFetchError('Could not load job'))
      .finally(() => setLoading(false))
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params.jobId])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-6 h-6 rounded-full border-2 border-sa-amber-500 border-t-transparent animate-spin" />
      </div>
    )
  }

  if (fetchError || !job) {
    return (
      <div className="flex flex-col items-center py-20 gap-4 text-center">
        <p className="text-sa-error text-sm">{fetchError ?? 'Job not found'}</p>
        <Link href="/library" className="text-sa-amber-600 dark:text-sa-amber-400 hover:underline text-sm">
          ← Back to Library
        </Link>
      </div>
    )
  }

  const currentLabel = selectionLabel(selection)

  function renderCanvas() {
    if (!job) return null

    if (selection.kind === 'results') {
      return <ResultsView job={job} />
    }

    if (selection.kind === 'photo') {
      if (selection.stepKey === 'glare_remove') {
        const n = photoNum(selection.photoIdx)
        return (
          <GlareRemovalView
            beforeUrl={job.debug_urls?.[`05b_photo_${n}_oriented`]}
            afterUrl={job.debug_urls?.[`07_photo_${n}_deglared`]}
            photoIdx={selection.photoIdx}
          />
        )
      }
      if (selection.stepKey === 'ai_orient') {
        return <OrientationView job={job} photoIdx={selection.photoIdx} />
      }
      if (selection.stepKey === 'color_restore') {
        return <ColorRestoreView job={job} photoIdx={selection.photoIdx} />
      }
    }

    return <DebugImageCanvas url={getDebugUrl(job, selection)} label={currentLabel} />
  }

  return (
    // -mx-4 -my-8 escapes the (app)/layout.tsx's px-4 py-8 on <main>
    // so the step-detail chrome can be flush edge-to-edge within max-w-6xl
    <div className="-mx-4 -my-8 flex flex-col" style={{ minHeight: 'calc(100vh - 56px)' }}>
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 px-6 py-3.5 border-b border-sa-stone-200 dark:border-sa-stone-800 text-[13px] flex-shrink-0">
        <Link
          href="/library"
          className="text-sa-stone-500 dark:text-sa-stone-400 hover:text-sa-stone-700 dark:hover:text-sa-stone-200 transition-colors duration-[200ms]"
        >
          Library
        </Link>
        <span className="text-sa-stone-300 dark:text-sa-stone-600">/</span>
        <span
          className="text-sa-stone-600 dark:text-sa-stone-300 truncate max-w-[200px]"
          title={job.input_filename}
        >
          {job.input_filename}
        </span>
        <span className="text-sa-stone-300 dark:text-sa-stone-600">/</span>
        <span className="text-sa-stone-900 dark:text-sa-stone-100 font-semibold">{currentLabel}</span>
      </div>

      {/* 3-pane body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left sidebar — 196px, independently scrollable */}
        <div className="border-r border-sa-stone-200 dark:border-sa-stone-800 overflow-y-auto flex-shrink-0">
          <StepTree job={job} selection={selection} onSelect={setSelection} />
        </div>

        {/* Right canvas — switches between views with a fade */}
        <AnimatePresence mode="wait">
          <motion.div
            key={JSON.stringify(selection)}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="flex-1 flex overflow-auto min-w-0"
          >
            {renderCanvas()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}
