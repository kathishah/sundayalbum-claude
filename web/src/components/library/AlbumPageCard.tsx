'use client'

import { useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { Job, StepUpdate } from '@/lib/types'
import { BACKEND_TO_VISUAL, TOTAL_VISUAL_STEPS } from '@/lib/constants'
import { useJobProgress } from '@/lib/websocket'
import { useJobsStore } from '@/stores/jobs-store'
import { getJob, deleteJob } from '@/lib/api'
import PipelineProgressWheel from './ProgressWheel'

interface AlbumPageCardProps {
  job: Job
  /** True when a different card is currently expanded — dims and scales this card */
  isOtherExpanded: boolean
  onExpand: () => void
}

// ── ThumbBox ────────────────────────────────────────────────────────────────

interface ThumbBoxProps {
  src: string | undefined
  width: number
  height: number
}

function ThumbBox({ src, width, height }: ThumbBoxProps) {
  return (
    <div
      className="flex-shrink-0 rounded-lg overflow-hidden bg-sa-surface"
      style={{ width, height }}
    >
      {src ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={src}
          alt="Before"
          className="w-full h-full object-cover"
        />
      ) : (
        <div className="w-full h-full flex items-center justify-center">
          {/* Photo icon placeholder — matches macOS empty state */}
          <svg
            viewBox="0 0 24 24"
            width={Math.round(width * 0.45)}
            height={Math.round(height * 0.45)}
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            className="text-sa-stone-300 dark:text-sa-stone-600"
          >
            <rect x="3" y="3" width="18" height="18" rx="3" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <path d="M21 15l-5-5L5 21" />
          </svg>
        </div>
      )}
    </div>
  )
}

// ── AfterSection ─────────────────────────────────────────────────────────────

interface AfterSectionProps {
  job: Job
  height: number
  /** When provided: equal-slot compact mode. When undefined: natural-ratio expanded mode (up to 3). */
  sectionWidth?: number
}

function AfterSection({ job, height, sectionWidth }: AfterSectionProps) {
  const completedCount =
    job.status === 'complete'
      ? TOTAL_VISUAL_STEPS
      : job.status === 'processing'
        ? (BACKEND_TO_VISUAL[job.current_step] ?? 0)
        : 0

  if (job.status === 'complete' && job.output_urls && job.output_urls.length > 0) {
    const photos = job.output_urls

    if (sectionWidth !== undefined) {
      // Compact equal-slot mode
      const gap = 4
      const n = photos.length
      const slotW = Math.max((sectionWidth - gap * (n - 1)) / n, 20)
      return (
        <div className="flex items-center" style={{ gap, height }}>
          {photos.map((url, i) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={i}
              src={url}
              alt={`Photo ${i + 1}`}
              className="object-cover rounded-[6px] flex-shrink-0"
              style={{ width: slotW, height }}
            />
          ))}
        </div>
      )
    }

    // Expanded natural-ratio mode (up to 3 + overflow badge)
    const visible = photos.slice(0, 3)
    const overflow = photos.length - visible.length
    return (
      <div className="flex items-center gap-2" style={{ height }}>
        {visible.map((url, i) => (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            key={i}
            src={url}
            alt={`Photo ${i + 1}`}
            className="object-cover rounded-[6px] flex-shrink-0"
            style={{ height, width: 'auto', maxWidth: height * 1.5 }}
          />
        ))}
        {overflow > 0 && (
          <div
            className="flex-shrink-0 flex items-center justify-center rounded-[6px] bg-sa-stone-200 dark:bg-sa-stone-700"
            style={{ width: height * 0.65, height }}
          >
            <span className="text-xs font-semibold text-sa-stone-500 dark:text-sa-stone-400">
              +{overflow}
            </span>
          </div>
        )}
      </div>
    )
  }

  // Processing / uploading / failed — pie wheel
  return (
    <div
      className="flex items-center justify-center"
      style={{ height, minWidth: height }}
    >
      <PipelineProgressWheel completedCount={completedCount} size={height} />
    </div>
  )
}

// ── AlbumPageCard ─────────────────────────────────────────────────────────────

export default function AlbumPageCard({ job, isOtherExpanded, onExpand }: AlbumPageCardProps) {
  const { upsertJob, removeJob } = useJobsStore()
  const [isHovered, setIsHovered] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)

  const isActive =
    job.status === 'uploading' || job.status === 'processing'

  // For complete jobs loaded from the list endpoint, output_urls (and debug_urls) are
  // absent. Fetch the full job record once on mount so the after-thumbnails appear.
  useEffect(() => {
    if (
      job.status === 'complete' &&
      (!job.output_urls || job.output_urls.length === 0)
    ) {
      getJob(job.job_id)
        .then((full) => upsertJob({ ...full, preview_url: job.preview_url }))
        .catch(() => {})
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.job_id])

  const handleUpdate = useCallback(
    (update: StepUpdate) => {
      if (update.status === 'complete' || update.status === 'failed') {
        getJob(update.jobId)
          .then((fullJob) => upsertJob(fullJob))
          .catch(() =>
            upsertJob({
              ...job,
              status: update.status,
              current_step: update.step,
              step_detail: update.detail,
            }),
          )
      } else {
        upsertJob({
          ...job,
          status: update.status,
          current_step: update.step,
          step_detail: update.detail,
          ...(update.photoCount > 0 && { photo_count: update.photoCount }),
        })
      }
    },
    [job, upsertJob],
  )

  useJobProgress({ jobId: job.job_id, onUpdate: handleUpdate, enabled: isActive })

  async function handleDelete(e: React.MouseEvent) {
    e.stopPropagation()
    setIsDeleting(true)
    try {
      await deleteJob(job.job_id)
    } finally {
      removeJob(job.job_id)
    }
  }

  // Before-thumbnail: prefer debug_urls['load'] (01_loaded.jpg), fall back to preview_url
  const beforeSrc = job.debug_urls?.['load'] ?? job.preview_url

  const thumbH = 88
  const deleteColor =
    job.status === 'processing' || job.status === 'uploading'
      ? 'text-sa-error'
      : 'text-sa-stone-400 dark:text-sa-stone-500'

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{
        opacity: isOtherExpanded ? 0.3 : 1,
        scale: isOtherExpanded ? 0.95 : 1,
        y: 0,
      }}
      transition={
        isOtherExpanded !== undefined
          ? { type: 'spring', stiffness: 280, damping: 22 }
          : { duration: 0.35, ease: [0.16, 1, 0.3, 1] }
      }
      className="relative rounded-xl overflow-hidden cursor-pointer select-none"
      style={{ background: 'rgb(var(--sa-card))', boxShadow: 'var(--sa-shadow-card)' }}
      onClick={onExpand}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* ── Thumbnail row (112px = 88px thumb + 24px vertical padding) ── */}
      <div className="flex items-center justify-center h-[112px] px-3">
        <div className="flex items-center gap-[10px] w-full">
          {/* Before thumbnail */}
          <ThumbBox src={beforeSrc} width={60} height={thumbH} />

          {/* Arrow — saAmber400 in compact card */}
          <span className="text-[11px] font-semibold text-sa-amber-400 flex-shrink-0 leading-none">
            →
          </span>

          {/* After section — flex:1 fills remaining space */}
          <div className="flex-1 min-w-0">
            <AfterSection job={job} height={thumbH} sectionWidth={undefined} />
          </div>
        </div>
      </div>

      {/* ── Filename ── */}
      <p
        className="text-center text-[12px] font-semibold text-sa-stone-700 dark:text-sa-stone-200 truncate px-3 pt-1 pb-2.5"
        title={job.input_filename}
        style={{
          // Middle truncation via direction trick isn't supported in CSS; use end truncation
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {job.input_filename}
      </p>

      {/* ── Hover delete button ── */}
      <AnimatePresence>
        {isHovered && !isDeleting && (
          <motion.button
            key="delete"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className={`absolute top-2 right-2 w-5 h-5 flex items-center justify-center ${deleteColor} hover:opacity-70`}
            onClick={handleDelete}
            aria-label="Delete job"
          >
            {/* × circle — 16px matching macOS xmark.circle.fill */}
            <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor">
              <path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm3.25 10.19L10.19 11.25 8 9.06l-2.19 2.19L4.75 10.19 6.94 8 4.75 5.81 5.81 4.75 8 6.94l2.19-2.19 1.06 1.06L9.06 8l2.19 2.19z" />
            </svg>
          </motion.button>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
