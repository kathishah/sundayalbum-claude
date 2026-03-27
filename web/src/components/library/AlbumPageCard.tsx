'use client'

import { useState, useCallback } from 'react'
import Link from 'next/link'
import clsx from 'clsx'
import { motion, AnimatePresence } from 'framer-motion'
import type { Job, StepUpdate } from '@/lib/types'
import { STEP_LABELS, STEP_PROGRESS } from '@/lib/constants'
import { useJobProgress } from '@/lib/websocket'
import { useJobsStore } from '@/stores/jobs-store'
import { getJob } from '@/lib/api'
import Spinner from '@/components/ui/Spinner'
import Button from '@/components/ui/Button'
import ProgressWheel from './ProgressWheel'
import ExpandedCard from './ExpandedCard'

interface AlbumPageCardProps {
  job: Job
}

export default function AlbumPageCard({ job: initialJob }: AlbumPageCardProps) {
  const { upsertJob } = useJobsStore()
  const [localJob, setLocalJob] = useState<Job>(initialJob)
  const [progress, setProgress] = useState(
    STEP_PROGRESS[initialJob.current_step] ?? 0,
  )
  const [expanded, setExpanded] = useState(false)

  const isActive =
    localJob.status === 'uploading' || localJob.status === 'processing'

  const handleUpdate = useCallback(
    (update: StepUpdate) => {
      const newProgress =
        update.progress >= 0
          ? update.progress
          : STEP_PROGRESS[update.step] ?? progress

      setProgress(newProgress)

      // If complete/failed, refresh full job record for output_urls
      if (update.status === 'complete' || update.status === 'failed') {
        getJob(update.jobId)
          .then((fullJob) => {
            setLocalJob(fullJob)
            upsertJob(fullJob)
          })
          .catch(() => {
            setLocalJob((prev) => ({
              ...prev,
              status: update.status,
              current_step: update.step,
              step_detail: update.detail,
            }))
          })
      } else {
        setLocalJob((prev) => ({
          ...prev,
          status: update.status,
          current_step: update.step,
          step_detail: update.detail,
          photo_count: update.photoCount || prev.photo_count,
        }))
      }
    },
    [upsertJob, progress],
  )

  useJobProgress({
    jobId: localJob.job_id,
    onUpdate: handleUpdate,
    enabled: isActive,
  })

  const stepLabel = STEP_LABELS[localJob.current_step] ?? localJob.current_step

  return (
    <>
      <motion.div
        layout
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
        className="rounded-xl bg-sa-stone-100 dark:bg-sa-stone-900 border border-sa-stone-200 dark:border-sa-stone-800 shadow-sm overflow-hidden"
      >
        {/* Uploading */}
        {localJob.status === 'uploading' && (
          <div className="flex items-center gap-4 p-4">
            <div className="w-16 h-16 rounded-lg bg-sa-stone-200 dark:bg-sa-stone-800 flex items-center justify-center flex-shrink-0">
              <Spinner size="md" className="text-sa-amber-500" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-sa-stone-800 dark:text-sa-stone-100 truncate">
                {localJob.input_filename}
              </p>
              <p className="text-xs text-sa-stone-500 dark:text-sa-stone-400 mt-0.5">
                Uploading…
              </p>
            </div>
          </div>
        )}

        {/* Processing */}
        {localJob.status === 'processing' && (
          <div className="flex items-center gap-4 p-4">
            {/* Thumbnail */}
            <div className="w-20 h-20 rounded-lg overflow-hidden bg-sa-stone-200 dark:bg-sa-stone-800 flex-shrink-0">
              {localJob.upload_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={localJob.upload_url}
                  alt={localJob.input_stem}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <span className="text-sa-stone-400 text-xs">
                    {localJob.input_stem.slice(0, 4)}
                  </span>
                </div>
              )}
            </div>

            {/* Text */}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-sa-stone-800 dark:text-sa-stone-100 truncate mb-1">
                {localJob.input_filename}
              </p>
              <p className="text-xs text-sa-stone-500 dark:text-sa-stone-400 truncate">
                {localJob.step_detail || stepLabel}
              </p>
            </div>

            {/* Progress wheel */}
            <div className="flex-shrink-0">
              <ProgressWheel progress={progress} stepLabel={stepLabel} size={72} />
            </div>
          </div>
        )}

        {/* Complete */}
        {localJob.status === 'complete' && (
          <div className="p-4">
            <div className="flex items-center justify-between mb-3">
              <p className="text-sm font-medium text-sa-stone-800 dark:text-sa-stone-100 truncate">
                {localJob.input_filename}
              </p>
              <div className="flex items-center gap-2 flex-shrink-0 ml-2">
                <span className="text-xs text-sa-success font-medium">
                  {localJob.photo_count > 0
                    ? `${localJob.photo_count} photo${localJob.photo_count !== 1 ? 's' : ''}`
                    : 'Complete'}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setExpanded(true)}
                  className="text-xs"
                >
                  Expand
                </Button>
              </div>
            </div>

            {/* Output thumbnails grid */}
            {localJob.output_urls && localJob.output_urls.length > 0 ? (
              <div
                className={clsx(
                  'grid gap-2',
                  localJob.output_urls.length === 1
                    ? 'grid-cols-1'
                    : localJob.output_urls.length === 2
                      ? 'grid-cols-2'
                      : 'grid-cols-3',
                )}
              >
                {localJob.output_urls.slice(0, 6).map((url, idx) => (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    key={idx}
                    src={url}
                    alt={`Photo ${idx + 1}`}
                    className="w-full aspect-[4/3] object-cover rounded-lg bg-sa-stone-200 dark:bg-sa-stone-800"
                  />
                ))}
              </div>
            ) : (
              <Link
                href={`/jobs/${localJob.job_id}`}
                className="text-xs text-sa-amber-600 dark:text-sa-amber-400 hover:underline"
              >
                View step detail →
              </Link>
            )}
          </div>
        )}

        {/* Failed */}
        {localJob.status === 'failed' && (
          <div className="flex items-center gap-4 p-4">
            <div className="w-10 h-10 rounded-full bg-red-100 dark:bg-red-950 flex items-center justify-center flex-shrink-0">
              <span className="text-sa-error text-lg">!</span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-sa-stone-800 dark:text-sa-stone-100 truncate">
                {localJob.input_filename}
              </p>
              <p className="text-xs text-sa-error mt-0.5 truncate">
                {localJob.error_message || 'Processing failed'}
              </p>
            </div>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                // Retry: reload page to re-trigger upload flow
                window.location.reload()
              }}
              className="flex-shrink-0"
            >
              Retry
            </Button>
          </div>
        )}
      </motion.div>

      {/* Expanded overlay */}
      <AnimatePresence>
        {expanded && localJob.status === 'complete' && (
          <ExpandedCard
            job={localJob}
            onClose={() => setExpanded(false)}
          />
        )}
      </AnimatePresence>
    </>
  )
}
