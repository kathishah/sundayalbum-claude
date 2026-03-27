'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import type { Job } from '@/lib/types'
import Button from '@/components/ui/Button'

interface ExpandedCardProps {
  job: Job
  onClose: () => void
}

export default function ExpandedCard({ job, onClose }: ExpandedCardProps) {
  return (
    <>
      {/* Backdrop */}
      <motion.div
        key="backdrop"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Panel */}
      <motion.div
        key="panel"
        initial={{ opacity: 0, scale: 0.96, y: 16 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.96, y: 16 }}
        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
        className="fixed inset-x-4 top-1/2 -translate-y-1/2 z-50 max-w-3xl mx-auto bg-white dark:bg-sa-stone-900 rounded-2xl border border-sa-stone-200 dark:border-sa-stone-800 shadow-xl overflow-hidden max-h-[90vh] flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-sa-stone-200 dark:border-sa-stone-800">
          <div>
            <h2 className="font-display text-lg font-semibold text-sa-stone-900 dark:text-sa-stone-50 truncate">
              {job.input_filename}
            </h2>
            <p className="text-xs text-sa-stone-500 dark:text-sa-stone-400 mt-0.5">
              {job.photo_count > 0
                ? `${job.photo_count} photo${job.photo_count !== 1 ? 's' : ''} extracted`
                : 'Processing complete'}
              {job.processing_time > 0 &&
                ` · ${job.processing_time.toFixed(1)}s`}
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="w-8 h-8 flex items-center justify-center rounded-lg text-sa-stone-500 hover:text-sa-stone-800 dark:hover:text-sa-stone-100 hover:bg-sa-stone-100 dark:hover:bg-sa-stone-800 transition-colors duration-[200ms]"
          >
            ✕
          </button>
        </div>

        {/* Photo grid */}
        <div className="flex-1 overflow-y-auto p-6">
          {job.output_urls && job.output_urls.length > 0 ? (
            <div
              className={
                job.output_urls.length === 1
                  ? 'grid grid-cols-1 gap-3'
                  : job.output_urls.length === 2
                    ? 'grid grid-cols-2 gap-3'
                    : 'grid grid-cols-3 gap-3'
              }
            >
              {job.output_urls.map((url, idx) => (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  key={idx}
                  src={url}
                  alt={`Extracted photo ${idx + 1}`}
                  className="w-full rounded-xl object-cover bg-sa-stone-100 dark:bg-sa-stone-800"
                  style={{ aspectRatio: '4/3' }}
                />
              ))}
            </div>
          ) : (
            <p className="text-sm text-sa-stone-500 dark:text-sa-stone-400 text-center py-8">
              No output photos available.
            </p>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-sa-stone-200 dark:border-sa-stone-800 flex items-center justify-between gap-4">
          <Link
            href={`/jobs/${job.job_id}`}
            className="text-sm text-sa-amber-600 dark:text-sa-amber-400 hover:underline transition-colors duration-[200ms]"
          >
            Open Step Detail →
          </Link>
          <Button variant="secondary" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>
      </motion.div>
    </>
  )
}
