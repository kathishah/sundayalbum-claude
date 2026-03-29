'use client'

import { useState } from 'react'
import type { Job } from '@/lib/types'

interface ResultsViewProps {
  job: Job
}

export default function ResultsView({ job }: ResultsViewProps) {
  const outputUrls = job.output_urls ?? []
  const [selected, setSelected] = useState(0)

  // "Before" for comparison: the loaded/input thumbnail
  const beforeUrl = job.debug_urls?.['01_loaded'] ?? job.thumbnail_url

  if (outputUrls.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-sa-stone-400 dark:text-sa-stone-500 text-sm">
        No output photos available
      </div>
    )
  }

  function handleDownloadAll() {
    outputUrls.forEach((url, i) => {
      const a = document.createElement('a')
      a.href = url
      a.download = `photo_${String(i + 1).padStart(2, '0')}.jpg`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
    })
  }

  return (
    <div className="flex-1 flex items-start gap-6 p-6 overflow-auto min-h-0">
      {/* Photo grid */}
      <div className="flex-1 min-w-0">
        <div
          className="grid gap-3"
          style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))' }}
        >
          {outputUrls.map((url, i) => (
            <button
              key={i}
              onClick={() => setSelected(i)}
              className={[
                'relative rounded-xl overflow-hidden bg-sa-surface transition-all duration-[200ms]',
                'aspect-[3/4]',
                selected === i
                  ? 'ring-2 ring-sa-amber-500 ring-offset-2 ring-offset-white dark:ring-offset-sa-stone-950'
                  : 'hover:ring-2 hover:ring-sa-stone-300 dark:hover:ring-sa-stone-600 hover:ring-offset-1',
              ].join(' ')}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={url}
                alt={`Photo ${i + 1}`}
                className="w-full h-full object-cover"
              />
              <div className="absolute bottom-0 inset-x-0 py-1 px-2 bg-black/40 backdrop-blur-sm">
                <span className="text-[10px] font-mono text-white/80">
                  Photo {i + 1}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Comparison panel */}
      <div className="w-72 flex-shrink-0 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h3 className="text-[13px] font-semibold text-sa-stone-700 dark:text-sa-stone-200">
            Photo {selected + 1}
            <span className="text-sa-stone-400 dark:text-sa-stone-500 font-normal ml-1">
              of {outputUrls.length}
            </span>
          </h3>
        </div>

        {/* Before */}
        <div className="flex flex-col gap-1.5">
          <p className="text-[10px] font-semibold text-sa-stone-400 dark:text-sa-stone-500 uppercase tracking-wider">
            Original
          </p>
          {beforeUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={beforeUrl}
              alt="Original input"
              className="w-full rounded-lg object-cover bg-sa-surface"
            />
          ) : (
            <div className="w-full h-32 rounded-lg bg-sa-surface" />
          )}
        </div>

        {/* After */}
        <div className="flex flex-col gap-1.5">
          <p className="text-[10px] font-semibold text-sa-stone-400 dark:text-sa-stone-500 uppercase tracking-wider">
            Processed
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={outputUrls[selected]}
            alt={`Photo ${selected + 1} — processed`}
            className="w-full rounded-lg object-cover bg-sa-surface"
          />
        </div>

        {/* Metadata */}
        <p className="text-[11px] font-mono text-sa-stone-400 dark:text-sa-stone-500 truncate" title={job.input_filename}>
          {job.input_filename}
        </p>

        {/* Download actions */}
        <a
          href={outputUrls[selected]}
          download={`photo_${String(selected + 1).padStart(2, '0')}.jpg`}
          className="block w-full text-center px-3 py-2 rounded-lg text-[13px] font-semibold text-white bg-sa-amber-500 hover:bg-sa-amber-600 transition-colors duration-[200ms]"
        >
          Download Photo {selected + 1}
        </a>

        {outputUrls.length > 1 && (
          <button
            onClick={handleDownloadAll}
            className="w-full text-center px-3 py-2 rounded-lg text-[13px] font-semibold text-sa-stone-600 dark:text-sa-stone-300 bg-sa-stone-100 dark:bg-sa-stone-800 hover:bg-sa-stone-200 dark:hover:bg-sa-stone-700 transition-colors duration-[200ms]"
          >
            Download All ({outputUrls.length})
          </button>
        )}
      </div>
    </div>
  )
}
