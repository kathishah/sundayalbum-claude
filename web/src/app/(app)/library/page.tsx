'use client'

import { useEffect, useState, useCallback } from 'react'
import { useJobsStore } from '@/stores/jobs-store'
import { listJobs, createJob, s3Upload, startJob } from '@/lib/api'
import { ALLOWED_EXTENSIONS } from '@/lib/constants'
import AlbumPageCard from '@/components/library/AlbumPageCard'
import DropZone from '@/components/library/DropZone'
import type { Job } from '@/lib/types'

export default function LibraryPage() {
  const { jobs, setJobs, upsertJob } = useJobsStore()
  const [loading, setLoading] = useState(true)
  const [uploadError, setUploadError] = useState<string | null>(null)

  useEffect(() => {
    listJobs()
      .then((res) => setJobs(res.jobs))
      .catch(() => {
        // silently fail — jobs list will be empty
      })
      .finally(() => setLoading(false))
  }, [setJobs])

  const handleFiles = useCallback(
    async (files: FileList) => {
      setUploadError(null)
      const fileArray = Array.from(files)

      for (const file of fileArray) {
        const ext = '.' + file.name.split('.').pop()?.toLowerCase()
        if (!ALLOWED_EXTENSIONS.has(ext)) {
          setUploadError(
            `Unsupported file type: ${ext}. Allowed: HEIC, JPG, PNG.`,
          )
          continue
        }

        // Optimistic placeholder
        const optimisticJob: Job = {
          job_id: `uploading-${Date.now()}`,
          status: 'uploading',
          current_step: '',
          step_detail: '',
          input_filename: file.name,
          input_stem: file.name.replace(/\.[^.]+$/, ''),
          photo_count: 0,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          error_message: '',
          processing_time: 0,
          output_keys: [],
        }
        upsertJob(optimisticJob)

        try {
          // 1. Create job → get presigned upload URL
          const created = await createJob(file.name, file.size)

          // 2. Upload directly to S3
          await s3Upload(created.upload_url, file)

          // 3. Start processing
          await startJob(created.job_id)

          // 4. Fetch the real job record and replace optimistic entry
          const jobRes = await listJobs()
          const realJob = jobRes.jobs.find((j) => j.job_id === created.job_id)
          if (realJob) {
            // Remove optimistic, add real
            setJobs([
              realJob,
              ...jobRes.jobs.filter((j) => j.job_id !== created.job_id),
            ])
          } else {
            setJobs(jobRes.jobs)
          }
        } catch (err) {
          // Replace optimistic with failed state
          upsertJob({
            ...optimisticJob,
            status: 'failed',
            error_message:
              err instanceof Error ? err.message : 'Upload failed',
          })
          setUploadError(
            err instanceof Error ? err.message : 'Upload failed. Try again.',
          )
        }
      }
    },
    [upsertJob, setJobs],
  )

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-8 h-8 rounded-full border-2 border-sa-amber-500 border-t-transparent animate-spin" />
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="font-display text-2xl font-bold text-sa-stone-900 dark:text-sa-stone-50">
          Library
        </h1>
      </div>

      {uploadError && (
        <div className="mb-4 p-3 rounded-lg bg-red-50 dark:bg-red-950 border border-sa-error text-sa-error text-sm">
          {uploadError}
        </div>
      )}

      {jobs.length === 0 ? (
        <DropZone onFiles={handleFiles} />
      ) : (
        <div>
          <div className="mb-6">
            <DropZone onFiles={handleFiles} compact />
          </div>
          <div className="grid grid-cols-1 gap-4">
            {jobs.map((job) => (
              <AlbumPageCard key={job.job_id} job={job} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
