import Link from 'next/link'

interface JobDetailPageProps {
  params: { jobId: string }
}

export default function JobDetailPage({ params }: JobDetailPageProps) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <p className="text-sa-stone-500 dark:text-sa-stone-400 text-lg mb-6">
        Step detail view coming in Phase 5
      </p>
      <p className="text-sa-stone-400 dark:text-sa-stone-600 text-sm mb-8 font-mono">
        Job: {params.jobId}
      </p>
      <Link
        href="/library"
        className="text-sa-amber-600 dark:text-sa-amber-400 hover:underline transition-colors duration-[200ms]"
      >
        ← Back to Library
      </Link>
    </div>
  )
}
