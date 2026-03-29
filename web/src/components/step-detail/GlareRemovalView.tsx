'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

interface GlareRemovalViewProps {
  beforeUrl: string | undefined
  afterUrl: string | undefined
  /** Changing photoIdx re-triggers the reveal animation */
  photoIdx: number
}

function ImagePane({
  url,
  label,
  children,
}: {
  url: string | undefined
  label: string
  children?: React.ReactNode
}) {
  return (
    <div className="flex-1 flex flex-col gap-2 min-w-0">
      <span className="text-[11px] font-semibold text-sa-stone-400 dark:text-sa-stone-500 uppercase tracking-wider text-center">
        {label}
      </span>
      <div className="relative rounded-xl overflow-hidden bg-sa-surface">
        {url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={url} alt={label} className="w-full object-contain" />
        ) : (
          <div className="h-64 flex items-center justify-center text-sa-stone-400 dark:text-sa-stone-500 text-sm">
            No image available
          </div>
        )}
        {children}
      </div>
    </div>
  )
}

export default function GlareRemovalView({
  beforeUrl,
  afterUrl,
  photoIdx,
}: GlareRemovalViewProps) {
  const [revealed, setRevealed] = useState(false)

  // Re-trigger reveal animation each time the photo changes
  useEffect(() => {
    setRevealed(false)
    // Two rAF ticks ensure the opacity-0 state is painted before we start the transition
    const raf1 = requestAnimationFrame(() => {
      const raf2 = requestAnimationFrame(() => setRevealed(true))
      return () => cancelAnimationFrame(raf2)
    })
    return () => cancelAnimationFrame(raf1)
  }, [photoIdx])

  return (
    <div className="flex-1 flex flex-col items-center p-6 gap-6 overflow-auto">
      <div className="flex items-start gap-8 w-full max-w-4xl">
        {/* Before */}
        <ImagePane url={beforeUrl} label="Before" />

        {/* After — saReveal: 600ms cubic-bezier(0.16,1,0.3,1); glow delay 400ms */}
        <div className="flex-1 flex flex-col gap-2 min-w-0">
          <span className="text-[11px] font-semibold text-sa-stone-400 dark:text-sa-stone-500 uppercase tracking-wider text-center">
            After
          </span>
          <motion.div
            className="rounded-xl overflow-hidden bg-sa-surface"
            animate={{
              opacity: revealed ? 1 : 0,
              boxShadow: revealed
                ? '0 0 40px 10px rgba(217,119,6,0.28)'
                : '0 0 0px 0px rgba(217,119,6,0)',
            }}
            transition={{
              opacity:    { duration: 0.6, ease: [0.16, 1, 0.3, 1] },
              boxShadow:  { duration: 0.6, ease: [0.16, 1, 0.3, 1], delay: 0.4 },
            }}
          >
            {afterUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={afterUrl} alt="After glare removal" className="w-full object-contain" />
            ) : (
              <div className="h-64 flex items-center justify-center text-sa-stone-400 dark:text-sa-stone-500 text-sm">
                No image available
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  )
}
