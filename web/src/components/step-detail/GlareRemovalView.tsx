'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

interface GlareRemovalViewProps {
  afterUrl: string | undefined
  /** Changing photoIdx re-triggers the reveal animation */
  photoIdx: number
}

export default function GlareRemovalView({ afterUrl, photoIdx }: GlareRemovalViewProps) {
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
    <div className="flex-1 flex justify-center p-6 overflow-auto">
      {/* saReveal: 600ms cubic-bezier(0.16,1,0.3,1); amber glow delay 400ms */}
      <motion.div
        className="rounded-xl overflow-hidden bg-sa-surface self-start"
        animate={{
          opacity: revealed ? 1 : 0,
          boxShadow: revealed
            ? '0 0 40px 10px rgba(217,119,6,0.28)'
            : '0 0 0px 0px rgba(217,119,6,0)',
        }}
        transition={{
          opacity:   { duration: 0.6, ease: [0.16, 1, 0.3, 1] },
          boxShadow: { duration: 0.6, ease: [0.16, 1, 0.3, 1], delay: 0.4 },
        }}
      >
        {afterUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={afterUrl} alt="Glare removed" className="max-w-full h-auto object-contain" />
        ) : (
          <div className="w-64 h-64 flex items-center justify-center text-sa-stone-400 dark:text-sa-stone-500 text-sm">
            No image available
          </div>
        )}
      </motion.div>
    </div>
  )
}
