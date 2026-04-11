'use client'

import { useRef, useState, useCallback, useEffect } from 'react'
import clsx from 'clsx'

interface BeforeAfterSliderProps {
  beforeSrc?: string
  afterSrc?: string
  beforeLabel?: string
  afterLabel?: string
  beforeAlt?: string
  afterAlt?: string
  className?: string
  /** Render slot for 'before' pane when no beforeSrc is supplied */
  beforeSlot?: React.ReactNode
  /** Render slot for 'after' pane when no afterSrc is supplied */
  afterSlot?: React.ReactNode
}

export default function BeforeAfterSlider({
  beforeSrc,
  afterSrc,
  beforeLabel = 'Before',
  afterLabel = 'After',
  beforeAlt = 'Before image',
  afterAlt = 'After image',
  className,
  beforeSlot,
  afterSlot,
}: BeforeAfterSliderProps) {
  const [position, setPosition] = useState(40)
  const containerRef = useRef<HTMLDivElement>(null)
  const isDragging = useRef(false)

  const updatePosition = useCallback((clientX: number) => {
    const el = containerRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width))
    setPosition((x / rect.width) * 100)
  }, [])

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      isDragging.current = true
      updatePosition(e.clientX)
      e.preventDefault()
    },
    [updatePosition],
  )

  const onTouchStart = useCallback(
    (e: React.TouchEvent) => {
      isDragging.current = true
      updatePosition(e.touches[0].clientX)
    },
    [updatePosition],
  )

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (isDragging.current) updatePosition(e.clientX)
    }
    const onTouchMove = (e: TouchEvent) => {
      if (isDragging.current) updatePosition(e.touches[0].clientX)
    }
    const onUp = () => {
      isDragging.current = false
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    window.addEventListener('touchmove', onTouchMove, { passive: true })
    window.addEventListener('touchend', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      window.removeEventListener('touchmove', onTouchMove)
      window.removeEventListener('touchend', onUp)
    }
  }, [updatePosition])

  return (
    <div
      ref={containerRef}
      className={clsx(
        'relative overflow-hidden rounded-2xl select-none cursor-ew-resize',
        'border border-sa-stone-200 dark:border-sa-stone-700',
        className,
      )}
      onMouseDown={onMouseDown}
      onTouchStart={onTouchStart}
      role="img"
      aria-label={`Before/after comparison: ${beforeLabel} and ${afterLabel}`}
    >
      {/* After (full width, behind) */}
      <div className="w-full h-full">
        {afterSrc ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={afterSrc} alt={afterAlt} className="w-full h-full object-cover" draggable={false} />
        ) : (
          afterSlot
        )}
      </div>

      {/* Before (clipped to left side) */}
      <div
        className="absolute inset-0 overflow-hidden"
        style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
      >
        {beforeSrc ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={beforeSrc}
            alt={beforeAlt}
            className="w-full h-full object-cover"
            draggable={false}
          />
        ) : (
          beforeSlot
        )}
      </div>

      {/* Divider line */}
      <div
        className="absolute inset-y-0 w-0.5 bg-white shadow-[0_0_6px_rgba(0,0,0,0.4)] pointer-events-none"
        style={{ left: `${position}%`, transform: 'translateX(-50%)' }}
      />

      {/* Handle knob */}
      <div
        className="absolute top-1/2 w-9 h-9 rounded-full bg-white shadow-[0_2px_12px_rgba(0,0,0,0.35)] flex items-center justify-center pointer-events-none"
        style={{ left: `${position}%`, transform: 'translate(-50%, -50%)' }}
      >
        <svg
          viewBox="0 0 20 20"
          className="w-5 h-5 text-sa-stone-600"
          fill="currentColor"
          aria-hidden="true"
        >
          <path d="M7 4l-4 6 4 6M13 4l4 6-4 6" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      {/* Labels */}
      <span className="absolute bottom-3 left-3 text-xs font-medium text-white bg-black/50 px-2 py-1 rounded-full pointer-events-none">
        {beforeLabel}
      </span>
      <span className="absolute bottom-3 right-3 text-xs font-medium text-white bg-black/50 px-2 py-1 rounded-full pointer-events-none">
        {afterLabel}
      </span>
    </div>
  )
}
