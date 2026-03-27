'use client'

import { motion } from 'framer-motion'

interface ProgressWheelProps {
  progress: number
  stepLabel: string
  size?: number
}

export default function ProgressWheel({
  progress,
  stepLabel,
  size = 80,
}: ProgressWheelProps) {
  const clampedProgress = Math.max(0, Math.min(1, progress))
  const strokeWidth = 6
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const dashOffset = circumference * (1 - clampedProgress)
  const cx = size / 2
  const cy = size / 2

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          {/* Track */}
          <circle
            cx={cx}
            cy={cy}
            r={radius}
            fill="none"
            stroke="currentColor"
            strokeWidth={strokeWidth}
            className="text-sa-stone-200 dark:text-sa-stone-700"
          />
          {/* Progress arc */}
          <motion.circle
            cx={cx}
            cy={cy}
            r={radius}
            fill="none"
            stroke="currentColor"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            className="text-sa-amber-500"
            animate={{ strokeDashoffset: dashOffset }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          />
        </svg>
        {/* Percentage text */}
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xs font-semibold text-sa-stone-700 dark:text-sa-stone-300">
            {Math.round(clampedProgress * 100)}%
          </span>
        </div>
      </div>
      <span className="text-xs text-sa-stone-500 dark:text-sa-stone-400 text-center max-w-[80px] leading-tight">
        {stepLabel}
      </span>
    </div>
  )
}
