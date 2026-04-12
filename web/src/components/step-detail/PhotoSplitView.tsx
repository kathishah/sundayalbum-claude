'use client'

/**
 * PhotoSplitView — interactive photo boundary editor.
 *
 * Displays the `04_photo_boundaries` debug image with an SVG overlay
 * containing four draggable corner handles per detected region.  The user
 * can resize existing regions, draw new ones, or delete unwanted ones, then
 * confirm to reprocess from `photo_detect` with the adjusted bboxes.
 *
 * Mirrors macOS PhotoSplitStepView: same geometry, same four-corner approach.
 */

import { useState, useRef, useEffect } from 'react'
import { reprocessJob } from '@/lib/api'
import type { Job } from '@/lib/types'

// ── Types ──────────────────────────────────────────────────────────────────────

interface Region {
  id: number
  bbox: [number, number, number, number]  // [x1, y1, x2, y2] in natural px
}

type Corner = 'tl' | 'tr' | 'br' | 'bl'

type DragState =
  | { type: 'corner'; regionId: number; corner: Corner }
  | { type: 'draw'; startX: number; startY: number; curX: number; curY: number }
  | null

interface NaturalSize { w: number; h: number }

// ── Helpers ────────────────────────────────────────────────────────────────────

const COLOURS = ['#f59e0b', '#16a34a', '#3b82f6', '#dc2626', '#9333ea']
const colour = (idx: number) => COLOURS[idx % COLOURS.length]

/** Convert a client-space point to SVG viewport coordinates. */
function clientToSvg(
  svg: SVGSVGElement,
  clientX: number,
  clientY: number,
): { x: number; y: number } {
  const pt = svg.createSVGPoint()
  pt.x = clientX
  pt.y = clientY
  const inv = svg.getScreenCTM()?.inverse()
  if (!inv) return { x: clientX, y: clientY }
  const r = pt.matrixTransform(inv)
  return { x: r.x, y: r.y }
}

// ── Component ──────────────────────────────────────────────────────────────────

interface PhotoSplitViewProps {
  job: Job
  onStarted: () => void
}

export default function PhotoSplitView({ job, onStarted }: PhotoSplitViewProps) {
  const imgRef = useRef<HTMLImageElement>(null)
  const svgRef  = useRef<SVGSVGElement>(null)

  const [size, setSize] = useState<NaturalSize | null>(null)
  const [regions, setRegions] = useState<Region[]>([])
  const [mode, setMode] = useState<'edit' | 'draw'>('edit')
  const [drag, setDrag] = useState<DragState>(null)
  const [nextId, setNextId] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  const imageUrl      = job.debug_urls?.['04_photo_boundaries']
  const detectionsUrl = job.debug_urls?.['05_photo_detections_json']

  // Load detection JSON on mount
  useEffect(() => {
    if (!detectionsUrl) return
    fetch(detectionsUrl)
      .then((r) => r.json())
      .then((data: { detections?: Array<{ bbox: number[] }> }) => {
        const list = (data.detections ?? []).map((d, i) => ({
          id: i,
          bbox: [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]] as [number, number, number, number],
        }))
        setRegions(list)
        setNextId(list.length)
      })
      .catch(() => { /* no JSON yet — start empty */ })
  }, [detectionsUrl])

  // Capture natural image dimensions once loaded
  function handleImageLoad(e: React.SyntheticEvent<HTMLImageElement>) {
    const img = e.currentTarget
    setSize({ w: img.naturalWidth, h: img.naturalHeight })
  }

  // ── Pointer helpers ──────────────────────────────────────────────────────────

  function getSvgPt(e: React.MouseEvent) {
    if (!svgRef.current) return null
    return clientToSvg(svgRef.current, e.clientX, e.clientY)
  }

  // ── SVG-level events (draw mode only) ────────────────────────────────────────

  function handleSvgMouseDown(e: React.MouseEvent) {
    if (mode !== 'draw') return
    e.preventDefault()
    const pt = getSvgPt(e)
    if (!pt) return
    setDrag({ type: 'draw', startX: pt.x, startY: pt.y, curX: pt.x, curY: pt.y })
  }

  function handleSvgMouseMove(e: React.MouseEvent) {
    if (!drag) return
    const pt = getSvgPt(e)
    if (!pt) return

    if (drag.type === 'draw') {
      setDrag((d) => d && d.type === 'draw' ? { ...d, curX: pt.x, curY: pt.y } : d)
    } else if (drag.type === 'corner') {
      const { regionId, corner } = drag
      setRegions((rs) =>
        rs.map((r) => {
          if (r.id !== regionId) return r
          let [x1, y1, x2, y2] = r.bbox
          const MIN = 30  // minimum region size in natural px
          if (corner === 'tl')      { x1 = Math.min(pt.x, x2 - MIN); y1 = Math.min(pt.y, y2 - MIN) }
          else if (corner === 'tr') { x2 = Math.max(pt.x, x1 + MIN); y1 = Math.min(pt.y, y2 - MIN) }
          else if (corner === 'br') { x2 = Math.max(pt.x, x1 + MIN); y2 = Math.max(pt.y, y1 + MIN) }
          else                      { x1 = Math.min(pt.x, x2 - MIN); y2 = Math.max(pt.y, y1 + MIN) }
          return { ...r, bbox: [x1, y1, x2, y2] as [number, number, number, number] }
        }),
      )
    }
  }

  function handleSvgMouseUp(e: React.MouseEvent) {
    if (!drag) return
    if (drag.type === 'draw') {
      const pt = getSvgPt(e)
      if (pt) {
        const x1 = Math.min(drag.startX, pt.x)
        const y1 = Math.min(drag.startY, pt.y)
        const x2 = Math.max(drag.startX, pt.x)
        const y2 = Math.max(drag.startY, pt.y)
        if (x2 - x1 > 30 && y2 - y1 > 30) {
          const id = nextId
          setRegions((rs) => [...rs, { id, bbox: [x1, y1, x2, y2] }])
          setNextId((n) => n + 1)
        }
      }
      setMode('edit')
    }
    setDrag(null)
  }

  // ── Corner handle events (edit mode only) ────────────────────────────────────

  function handleCornerDown(e: React.MouseEvent, regionId: number, corner: Corner) {
    e.preventDefault()
    e.stopPropagation()
    setDrag({ type: 'corner', regionId, corner })
  }

  // ── Delete ────────────────────────────────────────────────────────────────────

  function deleteRegion(id: number) {
    setRegions((rs) => rs.filter((r) => r.id !== id))
  }

  // ── Apply ─────────────────────────────────────────────────────────────────────

  async function handleApply() {
    setSubmitting(true)
    setSubmitError(null)
    try {
      await reprocessJob(job.job_id, {
        from_step: 'photo_detect',
        config: {
          forced_detections: regions.map((r) => ({
            bbox: r.bbox,
            confidence: 1.0,
            region_type: 'photo',
            orientation: 'unknown',
          })),
        },
      })
      onStarted()
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'Failed to start reprocessing')
    } finally {
      setSubmitting(false)
    }
  }

  // ── Draw preview rect ─────────────────────────────────────────────────────────

  const preview = drag?.type === 'draw' ? {
    x: Math.min(drag.startX, drag.curX),
    y: Math.min(drag.startY, drag.curY),
    w: Math.abs(drag.curX - drag.startX),
    h: Math.abs(drag.curY - drag.startY),
  } : null

  // Handle size in natural image pixels — scales with the image
  const H = size ? Math.round(size.w * 0.012) : 14   // ~1.2% of image width

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div className="flex-1 flex flex-col overflow-hidden">

      {/* Canvas */}
      <div className="flex-1 flex justify-center items-start p-6 overflow-auto min-h-0 bg-sa-stone-900">
        {imageUrl ? (
          <div className="relative" style={{ display: 'inline-block', lineHeight: 0 }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              ref={imgRef}
              src={imageUrl}
              alt="Photo boundaries"
              className="max-w-full h-auto block rounded-xl shadow-lg"
              onLoad={handleImageLoad}
              draggable={false}
              style={{ userSelect: 'none' }}
            />

            {size && (
              <svg
                ref={svgRef}
                className="absolute inset-0 w-full h-full rounded-xl"
                viewBox={`0 0 ${size.w} ${size.h}`}
                preserveAspectRatio="none"
                style={{ cursor: mode === 'draw' ? 'crosshair' : 'default' }}
                onMouseDown={handleSvgMouseDown}
                onMouseMove={handleSvgMouseMove}
                onMouseUp={handleSvgMouseUp}
                onMouseLeave={() => { if (drag?.type === 'draw') { setDrag(null); setMode('edit') } }}
              >
                {/* Regions — pointer-events disabled during draw mode */}
                <g style={{ pointerEvents: mode === 'draw' ? 'none' : 'auto' }}>
                  {regions.map((r, idx) => {
                    const [x1, y1, x2, y2] = r.bbox
                    const c = colour(idx)
                    const corners: Array<[Corner, number, number]> = [
                      ['tl', x1, y1], ['tr', x2, y1],
                      ['br', x2, y2], ['bl', x1, y2],
                    ]
                    return (
                      <g key={r.id}>
                        {/* Border */}
                        <rect
                          x={x1} y={y1} width={x2 - x1} height={y2 - y1}
                          fill="none" stroke={c} strokeWidth={3}
                          style={{ pointerEvents: 'none' }}
                        />
                        {/* Label */}
                        <text
                          x={x1 + H * 0.6} y={y1 + H * 1.8}
                          fontSize={H * 1.5} fontWeight="600"
                          fontFamily="var(--font-dm-sans)"
                          fill="white" stroke={c} strokeWidth={0.4}
                          style={{ pointerEvents: 'none' }}
                        >
                          Photo {idx + 1}
                        </text>
                        {/* Delete button (top-right of bbox) */}
                        <g
                          style={{ cursor: 'pointer' }}
                          onClick={() => deleteRegion(r.id)}
                        >
                          <circle cx={x2} cy={y1} r={H} fill={c} />
                          <text
                            x={x2} y={y1} fontSize={H * 1.4}
                            textAnchor="middle" dominantBaseline="central"
                            fill="white" style={{ pointerEvents: 'none' }}
                          >×</text>
                        </g>
                        {/* Corner handles */}
                        {corners.map(([corner, cx, cy]) => (
                          <circle
                            key={corner}
                            cx={cx} cy={cy} r={H}
                            fill={c} stroke="white" strokeWidth={2.5}
                            style={{
                              cursor: mode === 'edit' ? 'grab' : 'default',
                              filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.5))',
                            }}
                            onMouseDown={(e) => mode === 'edit' && handleCornerDown(e, r.id, corner)}
                          />
                        ))}
                      </g>
                    )
                  })}
                </g>

                {/* Draw preview */}
                {preview && preview.w > 5 && preview.h > 5 && (
                  <rect
                    x={preview.x} y={preview.y} width={preview.w} height={preview.h}
                    fill="rgba(245,158,11,0.12)" stroke="#f59e0b"
                    strokeWidth={3} strokeDasharray="12 6"
                    style={{ pointerEvents: 'none' }}
                  />
                )}
              </svg>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center w-full h-full text-sa-stone-500 text-sm">
            No photo boundaries image available
          </div>
        )}
      </div>

      {/* Footer controls */}
      <div className="flex-shrink-0 border-t border-sa-amber-200 dark:border-sa-stone-700 bg-sa-amber-50 dark:bg-sa-stone-900 px-6 py-4">
        <div className="flex items-center gap-3 flex-wrap">
          {/* Region count */}
          <span className="text-[12px] text-sa-stone-500 dark:text-sa-stone-400 flex-shrink-0">
            {regions.length} region{regions.length !== 1 ? 's' : ''}
          </span>

          {/* Toggle draw mode */}
          <button
            onClick={() => setMode((m) => m === 'draw' ? 'edit' : 'draw')}
            className={[
              'px-3 py-1.5 rounded-lg text-[12px] font-medium transition-colors duration-[200ms] flex-shrink-0',
              mode === 'draw'
                ? 'bg-sa-amber-500 text-white'
                : 'bg-white dark:bg-sa-stone-800 border border-sa-stone-200 dark:border-sa-stone-700 text-sa-stone-600 dark:text-sa-stone-300 hover:bg-sa-stone-100 dark:hover:bg-sa-stone-700',
            ].join(' ')}
          >
            {mode === 'draw' ? '✕  Cancel Draw' : '+ Add Region'}
          </button>

          <p className="text-[11px] text-sa-stone-400 dark:text-sa-stone-500 flex-shrink-0">
            {mode === 'draw'
              ? 'Click and drag to draw a new region'
              : 'Drag corner handles to resize · × to delete'}
          </p>

          <div className="flex-1" />

          {submitError && (
            <span className="text-[11px] text-sa-error flex-shrink-0">{submitError}</span>
          )}

          <button
            onClick={handleApply}
            disabled={submitting || regions.length === 0}
            className="px-4 py-1.5 rounded-lg text-[12px] font-semibold text-white bg-sa-amber-500 hover:bg-sa-amber-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors duration-[200ms] flex-shrink-0"
          >
            {submitting ? 'Starting…' : 'Confirm & Re-run'}
          </button>
        </div>
      </div>
    </div>
  )
}
