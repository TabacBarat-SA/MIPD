import { useEffect, useMemo, useRef } from 'react'
import type { Point } from '../lib/api'
import { GROUP_COLORS, useApp } from '../lib/app'

export type HoverPoint = Point & { x: number; y: number }

interface Props {
  points: Point[]
  /** Grupo resaltado: el resto se atenua. */
  active?: number | null
  onHover?: (p: HoverPoint | null) => void
  orbit?: boolean
  /** Posicion horizontal del centro del mapa (0-1) en pantallas anchas. */
  anchor?: number
  className?: string
}

const quantile = (sorted: number[], q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]

/** Mapa de similitud (PCA) dibujado en canvas: cada punto es un cliente real de la muestra. */
export default function Constellation({ points, active = null, onHover, orbit = true, anchor = 0.5, className }: Props) {
  const { theme } = useApp()
  const ref = useRef<HTMLCanvasElement>(null)
  const activeRef = useRef(active)
  activeRef.current = active
  const hoverCb = useRef(onHover)
  hoverCb.current = onHover

  const norm = useMemo(() => {
    const xs = points.map(p => p.pca_1).sort((a, b) => a - b)
    const ys = points.map(p => p.pca_2).sort((a, b) => a - b)
    const [x0, x1, y0, y1] = [quantile(xs, 0.02), quantile(xs, 0.98), quantile(ys, 0.02), quantile(ys, 0.98)]
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, s = Math.max(x1 - x0, y1 - y0) / 2 || 1
    const maxClv = Math.max(...points.map(p => p.clv_pred), 1)
    return points.map((p, i) => ({
      p, i,
      x: Math.max(-1.25, Math.min(1.25, (p.pca_1 - cx) / s)),
      y: Math.max(-1.25, Math.min(1.25, (p.pca_2 - cy) / s)),
      r: 1.1 + 2.4 * Math.sqrt(Math.max(p.clv_pred, 0) / maxClv),
      seed: ((i * 9301 + 49297) % 233280) / 233280,
    }))
  }, [points])

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches
    const css = getComputedStyle(document.documentElement)
    const colors = GROUP_COLORS.map(c => css.getPropertyValue(c.slice(4, -1)).trim())
    const textColor = css.getPropertyValue('--text').trim()
    const dark = theme === 'dark'
    let w = 0, h = 0, raf = 0, hover = -1
    const mouse = { x: 0, y: 0, inside: false }
    const t0 = performance.now()
    let spin = 0, last = t0

    const resize = () => {
      const r = canvas.getBoundingClientRect()
      const dpr = Math.min(devicePixelRatio || 1, 2)
      w = r.width; h = r.height
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(canvas)

    const ease = (t: number) => 1 - Math.pow(1 - t, 3)
    const screen: { x: number; y: number }[] = new Array(norm.length)

    const frame = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000); last = now
      if (orbit && !reduced && hover < 0) spin += dt * 0.035
      const intro = reduced ? 1 : Math.min(1, (now - t0) / 2200)
      const wide = w >= 1024
      const scale = Math.min(wide ? w * Math.min(anchor, 1 - anchor) * 1.9 : w, h) * 0.4
      const cx = (wide ? w * anchor : w / 2) + (mouse.inside ? (mouse.x - w / 2) * -0.015 : 0)
      const cy = h / 2 + (mouse.inside ? (mouse.y - h / 2) * -0.015 : 0)
      const cos = Math.cos(spin), sin = Math.sin(spin)
      ctx.clearRect(0, 0, w, h)
      ctx.globalCompositeOperation = dark ? 'lighter' : 'source-over'
      const act = activeRef.current
      for (const n of norm) {
        // Cada punto llega en espiral desde el centro con un retardo propio.
        const k = ease(Math.max(0, Math.min(1, intro * 1.6 - n.seed * 0.6)))
        const ang = (1 - k) * (3 + n.seed * 4)
        const x0 = n.x * k, y0 = n.y * k
        const rx = x0 * Math.cos(ang) - y0 * Math.sin(ang), ry = x0 * Math.sin(ang) + y0 * Math.cos(ang)
        const sx = cx + (rx * cos - ry * sin) * scale, sy = cy + (rx * sin + ry * cos) * scale
        const dim = act !== null && act !== n.p.segment
        ctx.globalAlpha = (dim ? 0.1 : dark ? 0.8 : 0.85) * k
        ctx.fillStyle = colors[n.p.segment % colors.length]
        ctx.beginPath()
        ctx.arc(sx, sy, n.r * (n.i === hover ? 2.2 : 1) * (0.4 + 0.6 * k), 0, 6.2832)
        ctx.fill()
        screen[n.i] = { x: sx, y: sy }
      }
      ctx.globalAlpha = 1
      ctx.globalCompositeOperation = 'source-over'
      if (hover >= 0 && screen[hover]) {
        ctx.strokeStyle = textColor
        ctx.lineWidth = 1.5
        ctx.beginPath(); ctx.arc(screen[hover].x, screen[hover].y, 9, 0, 6.2832); ctx.stroke()
      }
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)

    const move = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect()
      mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top; mouse.inside = true
      let best = -1, bd = 14 * 14
      const act = activeRef.current
      for (let i = 0; i < screen.length; i++) {
        const s = screen[i]
        if (!s) continue
        const d = (s.x - mouse.x) ** 2 + (s.y - mouse.y) ** 2
        if (d < bd && (act === null || norm[i].p.segment === act)) { bd = d; best = i }
      }
      if (best !== hover) {
        hover = best
        hoverCb.current?.(best >= 0 ? { ...norm[best].p, x: screen[best].x, y: screen[best].y } : null)
      }
    }
    const leave = () => { mouse.inside = false; hover = -1; hoverCb.current?.(null) }
    canvas.addEventListener('pointermove', move)
    canvas.addEventListener('pointerleave', leave)
    return () => {
      cancelAnimationFrame(raf); ro.disconnect()
      canvas.removeEventListener('pointermove', move); canvas.removeEventListener('pointerleave', leave)
    }
  }, [norm, orbit, theme, anchor])

  return <canvas ref={ref} className={className} role="img" aria-label="Mapa de similitud entre clientes: cada punto es un cliente y el color indica su grupo." />
}
