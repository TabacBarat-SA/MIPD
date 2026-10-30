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
  /** Nombres de grupo: se dibujan sobre su centroide. */
  labels?: { segment: number; name: string }[]
  /** Si es true, la camara se acerca al grupo activo. */
  camera?: boolean
  className?: string
}

const quantile = (sorted: number[], q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]

/** Mapa de similitud (PCA) dibujado en canvas: cada punto es un cliente real de la muestra. */
export default function Constellation({ points, active = null, onHover, orbit = true, anchor = 0.5, labels, camera = false, className }: Props) {
  const { theme } = useApp()
  const ref = useRef<HTMLCanvasElement>(null)
  const activeRef = useRef(active)
  activeRef.current = active
  const hoverCb = useRef(onHover)
  hoverCb.current = onHover
  const labelsRef = useRef(labels)
  labelsRef.current = labels

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

  const centroids = useMemo(() => {
    const acc = new Map<number, { x: number; y: number; n: number }>()
    for (const n of norm) {
      const a = acc.get(n.p.segment) ?? { x: 0, y: 0, n: 0 }
      a.x += n.x; a.y += n.y; a.n++
      acc.set(n.p.segment, a)
    }
    return new Map([...acc].map(([k, a]) => [k, { x: a.x / a.n, y: a.y / a.n }]))
  }, [norm])

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
    let visible = true, camX = 0, camY = 0, zoom = 1
    const alpha = new Float32Array(norm.length).fill(-1)
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting })
    io.observe(canvas)

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
      raf = requestAnimationFrame(frame)
      if (!visible || document.hidden) { last = now; return }
      const dt = Math.min(0.05, (now - last) / 1000); last = now
      if (orbit && !reduced && hover < 0) spin += dt * 0.035
      const intro = reduced ? 1 : Math.min(1, (now - t0) / 2200)
      const wide = w >= 1024
      const scale = Math.min(wide ? w * Math.min(anchor, 1 - anchor) * 1.9 : w, h) * 0.4
      const px = mouse.inside ? (mouse.x - w / 2) / w : 0, py = mouse.inside ? (mouse.y - h / 2) / h : 0
      const cx = wide ? w * anchor : w / 2
      const cy = h / 2
      const cos = Math.cos(spin), sin = Math.sin(spin)
      const act = activeRef.current

      // Camara: se acerca con suavidad al centroide del grupo activo y vuelve al plano general.
      const cen = camera && act !== null ? centroids.get(act) : undefined
      const tz = cen ? 1.7 : 1
      const tx = cen ? cen.x * cos - cen.y * sin : 0, ty = cen ? cen.x * sin + cen.y * cos : 0
      const k = reduced ? 1 : 1 - Math.exp(-dt * 4.5)
      zoom += (tz - zoom) * k; camX += (tx - camX) * k; camY += (ty - camY) * k

      ctx.clearRect(0, 0, w, h)
      ctx.globalCompositeOperation = dark ? 'lighter' : 'source-over'
      const fade = reduced ? 1 : 1 - Math.exp(-dt * 9)
      for (const n of norm) {
        // Cada punto llega en espiral desde el centro con un retardo propio.
        const e = ease(Math.max(0, Math.min(1, intro * 1.6 - n.seed * 0.6)))
        const ang = (1 - e) * (3 + n.seed * 4)
        const x0 = n.x * e, y0 = n.y * e
        const rx = x0 * Math.cos(ang) - y0 * Math.sin(ang), ry = x0 * Math.sin(ang) + y0 * Math.cos(ang)
        const z = n.seed - 0.5 // profundidad: los puntos cercanos se mueven mas con el cursor
        const sx = cx + ((rx * cos - ry * sin - camX) * zoom) * scale - px * z * 46
        const sy = cy + ((rx * sin + ry * cos - camY) * zoom) * scale - py * z * 46
        const dim = act !== null && act !== n.p.segment
        const target = dim ? 0.08 : dark ? 0.55 + 0.35 * n.seed : 0.6 + 0.3 * n.seed
        alpha[n.i] = alpha[n.i] < 0 ? target : alpha[n.i] + (target - alpha[n.i]) * fade
        ctx.globalAlpha = alpha[n.i] * e
        ctx.fillStyle = colors[n.p.segment % colors.length]
        ctx.beginPath()
        ctx.arc(sx, sy, n.r * (0.8 + 0.5 * n.seed) * Math.sqrt(zoom) * (n.i === hover ? 2.2 : 1) * (0.4 + 0.6 * e), 0, 6.2832)
        ctx.fill()
        screen[n.i] = { x: sx, y: sy }
      }
      ctx.globalAlpha = 1
      ctx.globalCompositeOperation = 'source-over'

      // Etiquetas sobre el centroide de cada grupo, una vez terminada la entrada.
      const labs = labelsRef.current
      if (labs && intro >= 1) {
        ctx.font = '600 13px "Inter Variable", system-ui, sans-serif'
        ctx.textAlign = 'center'
        for (const l of labs) {
          const c = centroids.get(l.segment)
          if (!c || (act !== null && act !== l.segment)) continue
          const lx = cx + (((c.x * cos - c.y * sin) - camX) * zoom) * scale - px * 0
          const ly = cy + (((c.x * sin + c.y * cos) - camY) * zoom) * scale
          const col = colors[l.segment % colors.length]
          const tw = ctx.measureText(l.name).width + 22
          ctx.globalAlpha = act === null ? 0.9 : 1
          ctx.fillStyle = dark ? 'rgba(10,10,14,0.72)' : 'rgba(255,255,255,0.82)'
          ctx.beginPath(); ctx.roundRect(lx - tw / 2, ly - 14, tw, 28, 14); ctx.fill()
          ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.stroke()
          ctx.fillStyle = textColor
          ctx.fillText(l.name, lx, ly + 4.5)
        }
        ctx.globalAlpha = 1
      }

      if (hover >= 0 && screen[hover]) {
        const hp = screen[hover], seg = norm[hover].p.segment, col = colors[seg % colors.length]
        // Vecinos del mismo grupo: hilos finos hacia los puntos mas cercanos.
        const near: { d: number; i: number }[] = []
        for (let i = 0; i < screen.length; i++) {
          const s = screen[i]
          if (!s || i === hover || norm[i].p.segment !== seg) continue
          near.push({ d: (s.x - hp.x) ** 2 + (s.y - hp.y) ** 2, i })
        }
        near.sort((a, b) => a.d - b.d)
        ctx.strokeStyle = col; ctx.lineWidth = 0.8
        for (const { i } of near.slice(0, 8)) {
          ctx.globalAlpha = 0.45
          ctx.beginPath(); ctx.moveTo(hp.x, hp.y); ctx.lineTo(screen[i].x, screen[i].y); ctx.stroke()
        }
        ctx.globalAlpha = 1
        const glow = ctx.createRadialGradient(hp.x, hp.y, 0, hp.x, hp.y, 46)
        glow.addColorStop(0, col + '66'); glow.addColorStop(1, col + '00')
        ctx.fillStyle = glow
        ctx.beginPath(); ctx.arc(hp.x, hp.y, 46, 0, 6.2832); ctx.fill()
        ctx.strokeStyle = textColor
        ctx.lineWidth = 1.5
        ctx.beginPath(); ctx.arc(hp.x, hp.y, 9, 0, 6.2832); ctx.stroke()
      }
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
      cancelAnimationFrame(raf); ro.disconnect(); io.disconnect()
      canvas.removeEventListener('pointermove', move); canvas.removeEventListener('pointerleave', leave)
    }
  }, [norm, centroids, orbit, theme, anchor, camera])

  return <canvas ref={ref} className={className} role="img" aria-label="Mapa de similitud entre clientes: cada punto es un cliente y el color indica su grupo." />
}
