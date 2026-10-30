import { AnimatePresence, motion } from 'motion/react'
import { AlertTriangle, ChevronDown, Info } from 'lucide-react'
import { useState, type PointerEvent, type ReactNode } from 'react'
import { useApp } from '../lib/app'
import { EASE, SPRING_SNAPPY } from '../lib/motion'
import { Reveal, TextReveal } from './motion'

export function PageHeader({ title, lead }: { title: string; lead: ReactNode }) {
  return (
    <header className="mb-14 max-w-4xl pt-4">
      <motion.span className="mb-6 block h-1 w-14 origin-left rounded-full bg-gradient-to-r from-[var(--g1)] via-[var(--g2)] to-[var(--g4)]"
        initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} transition={{ duration: 0.9, ease: EASE }} aria-hidden />
      <TextReveal text={title} className="display !text-[clamp(2.5rem,6vw,5rem)]" />
      <motion.p className="mt-6 max-w-[52ch] text-xl leading-snug text-muted sm:text-2xl" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.45, duration: 0.9, ease: EASE }}>{lead}</motion.p>
    </header>
  )
}

export function ScenarioNotice() {
  const { source } = useApp()
  return (
    <div className="mb-8 flex items-start gap-3 glass rounded-2xl px-4 py-3 text-sm text-muted">
      <Info className="mt-0.5 size-4 shrink-0 text-accent" aria-hidden />
      {source === 'synthetic'
        ? <p>Modo demostración: incluye compras artificiales identificadas. Sirve para aprender y probar decisiones, no para acreditar resultados reales de Olist.</p>
        : <p>Datos originales sin compras añadidas. Esta muestra tiene poca recompra: las comparaciones con pocos casos no permiten conclusiones fiables.</p>}
    </div>
  )
}

export function Caveat({ children, tone = 'note' }: { children: ReactNode; tone?: 'note' | 'warn' }) {
  const warn = tone === 'warn'
  return (
    <div className={`flex items-start gap-3 rounded-2xl border px-4 py-3 text-sm ${warn ? 'border-warn/40 bg-warn/10 text-text' : 'glass text-muted'}`}>
      {warn && <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warn" aria-hidden />}
      <p className="max-w-[75ch]">{children}</p>
    </div>
  )
}

export function Stat({ label, value, hint, color }: { label: string; value: ReactNode; hint?: string; color?: string }) {
  return (
    <div className="min-w-0 border-l-2 pl-4" style={{ borderColor: color ?? 'var(--line)' }} title={hint}>
      <div className="font-display text-3xl font-bold leading-none tracking-tight sm:text-4xl">{value}</div>
      <div className="mt-2 text-sm text-muted">{label}</div>
      {hint && <div className="mt-1 text-xs text-muted/80">{hint}</div>}
    </div>
  )
}

/** Mueve el foco luminoso del panel hacia el cursor. */
export const trackPointer = (e: PointerEvent<HTMLElement>) => {
  const r = e.currentTarget.getBoundingClientRect()
  e.currentTarget.style.setProperty('--mx', `${e.clientX - r.left}px`)
  e.currentTarget.style.setProperty('--my', `${e.clientY - r.top}px`)
}

export function Panel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <Reveal y={36} className="h-full">
      <section onPointerMove={trackPointer} className={`spot bento h-full p-6 sm:p-9 ${className}`}>
        {children}
      </section>
    </Reveal>
  )
}

export function Segmented<T extends string>({ value, onChange, options, label, full }: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[]; label: string; full?: boolean }) {
  return (
    <div role="radiogroup" aria-label={label} className={`relative glass rounded-full p-1 ${full ? 'flex w-full' : 'inline-flex max-w-full flex-wrap'}`}>
      {options.map(o => {
        const on = o.value === value
        return (
          <button key={o.value} role="radio" aria-checked={on} onClick={() => onChange(o.value)}
            className={`relative z-10 rounded-full px-4 py-1.5 text-sm font-medium ${full ? 'flex-1 !px-2' : ''} transition-colors ${on ? 'text-accent-ink' : 'text-muted hover:text-text'}`}>
            {on && <motion.span layoutId={`seg-${label}`} className="absolute inset-0 -z-10 rounded-full bg-accent" transition={SPRING_SNAPPY} />}
            {o.label}
          </button>
        )
      })}
    </div>
  )
}

export function Disclosure({ title, children, defaultOpen = false }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="bento !rounded-3xl">
      <button onClick={() => setOpen(o => !o)} aria-expanded={open} className="flex w-full items-center justify-between gap-4 px-5 py-4 text-left font-medium">
        {title}
        <ChevronDown className={`size-5 shrink-0 text-muted transition-transform duration-300 ${open ? 'rotate-180' : ''}`} aria-hidden />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }} className="overflow-hidden">
            <div className="px-5 pb-5">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export interface Column { key: string; label: string; align?: 'right'; render?: (v: any, row: any) => ReactNode }

export function DataTable({ columns, rows, maxHeight }: { columns: Column[]; rows: Record<string, any>[]; maxHeight?: string }) {
  return (
    <div className="glass overflow-auto rounded-3xl" style={{ maxHeight }}>
      <table className="w-full min-w-max border-collapse text-sm">
        <thead className="sticky top-0 bg-raised/95 backdrop-blur">
          <tr>{columns.map(c => <th key={c.key} className={`px-4 py-3 font-medium text-muted ${c.align === 'right' ? 'text-right' : 'text-left'}`}>{c.label}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-line/70 transition-colors hover:bg-raised/60">
              {columns.map(c => <td key={c.key} className={`num px-4 py-2.5 ${c.align === 'right' ? 'text-right' : ''}`}>{c.render ? c.render(r[c.key], r) : r[c.key]}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Loading({ label = 'Cargando los resultados guardados…' }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 py-24 text-muted" role="status">
      <span className="size-5 animate-spin rounded-full border-2 border-line border-t-accent" aria-hidden />
      {label}
    </div>
  )
}

export function ErrorState({ message }: { message: string }) {
  const { refresh } = useApp()
  return (
    <div className="max-w-xl rounded-3xl border border-warn/40 bg-warn/10 p-6" role="alert">
      <h2 className="font-display text-xl font-semibold">No se pudo completar la consulta</h2>
      <p className="mt-2 text-sm text-muted">{message}</p>
      <p className="mt-3 text-sm text-muted">Comprueba que los datos, pesos y métricas pertenecen a la misma ejecución. Si acabas de entrenar, ejecuta <code>evaluate</code> y <code>segment</code>, reinicia el servidor y vuelve a cargar.</p>
      <button className="btn mt-5" onClick={refresh}>Volver a intentar</button>
    </div>
  )
}

export const ChartTip = ({ active, payload, label, fmt }: { active?: boolean; payload?: any[]; label?: any; fmt?: (v: number, name: string) => string }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass rounded-xl px-3 py-2 text-sm">
      {label !== undefined && <div className="mb-1 font-medium">{label}</div>}
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2 text-muted">
          <span className="size-2 rounded-full" style={{ background: p.color || p.payload?.fill }} />
          <span>{p.name}</span>
          <span className="num ml-auto pl-4 text-text">{fmt ? fmt(p.value, p.name) : p.value}</span>
        </div>
      ))}
    </div>
  )
}
