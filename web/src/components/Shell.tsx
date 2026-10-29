import { AnimatePresence, motion } from 'motion/react'
import { BookOpen, CircleDollarSign, FlaskConical, LayoutGrid, Menu, Moon, RefreshCw, Sun, Target, UserRound, Users, X, Workflow } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { useApp } from '../lib/app'
import type { Source } from '../lib/api'

const NAV = [
  { to: '/', label: 'Resumen', icon: LayoutGrid },
  { to: '/grupos', label: 'Grupos', icon: Users },
  { to: '/cliente', label: 'Cliente', icon: UserRound },
  { to: '/campana', label: 'Campaña', icon: Target },
  { to: '/resultados', label: 'Resultados', icon: FlaskConical },
  { to: '/como-funciona', label: 'Cómo funciona', icon: Workflow },
  { to: '/guia', label: 'Guía de uso', icon: BookOpen },
]

export function Logo({ size = 34 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" aria-hidden>
      <circle cx="20" cy="20" r="17" stroke="var(--line)" strokeWidth="2" />
      <motion.circle cx="20" cy="20" r="17" stroke="var(--accent)" strokeWidth="3" strokeLinecap="round" strokeDasharray="80 107"
        initial={{ rotate: 0 }} animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 14, ease: 'linear' }} style={{ originX: '20px', originY: '20px' }} />
      <circle cx="20" cy="20" r="5" fill="var(--accent)" />
    </svg>
  )
}

function Controls() {
  const { source, setSource, theme, toggleTheme, currency, setCurrency, canConvert, refresh } = useApp()
  const sources: { v: Source; l: string; d: string }[] = [
    { v: 'synthetic', l: 'Demostración', d: 'Con compras simuladas' },
    { v: 'original', l: 'Originales', d: 'Misma muestra, sin añadidos' },
  ]
  return (
    <div className="space-y-5 text-sm">
      <fieldset>
        <legend className="mb-2 text-muted">Datos que exploras</legend>
        <div className="space-y-1.5">
          {sources.map(s => (
            <button key={s.v} onClick={() => setSource(s.v)} aria-pressed={source === s.v}
              className={`w-full rounded-xl border px-3 py-2 text-left transition-colors ${source === s.v ? 'border-accent bg-accent/10' : 'border-line hover:border-muted'}`}>
              <span className="block font-medium">{s.l}</span>
              <span className="block text-xs text-muted">{s.d}</span>
            </button>
          ))}
        </div>
      </fieldset>
      <div className="flex gap-2">
        <button className="btn btn-ghost flex-1 !px-3 !py-2" onClick={() => setCurrency(currency === 'USD' ? 'BRL' : 'USD')} disabled={!canConvert && currency === 'BRL'}
          title={canConvert ? 'Cambiar moneda' : 'Sin tipo de cambio disponible: se muestran reales'} aria-label={`Moneda: ${currency}. Cambiar moneda`}>
          <CircleDollarSign className="size-4" aria-hidden />{currency}
        </button>
        <button className="btn btn-ghost !px-3 !py-2" onClick={toggleTheme} aria-label={theme === 'dark' ? 'Cambiar a tema claro' : 'Cambiar a tema oscuro'}>
          {theme === 'dark' ? <Sun className="size-4" aria-hidden /> : <Moon className="size-4" aria-hidden />}
        </button>
        <button className="btn btn-ghost !px-3 !py-2" onClick={refresh} aria-label="Actualizar datos" title="Actualizar datos">
          <RefreshCw className="size-4" aria-hidden />
        </button>
      </div>
      <p className="text-xs leading-relaxed text-muted">
        Corte histórico 01/04/2018. Previsión a 3 meses. Importes originales en reales (BRL){currency === 'USD' ? ', convertidos a dólares con el tipo de cambio actual' : ''}. No envía mensajes ni activa anuncios.
      </p>
    </div>
  )
}

function NavList({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav aria-label="Principal" className="space-y-1">
      {NAV.map(n => (
        <NavLink key={n.to} to={n.to} end={n.to === '/'} onClick={onNavigate}
          className={({ isActive }) => `relative flex items-center gap-3 rounded-xl px-3 py-2.5 font-medium transition-colors ${isActive ? 'text-text' : 'text-muted hover:text-text'}`}>
          {({ isActive }) => (
            <>
              {isActive && <motion.span layoutId="nav-pill" className="absolute inset-0 rounded-xl bg-raised" transition={{ type: 'spring', stiffness: 500, damping: 40 }} />}
              <n.icon className={`relative size-[18px] ${isActive ? 'text-accent' : ''}`} aria-hidden />
              <span className="relative">{n.label}</span>
            </>
          )}
        </NavLink>
      ))}
    </nav>
  )
}

export default function Shell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false)
  const { pathname } = useLocation()
  useEffect(() => { setOpen(false); window.scrollTo({ top: 0 }) }, [pathname])

  return (
    <div className="min-h-dvh lg:grid lg:grid-cols-[264px_1fr]">
      <aside className="sticky top-0 hidden h-dvh flex-col gap-8 overflow-y-auto border-r border-line bg-surface/70 p-6 backdrop-blur lg:flex">
        <div className="flex items-center gap-3">
          <Logo />
          <div className="font-display text-xl font-semibold leading-none">360 Analytics<span className="mt-1 block text-xs font-normal tracking-normal text-muted">Decisiones de marketing</span></div>
        </div>
        <NavList />
        <div className="mt-auto"><Controls /></div>
      </aside>

      <header className="sticky top-0 z-30 flex items-center justify-between border-b border-line bg-ink/80 px-4 py-3 backdrop-blur lg:hidden">
        <div className="flex items-center gap-2.5"><Logo size={28} /><span className="font-display text-lg font-semibold">360 Analytics</span></div>
        <button className="btn btn-ghost !p-2.5" onClick={() => setOpen(true)} aria-label="Abrir menú" aria-expanded={open}><Menu className="size-5" aria-hidden /></button>
      </header>

      <AnimatePresence>
        {open && (
          <>
            <motion.div className="fixed inset-0 z-40 bg-black/60 lg:hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setOpen(false)} />
            <motion.aside className="fixed inset-y-0 right-0 z-50 flex w-[min(86vw,340px)] flex-col gap-6 overflow-y-auto border-l border-line bg-surface p-5 lg:hidden"
              initial={{ x: '100%' }} animate={{ x: 0 }} exit={{ x: '100%' }} transition={{ type: 'spring', stiffness: 420, damping: 42 }} role="dialog" aria-label="Menú">
              <button className="btn btn-ghost -mb-2 self-end !p-2.5" onClick={() => setOpen(false)} aria-label="Cerrar menú"><X className="size-5" aria-hidden /></button>
              <NavList onNavigate={() => setOpen(false)} />
              <div className="mt-auto"><Controls /></div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      <main className="relative min-w-0 overflow-hidden">
        <div className="grid-bg pointer-events-none absolute inset-x-0 top-0 h-[520px]" aria-hidden />
        <AnimatePresence mode="wait">
          <motion.div key={pathname} className="relative mx-auto max-w-[1180px] px-4 pb-24 pt-10 sm:px-8 lg:pt-14"
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}>
            {children}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
