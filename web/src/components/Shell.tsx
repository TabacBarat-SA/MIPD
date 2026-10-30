import { AnimatePresence, motion, useMotionValueEvent, useScroll, useSpring } from 'motion/react'
import { BookOpen, CircleDollarSign, FlaskConical, LayoutGrid, Menu, Moon, RefreshCw, SlidersHorizontal, Sun, Target, UserRound, Users, X, Workflow } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { useApp } from '../lib/app'
import type { Source } from '../lib/api'
import { EASE, SPRING_SNAPPY } from '../lib/motion'

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

function NavList({ onNavigate, mobile }: { onNavigate?: () => void; mobile?: boolean }) {
  return (
    <nav aria-label="Principal" className={mobile ? 'flex flex-col gap-1' : 'flex items-center gap-0.5'}>
      {NAV.map((n, i) => (
        <motion.div key={n.to} initial={mobile ? { opacity: 0, x: 24 } : false} animate={{ opacity: 1, x: 0 }} transition={{ delay: mobile ? 0.08 + i * 0.05 : 0, duration: 0.5, ease: EASE }}>
          <NavLink to={n.to} end={n.to === '/'} onClick={onNavigate}
            className={({ isActive }) => `relative flex items-center gap-3 rounded-full font-medium transition-colors ${mobile ? 'px-4 py-3 text-2xl' : 'px-3.5 py-1.5 text-[0.9rem]'} ${isActive ? 'text-text' : 'text-muted hover:text-text'}`}>
            {({ isActive }) => (
              <>
                {isActive && <motion.span layoutId={mobile ? 'nav-pill-m' : 'nav-pill'} className="absolute inset-0 rounded-full bg-text/10 shadow-[inset_0_0_0_1px_color-mix(in_oklab,var(--text)_10%,transparent)]" transition={SPRING_SNAPPY} />}
                {mobile && <n.icon className={`relative size-6 ${isActive ? 'text-accent' : ''}`} aria-hidden />}
                <span className="relative">{n.label}</span>
              </>
            )}
          </NavLink>
        </motion.div>
      ))}
    </nav>
  )
}

function SettingsPopover() {
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (e: PointerEvent) => { if (!box.current?.contains(e.target as Node)) setOpen(false) }
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('pointerdown', close); document.addEventListener('keydown', key)
    return () => { document.removeEventListener('pointerdown', close); document.removeEventListener('keydown', key) }
  }, [open])
  return (
    <div ref={box} className="relative">
      <button className="grid size-9 place-items-center rounded-full text-muted transition-colors hover:bg-text/10 hover:text-text" onClick={() => setOpen(o => !o)} aria-expanded={open} aria-label="Datos y preferencias">
        <SlidersHorizontal className="size-[18px]" aria-hidden />
      </button>
      <AnimatePresence>
        {open && (
          <motion.div className="glass absolute right-0 top-[calc(100%+12px)] z-50 w-[320px] origin-top-right rounded-3xl p-5"
            initial={{ opacity: 0, scale: 0.94, y: -8, filter: 'blur(6px)' }} animate={{ opacity: 1, scale: 1, y: 0, filter: 'blur(0px)' }} exit={{ opacity: 0, scale: 0.96, y: -4 }} transition={{ duration: 0.28, ease: EASE }}>
            <Controls />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function Shell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const { pathname } = useLocation()
  const { theme, toggleTheme } = useApp()
  const { scrollY, scrollYProgress } = useScroll()
  const progress = useSpring(scrollYProgress, { stiffness: 140, damping: 30, mass: 0.4 })
  useMotionValueEvent(scrollY, 'change', v => setScrolled(v > 24))
  useEffect(() => { setOpen(false); window.scrollTo({ top: 0 }) }, [pathname])

  return (
    <div className="min-h-dvh">
      <motion.div className="read-progress" style={{ scaleX: progress }} aria-hidden />
      <header className={`fixed inset-x-0 top-0 z-40 flex justify-center px-3 transition-[padding] duration-500 ${scrolled ? 'pt-2' : 'pt-4'}`}>
        <div className={`glass flex w-full max-w-[1180px] items-center justify-between gap-4 rounded-full pl-4 pr-2 transition-all duration-500 ${scrolled ? 'h-12' : 'h-14'}`}>
          <NavLink to="/" className="flex items-center gap-2.5" aria-label="360 Analytics, inicio"><Logo size={26} /><span className="font-display text-[1.05rem] font-semibold tracking-tight">360 Analytics</span></NavLink>
          <div className="hidden lg:block"><NavList /></div>
          <div className="flex items-center gap-1">
            <button className="grid size-9 place-items-center rounded-full text-muted transition-colors hover:bg-text/10 hover:text-text" onClick={toggleTheme} aria-label={theme === 'dark' ? 'Cambiar a tema claro' : 'Cambiar a tema oscuro'}>
              {theme === 'dark' ? <Sun className="size-[18px]" aria-hidden /> : <Moon className="size-[18px]" aria-hidden />}
            </button>
            <SettingsPopover />
            <button className="grid size-9 place-items-center rounded-full text-muted transition-colors hover:bg-text/10 hover:text-text lg:hidden" onClick={() => setOpen(true)} aria-label="Abrir menú" aria-expanded={open}><Menu className="size-5" aria-hidden /></button>
          </div>
        </div>
      </header>

      <AnimatePresence>
        {open && (
          <motion.div className="glass fixed inset-0 z-50 flex flex-col gap-8 overflow-y-auto !rounded-none p-6 lg:hidden" role="dialog" aria-label="Menú"
            initial={{ opacity: 0, filter: 'blur(12px)' }} animate={{ opacity: 1, filter: 'blur(0px)' }} exit={{ opacity: 0 }} transition={{ duration: 0.35, ease: EASE }}>
            <button className="grid size-10 self-end place-items-center rounded-full bg-text/10" onClick={() => setOpen(false)} aria-label="Cerrar menú"><X className="size-5" aria-hidden /></button>
            <NavList mobile onNavigate={() => setOpen(false)} />
            <div className="mt-auto"><Controls /></div>
          </motion.div>
        )}
      </AnimatePresence>

      <main className="relative min-w-0 overflow-x-clip">
        <div className="aurora pointer-events-none absolute inset-x-0 top-0 h-[760px]" aria-hidden />
        <motion.div key={pathname} className="relative mx-auto max-w-[1180px] px-4 pb-32 pt-28 sm:px-8 lg:pt-32"
          initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, ease: EASE }}>
          {children}
        </motion.div>
      </main>
    </div>
  )
}
