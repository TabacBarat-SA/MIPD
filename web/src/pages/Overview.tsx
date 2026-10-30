import { motion, useScroll, useTransform } from 'motion/react'
import { ArrowRight } from 'lucide-react'
import { useCallback, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import Constellation, { type HoverPoint } from '../components/Constellation'
import CountUp from '../components/CountUp'
import { Magnetic, Reveal, TextReveal } from '../components/motion'
import { Caveat, ChartTip, Disclosure, ErrorState, Loading, Panel, ScenarioNotice, Stat, trackPointer } from '../components/ui'
import type { Overview } from '../lib/api'
import { groupColor, num, pct, useApp } from '../lib/app'
import { useScenarioApi } from '../lib/hooks'
import { EASE } from '../lib/motion'

const DECISIONS = [
  { to: '/grupos', title: '¿A quién dirigirme?', body: 'Consulta los grupos y prepara una audiencia ordenada por gasto previsto. La prioridad de contacto no es la probabilidad de respuesta.', cta: 'Ver los grupos' },
  { to: '/cliente', title: '¿Qué ofrecerle?', body: 'Abre la ficha de un cliente y revisa los productos y categorías sugeridos. Contrasta la selección con su historial.', cta: 'Abrir una ficha' },
  { to: '/resultados', title: '¿Cómo medirlo?', body: 'Define un indicador, compara con un grupo de control y revisa la evaluación. Una previsión no demuestra el efecto de una campaña.', cta: 'Comprobar resultados' },
]

export default function OverviewPage() {
  const { money } = useApp()
  const { data, error } = useScenarioApi<Overview>('overview')
  const [hover, setHover] = useState<HoverPoint | null>(null)
  const [focus, setFocus] = useState<number | null>(null)
  const onHover = useCallback((p: HoverPoint | null) => setHover(p), [])
  const hero = useRef<HTMLElement>(null)
  const { scrollYProgress } = useScroll({ target: hero, offset: ['start start', 'end start'] })
  const mapScale = useTransform(scrollYProgress, [0, 1], [1, 1.35])
  const mapOpacity = useTransform(scrollYProgress, [0, 0.9], [1, 0.15])
  const textY = useTransform(scrollYProgress, [0, 1], [0, -70])

  if (error) return <ErrorState message={error} />
  if (!data) return <Loading />

  const total = data.customers
  const groups = data.profiles.map(p => ({ ...p, share: p.customers / total }))

  return (
    <div>
      <section ref={hero} className="relative -mx-4 -mt-28 mb-20 sm:-mx-8 lg:-mt-32 lg:min-h-[100dvh]">
        <motion.div className="absolute inset-x-0 top-0 h-[460px] lg:inset-0 lg:h-auto" style={{ scale: mapScale, opacity: mapOpacity }}>
          <Constellation points={data.points} active={focus} onHover={onHover} anchor={0.76} className="size-full touch-pan-y" />
          {hover && (
            <div className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-[130%] rounded-xl border border-line bg-raised px-3 py-2 text-sm shadow-xl" style={{ left: hover.x, top: hover.y }}>
              <div className="flex items-center gap-2 font-medium"><span className="size-2 rounded-full" style={{ background: groupColor(hover.segment) }} />{data.profiles.find(p => p.segment === hover.segment)?.name}</div>
              <div className="num text-muted">Gasto previsto {money(hover.clv_pred)}</div>
            </div>
          )}
        </motion.div>
        <div className="pointer-events-none relative flex flex-col px-4 pt-[470px] sm:px-8 lg:min-h-[100dvh] lg:justify-center lg:pt-16">
          <motion.div className="max-w-2xl" style={{ y: textY }}>
            <motion.p className="eyebrow mb-5" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3, duration: 1 }}>360 Analytics</motion.p>
            <TextReveal text="Cada punto es un cliente." className="display" delay={0.35} />
            <motion.p className="mt-7 max-w-[44ch] text-xl leading-snug text-muted sm:text-2xl" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.9, duration: 0.9, ease: EASE }}>
              360 Analytics reúne su historial en un mapa y responde tres cosas: a quién prestar atención, qué ofrecer y cómo comprobar que la propuesta funciona.
            </motion.p>
            <motion.div className="pointer-events-auto mt-9 flex flex-wrap items-center gap-3" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1.1, duration: 0.9, ease: EASE }}>
              <Magnetic><Link to="/grupos" className="btn !px-6 !py-3.5 !text-base">Explorar los grupos <ArrowRight className="size-4" aria-hidden /></Link></Magnetic>
              <Magnetic><Link to="/campana" className="btn btn-ghost glass !px-6 !py-3.5 !text-base">Preparar una campaña</Link></Magnetic>
            </motion.div>
            <ul className="pointer-events-auto mt-10 flex flex-wrap gap-2" aria-label="Grupos de clientes: pasa el cursor para resaltarlos en el mapa">
              {groups.map(g => {
                const on = focus === g.segment
                return (
                  <li key={g.segment}>
                    <button type="button" aria-pressed={on}
                      onPointerEnter={() => setFocus(g.segment)} onPointerLeave={() => setFocus(null)}
                      onFocus={() => setFocus(g.segment)} onBlur={() => setFocus(null)}
                      className={`flex items-center gap-2 rounded-full border px-3.5 py-2 text-sm backdrop-blur transition-all ${on ? 'border-transparent text-text' : 'glass text-muted hover:text-text'}`}
                      style={on ? { background: `color-mix(in oklab, ${groupColor(g.segment)} 22%, var(--surface))`, boxShadow: `0 0 22px -6px ${groupColor(g.segment)}` } : undefined}>
                      <span className="size-2 rounded-full" style={{ background: groupColor(g.segment) }} />
                      {g.name}
                      <span className="num text-xs opacity-70">{pct(g.share, 0)}</span>
                    </button>
                  </li>
                )
              })}
            </ul>
          </motion.div>
        </div>
      </section>

      <ScenarioNotice />

      <Reveal className="bento grid gap-10 p-8 sm:grid-cols-2 sm:p-12 lg:grid-cols-4">
        <Stat label="Clientes analizados" value={<CountUp value={total} format={num} />} hint="Muestra del estudio, no todo Olist." color="var(--g1)" />
        <Stat label="Gasto previsto a 3 meses" value={<CountUp value={data.predicted_spend} format={n => money(n)} />} hint="Estimación, no ventas garantizadas." color="var(--g2)" />
        <Stat label="Productos disponibles" value={<CountUp value={data.products} format={num} />} hint="Catálogo reducido del recomendador." color="var(--g3)" />
        <Stat label="Grupos de clientes" value={<CountUp value={data.groups} format={num} />} hint="Según la representación de la red." color="var(--g4)" />
      </Reveal>

      <section className="mt-20">
        <Reveal><p className="eyebrow mb-3">Una herramienta</p><h2 className="headline max-w-3xl">Tres preguntas, <span className="grad-text">una herramienta.</span></h2></Reveal>
        <ul className="mt-12 grid gap-5 md:grid-cols-3">
          {DECISIONS.map((d, i) => (
            <li key={d.to}>
              <Reveal delay={i * 0.12} className="h-full">
                <Link to={d.to} onPointerMove={trackPointer} className="group spot bento flex h-full flex-col p-8">
                  <span className="num font-display text-5xl font-bold" style={{ color: `var(--g${i + 1})` }}>{i + 1}</span>
                  <h3 className="mt-6 text-3xl font-bold leading-tight">{d.title}</h3>
                  <p className="mt-3 flex-1 text-muted">{d.body}</p>
                  <span className="mt-8 inline-flex items-center gap-2 font-medium text-accent">{d.cta}<ArrowRight className="size-4 transition-transform group-hover:translate-x-1.5" aria-hidden /></span>
                </Link>
              </Reveal>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-20 grid gap-6 lg:grid-cols-2">
        <Panel>
          <h2 className="text-3xl font-bold">¿Cuánto gasto prevé el modelo por cliente?</h2>
          <p className="mt-1 text-sm text-muted">Número de clientes por tramo de gasto previsto a 3 meses.</p>
          <div className="mt-6 h-72">
            <ResponsiveContainer>
              <BarChart data={data.histogram.map(b => ({ label: money((b.from + b.to) / 2), count: b.count, from: b.from, to: b.to }))} margin={{ left: -12, right: 4 }}>
                <CartesianGrid vertical={false} />
                <XAxis dataKey="label" interval={Math.ceil(data.histogram.length / 5)} tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip cursor={{ fill: 'var(--raised)' }} content={<ChartTip fmt={v => num(v)} />} labelFormatter={undefined} />
                <Bar dataKey="count" name="Clientes" fill="var(--g1)" radius={[5, 5, 0, 0]} animationDuration={900} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-3 text-xs text-muted">Es el importe previsto por el modelo, no una distribución de ventas ya realizadas.</p>
        </Panel>
        <Panel>
          <h2 className="text-3xl font-bold">¿Cómo se reparte la muestra?</h2>
          <p className="mt-1 text-sm text-muted">Clientes por grupo.</p>
          <div className="mt-6 h-72">
            <ResponsiveContainer>
              <BarChart data={groups} layout="vertical" margin={{ left: 8, right: 16 }}>
                <CartesianGrid horizontal={false} />
                <XAxis type="number" tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="name" width={170} tickLine={false} axisLine={false} />
                <Tooltip cursor={{ fill: 'var(--raised)' }} content={<ChartTip fmt={(v) => `${num(v)} clientes`} />} />
                <Bar dataKey="customers" name="Clientes" radius={[0, 6, 6, 0]} animationDuration={900}>
                  {groups.map(g => <Cell key={g.segment} fill={groupColor(g.segment)} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-3 text-xs text-muted">Los nombres describen reglas comerciales sobre cada grupo. No son etiquetas reales de fidelidad ni de abandono.</p>
        </Panel>
      </section>

      <section className="mt-10 space-y-3">
        <Disclosure title="Qué datos estoy viendo y de dónde salen">
          <div className="max-w-[70ch] space-y-3 text-muted">
            <p>El experimento conserva {num(data.summary.audit.original_orders)} pedidos originales y añade {num(data.summary.audit.added_orders)} pedidos artificiales. La muestra tiene {num(total)} personas.</p>
            <p>En la ventana de evaluación, el {pct(data.summary.future_purchase_rate)} de la muestra realizó alguna compra. Es una tasa observada del escenario, no una probabilidad individual.</p>
            <p>Motor registrado en los pesos: {data.summary.backend}. Instalar PyTorch no cambia el origen de unos pesos ya guardados.</p>
          </div>
        </Disclosure>
        <Caveat>Cada punto del mapa se coloca según la similitud de su historial (proyección PCA). Los ejes no son euros ni riesgo, y la proyección pierde información.</Caveat>
      </section>
    </div>
  )
}
