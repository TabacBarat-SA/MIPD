import { motion } from 'motion/react'
import { ArrowRight } from 'lucide-react'
import { useCallback, useState } from 'react'
import { Link } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import Constellation, { type HoverPoint } from '../components/Constellation'
import CountUp from '../components/CountUp'
import { Caveat, ChartTip, Disclosure, ErrorState, Loading, Panel, ScenarioNotice, Stat } from '../components/ui'
import type { Overview } from '../lib/api'
import { groupColor, num, pct, useApp } from '../lib/app'
import { useScenarioApi } from '../lib/hooks'

const DECISIONS = [
  { to: '/grupos', title: '¿A quién dirigirme?', body: 'Consulta los grupos y prepara una audiencia ordenada por gasto previsto. La prioridad de contacto no es la probabilidad de respuesta.', cta: 'Ver los grupos' },
  { to: '/cliente', title: '¿Qué ofrecerle?', body: 'Abre la ficha de un cliente y revisa los productos y categorías sugeridos. Contrasta la selección con su historial.', cta: 'Abrir una ficha' },
  { to: '/resultados', title: '¿Cómo medirlo?', body: 'Define un indicador, compara con un grupo de control y revisa la evaluación. Una previsión no demuestra el efecto de una campaña.', cta: 'Comprobar resultados' },
]

export default function OverviewPage() {
  const { money } = useApp()
  const { data, error } = useScenarioApi<Overview>('overview')
  const [hover, setHover] = useState<HoverPoint | null>(null)
  const onHover = useCallback((p: HoverPoint | null) => setHover(p), [])

  if (error) return <ErrorState message={error} />
  if (!data) return <Loading />

  const total = data.customers
  const groups = data.profiles.map(p => ({ ...p, share: p.customers / total }))

  return (
    <div>
      <section className="relative -mx-4 mb-14 sm:-mx-8 lg:min-h-[620px]">
        <div className="absolute inset-x-0 top-0 h-[340px] lg:inset-0 lg:h-auto">
          <Constellation points={data.points} onHover={onHover} anchor={0.7} className="size-full touch-pan-y" />
          {hover && (
            <div className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-[130%] rounded-xl border border-line bg-raised px-3 py-2 text-sm shadow-xl" style={{ left: hover.x, top: hover.y }}>
              <div className="flex items-center gap-2 font-medium"><span className="size-2 rounded-full" style={{ background: groupColor(hover.segment) }} />{data.profiles.find(p => p.segment === hover.segment)?.name}</div>
              <div className="num text-muted">Gasto previsto {money(hover.clv_pred)}</div>
            </div>
          )}
        </div>
        <div className="pointer-events-none relative flex flex-col px-4 pt-[330px] sm:px-8 lg:min-h-[620px] lg:justify-center lg:pt-0">
          <motion.div className="max-w-xl" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}>
            <h1 className="text-5xl font-semibold leading-[0.98] sm:text-6xl lg:text-7xl">Cada punto es un cliente.</h1>
            <p className="mt-5 max-w-[46ch] text-lg text-muted">
              360 Analytics reúne su historial en un mapa y responde tres cosas: a quién prestar atención, qué ofrecer y cómo comprobar que la propuesta funciona.
            </p>
            <div className="pointer-events-auto mt-7 flex flex-wrap gap-3">
              <Link to="/grupos" className="btn">Explorar los grupos <ArrowRight className="size-4" aria-hidden /></Link>
              <Link to="/campana" className="btn btn-ghost">Preparar una campaña</Link>
            </div>
          </motion.div>
        </div>
      </section>

      <ScenarioNotice />

      <motion.div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.9 }}>
        <Stat label="Clientes analizados" value={<CountUp value={total} format={num} />} hint="Muestra del estudio, no todo Olist." color="var(--g1)" />
        <Stat label="Gasto previsto a 3 meses" value={<CountUp value={data.predicted_spend} format={n => money(n)} />} hint="Estimación, no ventas garantizadas." color="var(--g2)" />
        <Stat label="Productos disponibles" value={<CountUp value={data.products} format={num} />} hint="Catálogo reducido del recomendador." color="var(--g3)" />
        <Stat label="Grupos de clientes" value={<CountUp value={data.groups} format={num} />} hint="Según la representación de la red." color="var(--g4)" />
      </motion.div>

      <section className="mt-20">
        <h2 className="max-w-2xl text-3xl font-semibold sm:text-4xl">Tres preguntas, una herramienta</h2>
        <ul className="mt-8 divide-y divide-line border-y border-line">
          {DECISIONS.map(d => (
            <li key={d.to}>
              <Link to={d.to} className="group grid gap-3 py-7 transition-colors hover:bg-surface/60 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)_auto] sm:items-center sm:gap-10 sm:px-4">
                <h3 className="text-2xl font-semibold">{d.title}</h3>
                <p className="max-w-[56ch] text-muted">{d.body}</p>
                <span className="inline-flex items-center gap-2 font-medium text-accent">{d.cta}<ArrowRight className="size-4 transition-transform group-hover:translate-x-1" aria-hidden /></span>
              </Link>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-20 grid gap-6 lg:grid-cols-2">
        <Panel>
          <h2 className="text-2xl font-semibold">¿Cuánto gasto prevé el modelo por cliente?</h2>
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
          <h2 className="text-2xl font-semibold">¿Cómo se reparte la muestra?</h2>
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
