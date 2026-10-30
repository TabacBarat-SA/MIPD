import { AnimatePresence, motion } from 'motion/react'
import { useCallback, useState } from 'react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import Constellation, { type HoverPoint } from '../components/Constellation'
import { Caveat, ChartTip, DataTable, Disclosure, ErrorState, Loading, PageHeader, ScenarioNotice } from '../components/ui'
import type { Groups } from '../lib/api'
import { groupColor, num, pct, useApp } from '../lib/app'
import { useScenarioApi } from '../lib/hooks'

export default function GroupsPage() {
  const { money } = useApp()
  const { data, error } = useScenarioApi<Groups>('groups')
  const [sel, setSel] = useState<number | null>(null)
  const [hover, setHover] = useState<HoverPoint | null>(null)
  const onHover = useCallback((p: HoverPoint | null) => setHover(p), [])

  if (error) return <ErrorState message={error} />
  if (!data) return <Loading />

  const current = data.profiles.find(p => p.segment === sel) ?? null
  const spaceName = (s: string) => (s === 'embeddings' ? 'Representación de la red' : 'Historial clásico (RFM)')
  const ks = [...new Set(data.metrics.map(m => m.k))].sort((a, b) => a - b)
  const silhouette = ks.map(k => ({ k, ...Object.fromEntries(data.metrics.filter(m => m.k === k).map(m => [spaceName(m.space), m.silhouette_train])) }))

  return (
    <div>
      <PageHeader title="Entender los grupos" lead="Los clientes se agrupan por similitud. Usa estos perfiles para diseñar mensajes distintos y contrastarlos después con resultados de campaña." />
      <ScenarioNotice />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,320px)_minmax(0,1fr)]">
        <ul className="space-y-2" aria-label="Grupos de clientes">
          <li>
            <button onClick={() => setSel(null)} aria-pressed={sel === null}
              className={`w-full rounded-2xl border px-4 py-3.5 text-left transition-all ${sel === null ? 'border-accent bg-accent/10' : 'glass hover:-translate-y-0.5'}`}>
              <span className="font-medium">Todos los grupos</span>
              <span className="num block text-sm text-muted">{num(data.total)} clientes</span>
            </button>
          </li>
          {data.profiles.map(p => (
            <li key={p.segment}>
              <button onClick={() => setSel(p.segment)} aria-pressed={sel === p.segment} style={sel === p.segment ? { borderColor: groupColor(p.segment), boxShadow: `0 0 28px -8px ${groupColor(p.segment)}` } : undefined}
                className={`w-full rounded-2xl border px-4 py-3.5 text-left transition-all ${sel === p.segment ? 'bg-accent/10' : 'glass hover:-translate-y-0.5'}`}>
                <span className="flex items-center gap-2.5 font-medium"><span className="size-3 rounded-full" style={{ background: groupColor(p.segment) }} />{p.name}</span>
                <span className="num mt-0.5 block pl-[22px] text-sm text-muted">{num(p.customers)} clientes · {pct(p.customers / data.total)}</span>
                <span className="mt-2 block h-1 overflow-hidden rounded-full bg-line"><motion.span className="block h-full rounded-full" style={{ background: groupColor(p.segment) }} initial={{ width: 0 }} animate={{ width: `${(p.customers / data.total) * 100}%` }} transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }} /></span>
              </button>
            </li>
          ))}
        </ul>

        <div className="space-y-6">
          <div className="bento relative h-[440px] overflow-hidden sm:h-[560px]">
            <Constellation points={data.points} active={sel} onHover={onHover} orbit={sel === null} camera labels={data.profiles.map(p => ({ segment: p.segment, name: p.name }))} className="size-full" />
            {hover && (
              <div className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-[130%] rounded-xl border border-line bg-raised px-3 py-2 text-sm shadow-xl" style={{ left: hover.x, top: hover.y }}>
                <div className="font-medium">{(hover as HoverPoint & { client_label?: string }).client_label}</div>
                <div className="num text-muted">Gasto previsto {money(hover.clv_pred)}</div>
              </div>
            )}
            <p className="pointer-events-none absolute bottom-3 left-4 right-4 text-xs text-muted">Cada punto es un cliente; la cercanía resume su similitud. Los ejes no son euros, gasto ni riesgo.</p>
          </div>

          <AnimatePresence mode="wait">
            <motion.div key={sel ?? 'all'} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.22 }}>
              {current ? (
                <article className="bento p-7 sm:p-10" style={{ borderTopColor: groupColor(current.segment), borderTopWidth: 3 }}>
                  <h2 className="headline !text-4xl">{current.name}</h2>
                  <p className="num mt-1 text-muted">{num(current.customers)} clientes · gasto previsto medio {money(current.clv_pred)}</p>
                  <dl className="mt-6 grid gap-x-10 gap-y-5 sm:grid-cols-2">
                    <div><dt className="font-medium">Qué observamos</dt><dd className="mt-1 text-muted">{current.reason.replace(/gasto estimado medio .*$/, `gasto estimado medio ${money(current.clv_pred)}.`)}</dd></div>
                    <div><dt className="font-medium">Categorías más compradas antes del corte</dt><dd className="mt-1 text-muted">{current.preferred_categories}</dd></div>
                    <div><dt className="font-medium">Qué probar</dt><dd className="mt-1 text-muted">{current.action}</dd></div>
                    <div><dt className="font-medium">Dónde y qué medir</dt><dd className="mt-1 text-muted">{current.channel}. Indicador: {current.kpi}.</dd></div>
                  </dl>
                </article>
              ) : (
                <p className="glass rounded-3xl p-6 text-muted">Elige un grupo para ver qué lo distingue, qué acción probar y qué medir. El resto de clientes se atenúa en el mapa.</p>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      <div className="mt-8 space-y-3">
        <Caveat>Las etiquetas comerciales son descriptivas: se asignan con reglas de gasto y días desde la última compra. No se ha entrenado una clasificación de abandono o fidelidad.</Caveat>
        <Disclosure title="Cómo se eligieron los grupos">
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="h-72">
              <ResponsiveContainer>
                <LineChart data={silhouette} margin={{ left: -10, right: 10 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="k" tickLine={false} axisLine={false} label={{ value: 'Número de grupos', position: 'insideBottom', offset: -4 }} />
                  <YAxis tickLine={false} axisLine={false} domain={[0, 'auto']} />
                  <Tooltip content={<ChartTip fmt={v => v.toFixed(3)} />} />
                  <Legend />
                  <Line dataKey="Representación de la red" stroke="var(--g1)" strokeWidth={2.5} dot={{ r: 4 }} />
                  <Line dataKey="Historial clásico (RFM)" stroke="var(--g3)" strokeWidth={2.5} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <DataTable maxHeight="288px" columns={[
              { key: 'space', label: 'Representación', render: spaceName },
              { key: 'k', label: 'Grupos', align: 'right' },
              { key: 'silhouette_train', label: 'Separación', align: 'right', render: (v: number) => v.toFixed(3) },
              { key: 'stability_ARI', label: 'Estabilidad', align: 'right', render: (v: number) => v.toFixed(3) },
            ]} rows={data.metrics} />
          </div>
          <p className="mt-4 text-sm text-muted">La silueta mide separación interna, no utilidad comercial demostrada. El número de grupos se elige en entrenamiento; la utilidad de cada acción requiere validación de negocio.</p>
        </Disclosure>
      </div>
    </div>
  )
}
