import { Search } from 'lucide-react'
import { motion } from 'motion/react'
import { useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import CountUp from '../components/CountUp'
import { Caveat, ChartTip, DataTable, Disclosure, ErrorState, Loading, PageHeader, Panel, ScenarioNotice, Stat } from '../components/ui'
import type { Customer, CustomerRow } from '../lib/api'
import { groupColor, num, useApp } from '../lib/app'
import { useScenarioApi } from '../lib/hooks'

export default function CustomerPage() {
  const { money } = useApp()
  const list = useScenarioApi<CustomerRow[]>('customers')
  const [query, setQuery] = useState('')
  const [id, setId] = useState<string | null>(null)
  const [k, setK] = useState(10)
  const [exclude, setExclude] = useState(false)

  useEffect(() => { if (list.data && !id) setId(list.data[0]?.id ?? null) }, [list.data, id])
  const resolved = useMemo(() => {
    if (!list.data) return []
    const q = query.trim().toLowerCase()
    if (!q) return list.data.slice(0, 8)
    return list.data.filter(c => c.label.toLowerCase().includes(q) || c.id.toLowerCase().startsWith(q)).slice(0, 8)
  }, [list.data, query])
  const typedId = query.trim().length === 32 ? query.trim() : null

  const row = useScenarioApi<Customer>(id ? `customer/${id}` : null, `k=${k}&exclude_seen=${exclude}`)

  if (list.error) return <ErrorState message={list.error} />
  if (!list.data) return <Loading />

  const c = row.data
  const history = c?.history.map(h => ({ date: h.order_purchase_timestamp.slice(0, 10), amount: h.amount, synthetic: h.is_synthetic })) ?? []
  const maxAffinity = c ? Math.max(...c.recommendations.map(r => r.affinity_score), 0.001) : 1
  const minAffinity = c ? Math.min(...c.recommendations.map(r => r.affinity_score), 0) : 0

  return (
    <div>
      <PageHeader title="Conocer a un cliente" lead="Consulta su historial, el gasto previsto y una propuesta de productos. Los alias facilitan la lectura; no son nombres reales." />
      <ScenarioNotice />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
        <aside className="space-y-5 lg:sticky lg:top-8 lg:self-start">
          <div>
            <label htmlFor="search" className="mb-2 block text-sm font-medium">Busca por alias o identificador</label>
            <div className="relative">
              <Search className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-muted" aria-hidden />
              <input id="search" className="field !pl-10" placeholder="Cliente 0042" value={query} onChange={e => setQuery(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') { if (typedId) setId(typedId); else if (resolved[0]) setId(resolved[0].id) } }} />
            </div>
            <ul className="mt-2 space-y-1" aria-label="Resultados">
              {typedId && <li><button className="w-full rounded-xl border border-line px-3 py-2 text-left text-sm hover:border-accent" onClick={() => setId(typedId)}>Usar el identificador completo</button></li>}
              {resolved.map(r => (
                <li key={r.id}>
                  <button onClick={() => setId(r.id)} aria-pressed={id === r.id}
                    className={`flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2 text-left text-sm transition-colors ${id === r.id ? 'bg-raised' : 'hover:bg-raised/60'}`}>
                    <span className="flex items-center gap-2"><span className="size-2.5 rounded-full" style={{ background: groupColor(r.segment) }} />{r.label}</span>
                    <span className="num text-muted">{money(r.clv_pred)}</span>
                  </button>
                </li>
              ))}
              {!resolved.length && !typedId && <li className="px-3 py-2 text-sm text-muted">Ningún cliente coincide. Escribe un alias como «Cliente 0042» o pega el identificador completo.</li>}
            </ul>
          </div>
          <div>
            <label htmlFor="k" className="mb-1 flex justify-between text-sm font-medium">Productos a revisar <span className="num text-muted">{k}</span></label>
            <input id="k" type="range" min={5} max={20} value={k} onChange={e => setK(+e.target.value)} />
          </div>
          <label className="flex cursor-pointer items-start gap-3 text-sm">
            <input type="checkbox" className="mt-1 size-4 accent-[var(--accent)]" checked={exclude} onChange={e => setExclude(e.target.checked)} />
            <span>Mostrar solo productos que no compró antes</span>
          </label>
        </aside>

        <div className="min-w-0 space-y-8">
          {row.error && <ErrorState message={row.error} />}
          {!c && !row.error && <Loading label="Consultando la ficha…" />}
          {c && (
            <motion.div key={c.customer_id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }} className="space-y-8">
              <div>
                <h2 className="text-3xl font-semibold">{c.label ?? 'Cliente de la muestra'}</h2>
                <div className="mt-6 grid grid-cols-2 gap-x-6 gap-y-7 xl:grid-cols-4">
                  <Stat label="Gasto previsto · 3 meses" value={<CountUp value={c.clv_3m_brl} format={n => money(n)} />} hint="No es beneficio ni valor de toda la vida." color={groupColor(c.segment)} />
                  <Stat label="Pedidos anteriores" value={<CountUp value={c.frequency} format={num} />} hint="Entregados antes del 01/04/2018." />
                  <Stat label="Última compra" value={<>Hace <CountUp value={c.recency_days} format={num} /> d</>} hint="Días en el corte histórico." />
                  <Stat label="Gasto anterior" value={<CountUp value={c.historical_spend_brl} format={n => money(n)} />} hint="Observado antes del corte." />
                </div>
              </div>

              <Panel>
                <h3 className="flex items-center gap-2.5 text-xl font-semibold"><span className="size-3 rounded-full" style={{ background: groupColor(c.segment) }} />Grupo: {c.profile.name}</h3>
                <p className="mt-3"><span className="font-medium">Acción que podrías probar:</span> <span className="text-muted">{c.profile.action}</span></p>
                <p className="mt-2"><span className="font-medium">Canal propuesto:</span> <span className="text-muted">{c.profile.channel}. Qué medir: {c.profile.kpi}.</span></p>
                <p className="mt-4 text-xs text-muted">Sugerencia basada en el perfil del grupo, no explicación causal de la predicción individual. Verifica permisos de contacto y condiciones comerciales antes de actuar.</p>
              </Panel>

              <section>
                <h3 className="text-2xl font-semibold">Qué productos podrías mostrarle</h3>
                {c.recommendations.length ? (
                  <>
                    <ol className="mt-5 divide-y divide-line border-y border-line">
                      {c.recommendations.map((r, i) => (
                        <li key={r.product_id} className="grid grid-cols-[2rem_1fr_auto] items-center gap-4 py-3">
                          <span className="num text-muted">{i + 1}</span>
                          <div className="min-w-0">
                            <div className="flex flex-wrap items-baseline gap-x-3"><span className="font-medium">Producto {r.product_id.slice(0, 8)}</span><span className="text-sm text-muted">{r.category_es}</span></div>
                            <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-line">
                              <motion.div className="h-full rounded-full bg-accent" initial={{ width: 0 }} animate={{ width: `${Math.max(4, ((r.affinity_score - minAffinity) / (maxAffinity - minAffinity || 1)) * 100)}%` }} transition={{ delay: i * 0.04, duration: 0.7, ease: [0.16, 1, 0.3, 1] }} />
                            </div>
                          </div>
                          <span className="num text-sm text-muted" title="Afinidad sin calibrar">{r.affinity_score.toFixed(2)}</span>
                        </li>
                      ))}
                    </ol>
                    <p className="mt-3 text-xs text-muted">El orden procede de la afinidad del modelo (no es una probabilidad). Olist no aporta nombres comerciales ni imágenes: no se inventan. La descarga incluye los IDs completos.</p>
                    <a className="btn btn-ghost mt-4" href={`data:text/csv;charset=utf-8,${encodeURIComponent('﻿' + ['orden,product_id,categoria,afinidad', ...c.recommendations.map((r, i) => `${i + 1},${r.product_id},${r.category_es},${r.affinity_score}`)].join('\n'))}`} download="seleccion_productos.csv">Descargar selección con IDs completos</a>
                  </>
                ) : <Caveat tone="warn">No hay recomendaciones disponibles con señal suficiente en este escenario y selección.</Caveat>}
              </section>

              <section>
                <h3 className="text-2xl font-semibold">Cómo ha comprado antes del corte</h3>
                {history.length ? (
                  <div className="mt-5 space-y-5">
                    <div className="h-64">
                      <ResponsiveContainer>
                        <BarChart data={history} margin={{ left: -8, right: 4 }}>
                          <CartesianGrid vertical={false} />
                          <XAxis dataKey="date" tickLine={false} axisLine={false} minTickGap={32} />
                          <YAxis tickLine={false} axisLine={false} tickFormatter={v => money(v)} />
                          <Tooltip cursor={{ fill: 'var(--raised)' }} content={<ChartTip fmt={v => money(v, 2)} />} />
                          <Bar dataKey="amount" name="Importe" radius={[4, 4, 0, 0]} animationDuration={700}>
                            {history.map((h, i) => <Cell key={i} fill={h.synthetic ? 'var(--g2)' : 'var(--g1)'} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <p className="flex flex-wrap gap-5 text-sm text-muted"><span className="flex items-center gap-2"><i className="size-2.5 rounded-full" style={{ background: 'var(--g1)' }} />Compra original</span><span className="flex items-center gap-2"><i className="size-2.5 rounded-full" style={{ background: 'var(--g2)' }} />Compra simulada</span></p>
                    <Disclosure title={`Ver los ${history.length} pedidos`}>
                      <DataTable maxHeight="320px" columns={[{ key: 'date', label: 'Fecha' }, { key: 'amount', label: 'Importe', align: 'right', render: (v: number) => money(v, 2) }, { key: 'synthetic', label: 'Origen', render: (v: boolean) => (v ? 'Compra simulada' : 'Compra original') }]} rows={history} />
                    </Disclosure>
                  </div>
                ) : <p className="mt-3 text-muted">Este cliente no tiene pedidos anteriores al corte.</p>}
              </section>

              <Disclosure title="Identificador y límites de esta ficha">
                <code className="block break-all rounded-xl bg-ink p-3 text-sm">{c.customer_id}</code>
                <p className="mt-3 text-sm text-muted">{c.warning}</p>
                <p className="mt-2 text-sm text-muted">La evaluación permite recomendar productos ya comprados. La opción de excluirlos cambia esta selección, no las métricas guardadas.</p>
              </Disclosure>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  )
}
