import { Download } from 'lucide-react'
import { motion } from 'motion/react'
import { useEffect, useState } from 'react'
import CountUp from '../components/CountUp'
import { Caveat, DataTable, ErrorState, Loading, PageHeader, Panel, ScenarioNotice, Segmented, Stat } from '../components/ui'
import type { Campaign, Groups } from '../lib/api'
import { groupColor, num, useApp } from '../lib/app'
import { useScenarioApi } from '../lib/hooks'

const GOALS = ['Venta cruzada personalizada', 'Reactivación de clientes', 'Fidelización']

function useDebounced<T>(value: T, ms = 250) {
  const [v, setV] = useState(value)
  useEffect(() => { const t = setTimeout(() => setV(value), ms); return () => clearTimeout(t) }, [value, ms])
  return v
}

export default function CampaignPage() {
  const { money, source, currency } = useApp()
  const groups = useScenarioApi<Groups>('groups')
  const [goal, setGoal] = useState(GOALS[0])
  const [picked, setPicked] = useState<number[] | null>(null)
  const [minClv, setMinClv] = useState(0)
  const [limit, setLimit] = useState(100)
  const [recency, setRecency] = useState<number | null>(null)
  const [cost, setCost] = useState(0)
  const [ack, setAck] = useState(false)

  const total = groups.data?.total ?? 0
  const selected = picked ?? groups.data?.profiles.map(p => p.segment) ?? []
  const q = useDebounced(`segments=${selected.join(',')}&min_clv=${minClv}&limit=${limit}&cost=${cost}${recency !== null ? `&max_recency=${recency}` : ''}`)
  const plan = useScenarioApi<Campaign>(groups.data ? 'campaign' : null, q)

  if (groups.error) return <ErrorState message={groups.error} />
  if (!groups.data) return <Loading />

  const p = plan.data
  const toggle = (s: number) => setPicked(selected.includes(s) ? selected.filter(x => x !== s) : [...selected, s])
  const csv = `/api/campaign.csv?source=${source}&${q}&goal=${encodeURIComponent(goal)}`

  return (
    <div>
      <PageHeader title="Preparar una campaña" lead="Construye una audiencia de prueba y descarga una lista revisable. Este módulo no envía emails, no publica anuncios y no dispone de datos de consentimiento." />
      <ScenarioNotice />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
        <Panel className="space-y-6 lg:sticky lg:top-8 lg:self-start">
          <div>
            <p className="mb-2 font-medium">Objetivo de la prueba</p>
            <Segmented full label="Objetivo" value={goal} onChange={setGoal} options={GOALS.map(g => ({ value: g, label: g.split(' ')[0] === 'Venta' ? 'Venta cruzada' : g.split(' ')[0] }))} />
            <p className="mt-2 text-xs text-muted">Se guarda como contexto. No cambia la puntuación: la audiencia se ordena por gasto previsto.</p>
          </div>
          <div>
            <p className="mb-2 font-medium">Grupos que quieres incluir</p>
            <div className="flex flex-wrap gap-2">
              {groups.data.profiles.map(g => {
                const on = selected.includes(g.segment)
                return (
                  <button key={g.segment} onClick={() => toggle(g.segment)} aria-pressed={on}
                    className={`flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm transition-colors ${on ? 'border-transparent text-accent-ink' : 'border-line text-muted hover:text-text'}`}
                    style={on ? { background: groupColor(g.segment) } : undefined}>
                    {g.name}
                  </button>
                )
              })}
            </div>
          </div>
          <div>
            <label htmlFor="min" className="mb-1 flex justify-between font-medium">Gasto previsto mínimo <span className="num text-muted">{money(minClv)}</span></label>
            <input id="min" type="range" min={0} max={1500} step={25} value={minClv} onChange={e => setMinClv(+e.target.value)} />
            <p className="text-xs text-muted">Umbral en reales (BRL) por cliente: {num(minClv)} BRL.</p>
          </div>
          <div>
            <label htmlFor="limit" className="mb-1 flex justify-between font-medium">Máximo de clientes <span className="num text-muted">{num(limit)}</span></label>
            <input id="limit" type="range" min={1} max={Math.max(1, total)} value={limit} onChange={e => setLimit(+e.target.value)} />
          </div>
          <div>
            <label className="flex cursor-pointer items-center gap-3 font-medium">
              <input type="checkbox" className="size-4 accent-[var(--accent)]" checked={recency !== null} onChange={e => setRecency(e.target.checked ? 90 : null)} />
              Solo clientes que compraron hace poco
            </label>
            {recency !== null && (
              <div className="mt-3">
                <label htmlFor="rec" className="mb-1 flex justify-between text-sm text-muted">Máximo de días desde la última compra <span className="num text-text">{recency}</span></label>
                <input id="rec" type="range" min={0} max={365} step={10} value={recency} onChange={e => setRecency(+e.target.value)} />
              </div>
            )}
          </div>
          <div>
            <label htmlFor="cost" className="mb-1 block font-medium">Coste supuesto por contacto (BRL)</label>
            <input id="cost" className="field num" type="number" min={0} step={0.1} value={cost} onChange={e => setCost(Math.max(0, +e.target.value))} />
            <p className="mt-1 text-xs text-muted">Es un supuesto tuyo. El sistema no conoce costes reales ni calcula efecto incremental.</p>
          </div>
        </Panel>

        <div className="min-w-0 space-y-6">
          {plan.error ? <ErrorState message={plan.error} /> : !p ? <Loading label="Calculando la audiencia…" /> : (
            <motion.div className="space-y-6" animate={{ opacity: plan.loading ? 0.6 : 1 }} transition={{ duration: 0.15 }}>
              <div className="grid grid-cols-2 gap-x-6 gap-y-7 xl:grid-cols-4">
                <Stat label="Cumplen los filtros" value={<CountUp value={p.eligible} format={num} duration={0.5} />} color="var(--g1)" />
                <Stat label="Audiencia seleccionada" value={<CountUp value={p.selected} format={num} duration={0.5} />} hint="De mayor a menor gasto." color="var(--g2)" />
                <Stat label="Gasto previsto" value={<CountUp value={p.predicted_spend} format={n => money(n)} duration={0.5} />} hint="No es facturación de la campaña." color="var(--g3)" />
                <Stat label="Coste supuesto" value={<CountUp value={p.contact_cost} format={n => money(n)} duration={0.5} />} hint="Seleccionados × coste." color="var(--g4)" />
              </div>
              <Caveat tone="warn">El gasto previsto no es el ingreso que generará la campaña. No se calcula ROI ni aumento de ventas porque no se conoce el efecto causal de contactar a estos clientes.</Caveat>

              {p.selected === 0 ? (
                <Caveat>No hay clientes que cumplan los filtros. Amplía los grupos o reduce el mínimo de gasto previsto.</Caveat>
              ) : (
                <section className="space-y-4">
                  <h2 className="text-2xl font-semibold">Audiencia priorizada para revisar</h2>
                  <DataTable maxHeight="440px" columns={[
                    { key: 'alias_interfaz', label: 'Cliente' },
                    { key: 'grupo', label: 'Grupo', render: (v: string) => <span className="whitespace-nowrap">{v}</span> },
                    { key: 'gasto_estimado_3_meses_brl', label: `Gasto previsto (${currency})`, align: 'right', render: (v: number) => money(v, 2) },
                    { key: 'dias_desde_ultima_compra', label: 'Días desde la última compra', align: 'right', render: (v: number) => Math.round(v) },
                  ]} rows={p.rows} />
                  {p.selected > 100 && <p className="text-sm text-muted">Se muestran los primeros 100. La descarga incluye toda la audiencia seleccionada.</p>}
                  <p className="text-sm text-muted">Los IDs son históricos y anonimizados. No hay emails ni permisos de contacto: esta lista no es una audiencia lista para subir a una plataforma publicitaria.</p>
                  <label className="flex cursor-pointer items-start gap-3 rounded-2xl border border-line p-4 text-sm">
                    <input type="checkbox" className="mt-0.5 size-4 accent-[var(--accent)]" checked={ack} onChange={e => setAck(e.target.checked)} />
                    <span>Entiendo que es una simulación histórica y que debo revisar los permisos y la selección antes de cualquier uso real.</span>
                  </label>
                  <a className="btn" href={ack ? csv : undefined} aria-disabled={!ack} onClick={e => { if (!ack) e.preventDefault() }} style={ack ? undefined : { opacity: 0.4, cursor: 'not-allowed', boxShadow: 'none' }}>
                    <Download className="size-4" aria-hidden />Descargar audiencia de demostración
                  </a>
                </section>
              )}
            </motion.div>
          )}

          <Panel>
            <h2 className="text-2xl font-semibold">Cómo comprobar la idea de marketing</h2>
            <dl className="mt-5 space-y-4 text-muted">
              <div><dt className="font-medium text-text">Hipótesis propuesta</dt><dd>Una oferta adaptada al perfil puede mejorar la respuesta frente a una comunicación genérica. No es un resultado demostrado.</dd></div>
              <div><dt className="font-medium text-text">Diseño propuesto</dt><dd>Dentro de una audiencia válida, asignar al azar un grupo al mensaje personalizado y otro a una referencia o control. Fijar antes el indicador, el periodo y las condiciones de la oferta.</dd></div>
              <div><dt className="font-medium text-text">Qué registrar</dt><dd>Personas asignadas, mensajes entregados, clics, compras, gasto, costes y bajas. Comparar tasas sobre el mismo denominador y revisar la incertidumbre antes de concluir.</dd></div>
            </dl>
            <p className="mt-5 text-xs text-muted">El proyecto no ejecuta esta prueba. La asignación, los resultados y el consentimiento no están en los datos disponibles.</p>
          </Panel>
        </div>
      </div>
    </div>
  )
}
