import { motion } from 'motion/react'
import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Caveat, ChartTip, DataTable, Disclosure, ErrorState, Loading, PageHeader, Panel, ScenarioNotice, Segmented, Stat, type Column } from '../components/ui'
import type { Evaluation, Row } from '../lib/api'
import { modelName, num, useApp } from '../lib/app'
import { useScenarioApi } from '../lib/hooks'

type Tab = 'gasto' | 'recomendacion' | 'conjunto'
const f = (d: number) => (v: number) => (typeof v === 'number' ? v.toFixed(d) : String(v ?? '–'))

function Bars({ data, x, label, fmt, highlight, lowerBetter }: { data: Row[]; x: string; label: string; fmt: (v: number) => string; highlight: string; lowerBetter?: boolean }) {
  const rows: Row[] = [...data].sort((a, b) => (lowerBetter ? 1 : -1) * ((a[x] as number) - (b[x] as number))).map(r => ({ ...r, name: modelName(r.model as string) }))
  return (
    <div style={{ height: 56 * rows.length + 40 }}>
      <ResponsiveContainer>
        <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 24 }}>
          <CartesianGrid horizontal={false} />
          <XAxis type="number" tickLine={false} axisLine={false} tickFormatter={fmt} />
          <YAxis type="category" dataKey="name" width={200} tickLine={false} axisLine={false} />
          <Tooltip cursor={{ fill: 'var(--raised)' }} content={<ChartTip fmt={v => fmt(v)} />} />
          <Bar dataKey={x} name={label} radius={[0, 6, 6, 0]} animationDuration={800}>
            {rows.map(r => <Cell key={r.model as string} fill={r.model === highlight ? 'var(--g2)' : r.model === 'multi' ? 'var(--g1)' : 'var(--line)'} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function ResultsPage() {
  const { money } = useApp()
  const { data: e, error } = useScenarioApi<Evaluation>('evaluation')
  const [tab, setTab] = useState<Tab>('gasto')
  const [split, setSplit] = useState<'test' | 'val'>('test')

  if (error) return <ErrorState message={error} />
  if (!e) return <Loading />

  const clv = e.clv.filter(r => r.split === split)
  const rec = e.recommendation.filter(r => r.split === split)
  const bestClv = [...clv].sort((a, b) => (a.RMSE as number) - (b.RMSE as number))[0]
  const jointClv = clv.find(r => r.model === 'multi')
  const eligible = Math.max(0, ...rec.map(r => Number(r.eligible_users) || 0))
  const bestRec = [...rec].sort((a, b) => (b['NDCG@10'] as number) - (a['NDCG@10'] as number))[0]
  const bySplit = (rows: Row[]) => rows.filter(r => r.split === split)

  const extraCols = (rows: Row[]): Column[] => Object.keys(rows[0] ?? {}).filter(k => !['source', 'split'].includes(k)).map(k => ({
    key: k, label: k, align: typeof rows[0][k] === 'number' ? 'right' : undefined, render: (v: any) => (typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(4) : v),
  }))

  const extraBlock = (title: string, rows: Row[]) => rows.length ? (
    <div key={title} className="space-y-3">
      <h3 className="text-xl font-semibold">{title}</h3>
      <DataTable columns={extraCols(bySplit(rows))} rows={bySplit(rows)} />
    </div>
  ) : null

  return (
    <div>
      <PageHeader title="Comprobar los resultados" lead="Comprueba si la herramienta aporta valor frente a alternativas más sencillas. Una red más compleja no tiene por qué ser mejor." />
      <ScenarioNotice />

      <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <Segmented label="Sección" value={tab} onChange={setTab} options={[{ value: 'gasto', label: 'Prever gasto' }, { value: 'recomendacion', label: 'Recomendar productos' }, { value: 'conjunto', label: 'Aprendizaje conjunto' }]} />
        <Segmented label="Periodo" value={split} onChange={setSplit} options={[{ value: 'test', label: 'Evaluación final' }, { value: 'val', label: 'Ajuste (validación)' }]} />
      </div>

      <motion.div key={tab + split} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }} className="space-y-8">
        {tab === 'gasto' && (
          <>
            <h2 className="max-w-2xl text-3xl font-semibold">¿Qué modelo se equivoca menos al prever el gasto?</h2>
            <p className="max-w-[65ch] text-lg text-muted">
              En este periodo, <strong className="text-text">{modelName(bestClv.model as string)}</strong> tiene el menor error RMSE: <strong className="text-text">{money(bestClv.RMSE as number)}</strong>.
              {jointClv && <> La red conjunta registra <strong className="text-text">{money(jointClv.RMSE as number)}</strong>.</>} Es una comparación del escenario, no una garantía comercial.
            </p>
            <Panel>
              <p className="mb-4 text-sm text-muted">Error de previsión (RMSE): una barra menor es mejor. En verde, el mejor; en violeta, la red conjunta.</p>
              <Bars data={clv} x="RMSE" label="Error RMSE" fmt={v => money(v)} highlight={bestClv.model as string} lowerBetter />
            </Panel>
            <DataTable columns={[
              { key: 'model', label: 'Modelo', render: modelName },
              { key: 'MAE', label: 'Error absoluto medio', align: 'right', render: (v: number) => money(v, 2) },
              { key: 'RMSE', label: 'Error RMSE', align: 'right', render: (v: number) => money(v, 2) },
              { key: 'Spearman', label: 'Ordenación (Spearman)', align: 'right', render: f(3) },
              { key: 'lift_top10', label: 'Concentración top 10 %', align: 'right', render: f(2) },
            ]} rows={clv} />
            <Caveat>MAE es el error absoluto medio; el RMSE penaliza más los errores grandes. Spearman evalúa el orden. El lift compara el gasto observado del decil predicho superior con la media y no es incremento causado por publicidad.</Caveat>
          </>
        )}

        {tab === 'recomendacion' && (
          <>
            <h2 className="max-w-2xl text-3xl font-semibold">¿Quién ordena mejor los productos?</h2>
            <p className="max-w-[65ch] text-lg text-muted">Clientes con compras futuras relevantes en esta evaluación: <strong className="num text-text">{num(eligible)}</strong>.</p>
            {eligible < 30 ? (
              <Caveat tone="warn">Hay muy pocos clientes evaluables. No interpretes estos resultados como evidencia fiable sobre la calidad del recomendador.</Caveat>
            ) : (
              <p className="max-w-[65ch] text-muted"><strong className="text-text">{modelName(bestRec.model as string)}</strong> obtiene la mejor ordenación NDCG@10. La herramienta sigue mostrando la red conjunta para estudiar el enfoque compartido, no porque gane necesariamente.</p>
            )}
            <Panel>
              <p className="mb-4 text-sm text-muted">Calidad del orden (NDCG@10): una barra mayor es mejor.</p>
              <Bars data={rec} x="NDCG@10" label="NDCG@10" fmt={v => v.toFixed(3)} highlight={bestRec.model as string} />
            </Panel>
            <DataTable columns={[
              { key: 'model', label: 'Modelo', render: modelName },
              { key: 'Precision@10', label: 'Aciertos entre 10 sugerencias', align: 'right', render: f(4) },
              { key: 'Recall@10', label: 'Compras relevantes recuperadas', align: 'right', render: f(4) },
              { key: 'NDCG@10', label: 'Calidad del orden', align: 'right', render: f(4) },
              { key: 'coverage', label: 'Cobertura del catálogo', align: 'right', render: f(3) },
              { key: 'eligible_users', label: 'Clientes evaluables', align: 'right', render: num },
            ]} rows={rec} />
            <Caveat>Las métricas están en escala 0–1. Solo se incluyen clientes con compras relevantes futuras; el catálogo reducido completo es el conjunto de candidatos. No son tasas de conversión de una campaña.</Caveat>
          </>
        )}

        {tab === 'conjunto' && (
          <>
            <h2 className="max-w-2xl text-3xl font-semibold">¿Ayuda compartir lo aprendido?</h2>
            <p className="max-w-[65ch] text-lg text-muted">Compara «Red conjunta» con las redes especializadas de las otras secciones. Si una especializada mejora, compartir la representación no ha aportado ventaja en esa tarea. Ese resultado es válido para la asignatura.</p>

            <section className="space-y-3">
              <h3 className="text-xl font-semibold">Selección retrospectiva frente a azar</h3>
              <div className="grid gap-6 sm:grid-cols-3">
                <Stat label="Clientes seleccionados" value={num(e.campaign.n_targeted)} hint="Decil superior según el modelo." color="var(--g1)" />
                <Stat label="Gasto observado de la selección" value={money(e.campaign.selected_observed_revenue)} hint="Compras en la ventana de evaluación." color="var(--g2)" />
                <Stat label="Referencia aleatoria esperada" value={money(e.campaign.random_expected_revenue)} hint="Promedio de selecciones al azar del mismo tamaño." color="var(--g3)" />
              </div>
              <Caveat tone="warn">Es una comparación retrospectiva de selección. No demuestra que una campaña haya generado ese gasto ni estima ingresos incrementales.</Caveat>
            </section>

            {e.sensitivity.length > 0 ? (
              <section className="space-y-3">
                <h3 className="text-xl font-semibold">Sensibilidad de configuración (solo validación)</h3>
                <DataTable columns={extraCols(e.sensitivity)} rows={e.sensitivity.map(r => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, v])))} maxHeight="360px" />
              </section>
            ) : <Caveat>No hay una exploración de sensibilidad guardada para este escenario.</Caveat>}

            <section className="space-y-3">
              <h3 className="text-xl font-semibold">Historial escaso</h3>
              <DataTable columns={extraCols(e.sparse_history)} rows={e.sparse_history} />
              <p className="text-sm text-muted">No equivale a evaluar clientes completamente nuevos.</p>
            </section>

            {extraBlock('Recomendación por categoría (K=3)', e.extras.category)}
            {extraBlock('Productos con pocas interacciones', e.extras.sparse_products)}
            {extraBlock('Objetivo alternativo: recompra en tres meses', e.extras.repurchase)}
            <Caveat>La versión por categoría promedia afinidades de productos, no entrena otra red. La recompra se compara con clasificadores independientes, no con una cabeza multi-tarea nueva. No se estima respuesta a publicidad.</Caveat>

            <Disclosure title="Curvas de aprendizaje del modelo actual">
              <div className="h-72">
                <ResponsiveContainer>
                  <LineChart data={e.learning} margin={{ left: -10, right: 10 }}>
                    <CartesianGrid vertical={false} />
                    <XAxis dataKey="epoch" tickLine={false} axisLine={false} label={{ value: 'Época', position: 'insideBottom', offset: -4 }} />
                    <YAxis tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTip fmt={v => v.toFixed(4)} />} />
                    <Legend />
                    <Line dataKey="clv_loss" name="Gasto" stroke="var(--g1)" strokeWidth={2} dot={false} />
                    <Line dataKey="rec_loss" name="Recomendación" stroke="var(--g2)" strokeWidth={2} dot={false} />
                    <Line dataKey="val_loss" name="Validación" stroke="var(--g3)" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="mt-3 text-sm text-muted">Las pérdidas no son porcentajes de ventas ni probabilidades. Su escala depende de la tarea y de su ponderación.</p>
            </Disclosure>
          </>
        )}
      </motion.div>
    </div>
  )
}
