import { motion } from 'motion/react'
import { Check, X } from 'lucide-react'
import { Caveat, Disclosure, ErrorState, Loading, PageHeader, ScenarioNotice } from '../components/ui'
import type { Overview } from '../lib/api'
import { useScenarioApi } from '../lib/hooks'

const STEPS = [
  ['Reunir el historial', 'Se unen clientes, pedidos, artículos, pagos y productos. La identidad persistente es customer_unique_id; customer_id cambia entre pedidos.'],
  ['Aprender una representación compartida', 'La red resume patrones del cliente y del producto en vectores numéricos. Combina identidad, historial, categoría y atributos físicos.'],
  ['Estimar y recomendar', 'Una salida estima el gasto futuro; otra ordena productos por afinidad. Ambas comparten parte de lo aprendido. La segmentación usa después esa representación, no una tercera cabeza supervisada.'],
  ['Traducir a acciones y comprobar', 'Los grupos reciben etiquetas comerciales descriptivas. Se proponen acciones, se prepara una audiencia y se compara el modelo con alternativas. La eficacia publicitaria necesita un experimento adicional.'],
]

const KNOWS: [string, string, boolean][] = [
  ['Disponible', 'Historial de pedidos, importes, productos, categorías y atributos.', true],
  ['No disponible', 'Emails, permisos de contacto, exposición a anuncios, clics o resultados de campañas.', false],
  ['Estimado', 'Gasto futuro y orden de afinidad de productos.', true],
  ['No demostrado', 'Probabilidad calibrada de compra, abandono individual, beneficio, ROI o efecto causal de publicidad.', false],
]

const WINDOWS = [
  { name: 'Entrenamiento', cut: '01/10/2017', end: '01/01/2018', color: 'var(--g1)' },
  { name: 'Validación', cut: '01/01/2018', end: '01/04/2018', color: 'var(--g3)' },
  { name: 'Prueba final', cut: '01/04/2018', end: '01/07/2018', color: 'var(--g2)' },
]

export default function HowItWorks() {
  const { data, error } = useScenarioApi<Overview>('overview')
  if (error) return <ErrorState message={error} />
  if (!data) return <Loading />

  return (
    <div>
      <PageHeader title="Cómo funciona" lead="Un mismo historial alimenta tres decisiones. La evaluación técnica permite comprobar si la complejidad está justificada." />
      <ScenarioNotice />

      <ol className="relative max-w-3xl space-y-10 pl-14">
        <motion.span className="absolute bottom-3 left-[19px] top-3 w-0.5 origin-top bg-accent/50" initial={{ scaleY: 0 }} whileInView={{ scaleY: 1 }} viewport={{ once: true }} transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1] }} aria-hidden />
        {STEPS.map(([title, body], i) => (
          <motion.li key={title} className="relative" initial={{ opacity: 0, x: -12 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true, margin: '-60px' }} transition={{ duration: 0.5, delay: 0.05 }}>
            <span className="num absolute -left-14 top-0 grid size-10 place-items-center rounded-full border border-accent bg-ink font-display font-semibold text-accent">{i + 1}</span>
            <h2 className="text-2xl font-semibold">{title}</h2>
            <p className="mt-2 max-w-[60ch] text-muted">{body}</p>
          </motion.li>
        ))}
      </ol>

      <section className="mt-24">
        <h2 className="text-3xl font-semibold">Qué sabe el sistema y qué no sabe</h2>
        <dl className="mt-8 grid gap-px overflow-hidden rounded-3xl border border-line bg-line sm:grid-cols-2">
          {KNOWS.map(([t, d, ok]) => (
            <div key={t} className="bg-surface p-6">
              <dt className="flex items-center gap-2 font-display text-lg font-semibold">
                {ok ? <Check className="size-5 text-good" aria-hidden /> : <X className="size-5 text-warn" aria-hidden />}{t}
              </dt>
              <dd className="mt-2 text-muted">{d}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="mt-24">
        <h2 className="text-3xl font-semibold">Protocolo: el modelo nunca ve el futuro</h2>
        <p className="mt-3 max-w-[62ch] text-muted">Las entradas usan compras anteriores a cada corte. Las etiquetas son el gasto y los productos de los tres meses siguientes. Las ventanas no se solapan.</p>
        <div className="mt-10 grid gap-4 md:grid-cols-3">
          {WINDOWS.map((w, i) => (
            <motion.div key={w.name} initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.12, duration: 0.5 }}>
              <div className="h-2 rounded-full" style={{ background: w.color }} />
              <h3 className="mt-4 text-xl font-semibold">{w.name}</h3>
              <p className="num mt-1 text-sm text-muted">Corte {w.cut}</p>
              <p className="num text-sm text-muted">Etiquetas hasta {w.end}</p>
            </motion.div>
          ))}
        </div>
        <div className="mt-8">
          <Disclosure title="Protocolo académico y correspondencia con la propuesta">
            <ul className="max-w-[72ch] list-disc space-y-2 pl-5 text-muted">
              <li>Escalador, centroides y PCA se ajustan con entrenamiento. Los pesos no se entrenan con etiquetas de test.</li>
              <li>La red combina MSE de gasto y BCE de recomendación, con negativos muestreados y ponderación lambda.</li>
              <li>Se comparan modelos clásicos, tareas aisladas, sensibilidad y segmentación RFM frente a representación de la red.</li>
              <li>Se conserva el resultado aunque la red no mejore a los modelos clásicos.</li>
            </ul>
            <p className="mt-4 text-muted">Motor registrado para estos pesos: <strong className="text-text">{data.summary.backend}</strong>. La configuración permite entrenar con PyTorch y CUDA; los pesos conservan su procedencia real.</p>
          </Disclosure>
        </div>
        <div className="mt-4"><Caveat tone="warn">Cobertura parcial de la propuesta: la variante por categoría agrega la salida de producto y la recompra usa clasificadores independientes; no se ha entrenado una cabeza adicional. La validación completa de Docker y CUDA en el equipo destino sigue pendiente. Consulta docs/cobertura.md.</Caveat></div>
      </section>
    </div>
  )
}
