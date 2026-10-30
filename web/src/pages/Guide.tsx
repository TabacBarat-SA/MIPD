import { Search } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { PageHeader } from '../components/ui'

const STEPS = [
  ['/', 'Resumen', 'Identifica el tamaño de la muestra y entiende qué se está estimando.'],
  ['/grupos', 'Grupos', 'Revisa qué diferencia a cada perfil y qué acción podrías probar.'],
  ['/cliente', 'Cliente', 'Contrasta la propuesta de productos con su historial.'],
  ['/campana', 'Campaña', 'Elige objetivo, filtros y tamaño; revisa la audiencia antes de descargarla.'],
  ['/resultados', 'Resultados', 'Verifica si la red aporta algo frente a métodos más sencillos.'],
]

const GLOSSARY = [
  ['Gasto previsto / CLV a 3 meses', 'Estimación de cuánto gastará el cliente en ese periodo; no beneficio ni valor de toda su vida.'],
  ['Días desde la última compra', 'Tiempo transcurrido en la fecha histórica de corte. En los datos se llama recencia.'],
  ['Grupo / segmento', 'Conjunto de clientes con patrones parecidos. No demuestra intención de compra o abandono.'],
  ['Afinidad', 'Puntuación para ordenar productos. No significa un porcentaje de probabilidad.'],
  ['Representación / embedding', 'Resumen numérico que aprende la red a partir de los datos.'],
  ['Ablación', 'Comparación de la red conjunta con las tareas entrenadas por separado.'],
  ['Datos semisintéticos', 'Originales más compras artificiales identificadas. No equivalen a comportamiento real validado.'],
  ['BRL / R$', 'Reales brasileños. Los datos están en BRL; la web puede mostrarlos convertidos a dólares con el tipo de cambio actual.'],
]

export default function Guide() {
  const [q, setQ] = useState('')
  const terms = GLOSSARY.filter(([t, d]) => (t + d).toLowerCase().includes(q.trim().toLowerCase()))
  return (
    <div>
      <PageHeader title="Guía de uso" lead="Un recorrido pensado para una persona de marketing, sin conocimientos técnicos." />

      <ol className="max-w-3xl divide-y divide-line border-y border-line">
        {STEPS.map(([to, name, body], i) => (
          <li key={to}>
            <Link to={to} className="group grid grid-cols-[2.5rem_1fr] gap-x-4 py-5 transition-colors hover:bg-surface/60 sm:px-3">
              <span className="num font-display text-2xl font-semibold text-accent">{i + 1}</span>
              <span><span className="block text-xl font-semibold group-hover:text-accent">{name}</span><span className="text-muted">{body}</span></span>
            </Link>
          </li>
        ))}
      </ol>

      <section className="mt-20 max-w-3xl">
        <h2 className="headline !text-4xl">Diccionario sin tecnicismos</h2>
        <div className="relative mt-6">
          <Search className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-muted" aria-hidden />
          <input className="field !pl-10" placeholder="Busca un concepto" aria-label="Buscar en el diccionario" value={q} onChange={e => setQ(e.target.value)} />
        </div>
        <dl className="mt-6 divide-y divide-line">
          {terms.map(([t, d]) => (
            <div key={t} className="py-4 sm:grid sm:grid-cols-[14rem_1fr] sm:gap-8">
              <dt className="font-display font-semibold">{t}</dt>
              <dd className="text-muted">{d}</dd>
            </div>
          ))}
          {!terms.length && <p className="py-4 text-muted">No hay conceptos que coincidan con «{q}».</p>}
        </dl>
      </section>

      <section className="mt-20 max-w-3xl">
        <h2 className="headline !text-4xl">Antes de presentar o utilizar la herramienta</h2>
        <p className="mt-3 max-w-[62ch] text-muted">Aclara el escenario, la fecha histórica y la moneda. No atribuyas ventas a una campaña no ejecutada. No presentes afinidades como probabilidades. Muestra también los modelos que superan a la red y los requisitos pendientes.</p>
        <p className="mt-3 text-sm text-muted">Para instalar y entrenar: README.md. Para justificar decisiones: docs/metodologia.md. Para revisar el alcance: docs/cobertura.md.</p>
      </section>
    </div>
  )
}
