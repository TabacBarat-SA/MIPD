import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { clearApiCache, getJson, type Source } from './api'

type Theme = 'dark' | 'light'
type Currency = 'USD' | 'BRL'

interface AppState {
  source: Source; setSource: (s: Source) => void
  theme: Theme; toggleTheme: () => void
  currency: Currency; setCurrency: (c: Currency) => void; canConvert: boolean
  money: (brl: number, digits?: number) => string
  refresh: () => void; version: number
}

const Ctx = createContext<AppState | null>(null)

const read = <T extends string>(key: string, fallback: T): T => {
  try { return (localStorage.getItem(key) as T) || fallback } catch { return fallback }
}
const write = (key: string, value: string) => { try { localStorage.setItem(key, value) } catch { /* sin almacenamiento */ } }

export function AppProvider({ children }: { children: ReactNode }) {
  const [source, setSourceState] = useState<Source>(() => read('360-source', 'synthetic'))
  const [theme, setTheme] = useState<Theme>(() => (document.documentElement.dataset.theme as Theme) || 'dark')
  const [currency, setCurrencyState] = useState<Currency>(() => read('360-currency', 'USD'))
  const [rate, setRate] = useState<number | null>(null)
  const [version, setVersion] = useState(0)

  useEffect(() => { getJson<{ BRL_USD: number | null }>('/api/fx').then(r => setRate(r.BRL_USD)).catch(() => setRate(null)) }, [])
  useEffect(() => { document.documentElement.dataset.theme = theme }, [theme])

  const effective: Currency = currency === 'USD' && rate ? 'USD' : 'BRL'
  const money = useCallback((brl: number, digits = 0) => {
    const value = effective === 'USD' ? brl * (rate as number) : brl
    return new Intl.NumberFormat(effective === 'USD' ? 'en-US' : 'pt-BR', {
      style: 'currency', currency: effective, minimumFractionDigits: digits, maximumFractionDigits: digits,
    }).format(value)
  }, [effective, rate])

  const value = useMemo<AppState>(() => ({
    source, setSource: s => { write('360-source', s); setSourceState(s) },
    theme, toggleTheme: () => setTheme(t => { const n = t === 'dark' ? 'light' : 'dark'; write('360-theme', n); return n }),
    currency: effective, setCurrency: c => { write('360-currency', c); setCurrencyState(c) }, canConvert: !!rate,
    money, version,
    refresh: () => { clearApiCache(); setVersion(v => v + 1) },
  }), [source, theme, effective, rate, money, version])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useApp() {
  const v = useContext(Ctx)
  if (!v) throw new Error('useApp fuera de AppProvider')
  return v
}

export const GROUP_COLORS = ['var(--g1)', 'var(--g2)', 'var(--g3)', 'var(--g4)', 'var(--g5)', 'var(--g6)']
export const groupColor = (segment: number) => GROUP_COLORS[segment % GROUP_COLORS.length]
export const num = (n: number) => new Intl.NumberFormat('es-ES', { useGrouping: true, minimumGroupingDigits: 1 } as Intl.NumberFormatOptions).format(Math.round(n))
export const pct = (n: number, d = 1) => `${(n * 100).toFixed(d).replace('.', ',')} %`

export const MODEL_NAMES: Record<string, string> = {
  multi: 'Red conjunta', clv_only: 'Red especializada en gasto', rec_only: 'Red especializada en recomendación',
  Media: 'Gasto medio (referencia)', 'Regresion lineal': 'Regresión lineal', 'Random Forest': 'Bosque aleatorio',
  'Gradient Boosting': 'Árboles potenciados', Popularidad: 'Productos populares', SVD: 'Filtrado colaborativo (SVD)',
}
export const modelName = (m: string) => MODEL_NAMES[m] ?? m
