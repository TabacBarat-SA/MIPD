import { useEffect, useState } from 'react'

export type Source = 'synthetic' | 'original'

const cache = new Map<string, unknown>()

export async function getJson<T>(url: string): Promise<T> {
  if (cache.has(url)) return cache.get(url) as T
  const res = await fetch(url)
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail ?? detail } catch { /* sin cuerpo JSON */ }
    throw new Error(String(detail))
  }
  const data = (await res.json()) as T
  cache.set(url, data)
  return data
}

export function clearApiCache() { cache.clear() }

export interface ApiState<T> { data: T | null; error: string | null; loading: boolean }

/** Carga una URL de la API. Con url=null no consulta. Mantiene el dato anterior mientras llega el nuevo. */
export function useApi<T>(url: string | null): ApiState<T> {
  const [state, setState] = useState<ApiState<T>>({ data: null, error: null, loading: !!url })
  useEffect(() => {
    if (!url) { setState({ data: null, error: null, loading: false }); return }
    let live = true
    setState(s => ({ ...s, error: null, loading: true }))
    getJson<T>(url)
      .then(data => live && setState({ data, error: null, loading: false }))
      .catch((e: Error) => live && setState(s => ({ ...s, error: e.message, loading: false })))
    return () => { live = false }
  }, [url])
  return state
}

export interface Profile {
  segment: number; name: string; customers: number; frequency: number; monetary: number
  recency: number; clv_pred: number; action: string; channel: string; kpi: string
  reason: string; preferred_categories: string
}
export interface Point { segment: number; pca_1: number; pca_2: number; clv_pred: number; client_label?: string }
export interface Overview {
  summary: { products: number; future_purchase_rate: number; backend: string; audit: { original_orders: number; added_orders: number } }
  customers: number; predicted_spend: number; products: number; groups: number
  histogram: { from: number; to: number; count: number }[]
  profiles: Pick<Profile, 'segment' | 'name' | 'customers' | 'clv_pred'>[]
  points: Point[]
}
export interface Groups {
  total: number; profiles: Profile[]; points: Point[]
  metrics: { space: string; k: number; silhouette_train: number; stability_ARI: number }[]
}
export interface CustomerRow { id: string; label: string; segment: number; clv_pred: number }
export interface Customer {
  customer_id: string; clv_3m_brl: number; segment: number; frequency: number; recency_days: number
  historical_spend_brl: number; warning: string; label: string
  recommendations: { product_id: string; category: string; category_es: string; affinity_score: number }[]
  profile: { name: string; action: string; channel: string; kpi: string }
  history: { order_id: string; order_purchase_timestamp: string; amount: number; is_synthetic: boolean }[]
  categories: { 'Categoría': string; 'Productos en la selección': number }[]
}
export interface CampaignRow {
  alias_interfaz: string; grupo: string; gasto_estimado_3_meses_brl: number
  dias_desde_ultima_compra: number; pedidos_previos: number
}
export interface Campaign { eligible: number; selected: number; predicted_spend: number; contact_cost: number; rows: CampaignRow[] }
export type Row = Record<string, string | number | null>
export interface Evaluation {
  clv: Row[]; recommendation: Row[]; sensitivity: Row[]; sparse_history: Row[]
  campaign: { n_targeted: number; selected_observed_revenue: number; random_expected_revenue: number }
  extras: { category: Row[]; sparse_products: Row[]; repurchase: Row[] }
  learning: { epoch: number; clv_loss: number; rec_loss: number; val_loss: number }[]
}
