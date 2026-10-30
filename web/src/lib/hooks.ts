import { useApi } from './api'
import { useApp } from './app'

/** Consulta un endpoint de /api con el escenario activo. El parametro v invalida cuando se pulsa Actualizar. */
export function useScenarioApi<T>(path: string | null, params = '') {
  const { source, version } = useApp()
  const url = path ? `/api/${path}?source=${source}${params ? `&${params}` : ''}&v=${version}` : null
  return useApi<T>(url)
}
