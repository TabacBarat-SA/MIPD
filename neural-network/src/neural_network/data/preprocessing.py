"""
Lectura de parquets, construcción de mapeos de IDs, verificación anti-fuga
y estadísticas de cold-start.

Decisiones documentadas:
- El target CLV se usa directamente de la columna ``clv_log`` (ya es log1p(clv_futuro)),
  calculada por data-processing. No se recalcula.
- Los índices de cliente y producto son directamente ``customer_idx`` y ``product_idx``
  (enteros ya asignados por data-processing). No se crean nuevos mapeos; se usan
  como están.
- Cold-start: los embeddings de clientes/productos no vistos en train se inicializan
  aleatoriamente (mismo proceso que el resto). Se reportan métricas separadas
  para "vistos" vs "nuevos" en val/test.
- Anti-fuga: se verifica que max(fecha_train) < min(fecha_val) < min(fecha_test).
- Normalización de features: las stats (media, std) se calculan SOLO sobre el
  split de train y se devuelven junto con los DataFrames para aplicarlas de forma
  consistente a val/test sin data leakage.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ─── Carga ────────────────────────────────────────────────────────────────── #

def cargar_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    """Lee y devuelve el archivo ``metadata.json``."""
    ruta: Path = cfg["datos"]["_metadata_abs"]
    with ruta.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def cargar_parquets(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Lee todos los parquets y los devuelve en un diccionario.

    Claves: ``clientes_train``, ``clientes_val``, ``clientes_test``,
    ``interacciones_train``, ``interacciones_val``, ``interacciones_test``,
    ``productos``.
    """
    claves = [
        "clientes_train", "clientes_val", "clientes_test",
        "interacciones_train", "interacciones_val", "interacciones_test",
        "productos",
    ]
    datos: dict[str, pd.DataFrame] = {}
    for clave in claves:
        ruta: Path = cfg["datos"][f"_{clave}_abs"]
        datos[clave] = pd.read_parquet(ruta)
    return datos


# ─── Validación de integridad ─────────────────────────────────────────────── #

def validar_rangos_idx(datos: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> None:
    """Verifica que customer_idx y product_idx estén dentro del rango del config.

    Lanza ``ValueError`` si algún índice cae fuera del rango esperado, lo que
    evitaría errores crípticos de CUDA al acceder a la tabla de embeddings.
    """
    n_clientes = cfg["modelo"]["n_clientes"]
    n_productos = cfg["modelo"]["n_productos"]
    col_c = cfg["columnas"]["interaccion_cliente_idx"]
    col_p = cfg["columnas"]["interaccion_producto_idx"]

    for split in ("train", "val", "test"):
        df = datos[f"interacciones_{split}"]

        max_c = int(df[col_c].max())
        min_c = int(df[col_c].min())
        if min_c < 0 or max_c >= n_clientes:
            raise ValueError(
                f"[validar_rangos] interacciones_{split}: customer_idx fuera de "
                f"[0, {n_clientes}): rango encontrado [{min_c}, {max_c}]"
            )

        max_p = int(df[col_p].max())
        min_p = int(df[col_p].min())
        if min_p < 0 or max_p >= n_productos:
            raise ValueError(
                f"[validar_rangos] interacciones_{split}: product_idx fuera de "
                f"[0, {n_productos}): rango encontrado [{min_p}, {max_p}]"
            )

    col_c_cli = cfg["columnas"]["cliente_idx"]
    for split in ("train", "val", "test"):
        df_c = datos[f"clientes_{split}"]
        max_c = int(df_c[col_c_cli].max())
        min_c = int(df_c[col_c_cli].min())
        if min_c < 0 or max_c >= n_clientes:
            raise ValueError(
                f"[validar_rangos] clientes_{split}: customer_idx fuera de "
                f"[0, {n_clientes}): rango encontrado [{min_c}, {max_c}]"
            )

    # silent: índices válidos


def verificar_anti_fuga(datos: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> None:
    """Verifica que no haya fuga temporal entre splits.

    Comprueba que max(fecha_train) < min(fecha_val) y max(fecha_val) < min(fecha_test).
    Lanza un Warning (no excepción) para no bloquear el pipeline en caso de que
    el dataset no tenga columna 'fecha' en clientes_*.
    """
    col_fecha = cfg["columnas"]["interaccion_fecha"]
    try:
        max_train = datos["interacciones_train"][col_fecha].max()
        min_val   = datos["interacciones_val"][col_fecha].min()
        max_val   = datos["interacciones_val"][col_fecha].max()
        min_test  = datos["interacciones_test"][col_fecha].min()

        ok_1 = max_train < min_val
        ok_2 = max_val < min_test

        if not ok_1:
            warnings.warn(
                f"FUGA DETECTADA: max fecha train ({max_train}) >= min fecha val ({min_val})",
                stacklevel=2,
            )
        if not ok_2:
            warnings.warn(
                f"FUGA DETECTADA: max fecha val ({max_val}) >= min fecha test ({min_test})",
                stacklevel=2,
            )
    except KeyError as e:
        warnings.warn(f"No se pudo verificar anti-fuga: {e}", stacklevel=2)


# ─── Construcción de estructuras de datos ─────────────────────────────────── #

def construir_positivos_por_cliente(
    df: pd.DataFrame,
    col_cliente: str,
    col_producto: str,
) -> dict[int, set[int]]:
    """Construye {customer_idx → set(product_idx)} desde un DataFrame.

    Esta es la implementación canónica; el resto del módulo la importa desde aquí.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de interacciones.
    col_cliente : str
        Nombre de la columna de índice de cliente.
    col_producto : str
        Nombre de la columna de índice de producto.
    """
    pos: dict[int, set[int]] = {}
    # groupby + agg es más rápido que iterar fila a fila con zip
    for c, group in df.groupby(col_cliente)[col_producto]:
        pos[int(c)] = set(group.astype(int).tolist())
    return pos


# Alias de compatibilidad hacia atrás (usado en algunas versiones antiguas del módulo)
construir_positivos_por_cliente_desde_df = construir_positivos_por_cliente


# ─── Estadísticas ─────────────────────────────────────────────────────────── #

def calcular_estadisticas_coldstart(
    datos: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Calcula y reporta estadísticas de cold-start.

    Devuelve un dict con:
        - ``clientes_train_vistos``: set de customer_idx en interacciones_train
        - ``productos_train_vistos``: set de product_idx en interacciones_train
        - estadísticas de solapamiento para val y test
    """
    col_c = cfg["columnas"]["interaccion_cliente_idx"]
    col_p = cfg["columnas"]["interaccion_producto_idx"]

    train_c = set(datos["interacciones_train"][col_c].unique())
    train_p = set(datos["interacciones_train"][col_p].unique())

    stats: dict[str, Any] = {"clientes_train_vistos": train_c, "productos_train_vistos": train_p}

    for split in ("val", "test"):
        df_i = datos[f"interacciones_{split}"]
        c_split = set(df_i[col_c].unique())
        p_split = set(df_i[col_p].unique())

        c_nuevos = c_split - train_c
        p_nuevos = p_split - train_p

        stats[f"clientes_{split}_nuevos"] = c_nuevos
        stats[f"clientes_{split}_vistos"] = c_split & train_c
        stats[f"productos_{split}_nuevos"] = p_nuevos
        stats[f"productos_{split}_vistos"] = p_split & train_p

        # (stats internas, no se imprime)

    return stats


def calcular_estadisticas_clv(
    datos: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Calcula % de CLV=0 y descripción de la distribución."""
    col_clv = cfg["columnas"]["clv_futuro"]
    stats: dict[str, Any] = {}

    for split in ("train", "val", "test"):
        clv = datos[f"clientes_{split}"][col_clv]
        stats[f"pct_clv_cero_{split}"] = round((clv == 0).mean() * 100, 2)

    return stats


def calcular_popularidad_productos(
    interacciones_train: pd.DataFrame,
    n_productos: int,
    cfg: dict[str, Any],
) -> np.ndarray:
    """Calcula la popularidad de cada producto en train para muestreo sesgado.

    Popularidad = nº de interacciones en train. Devuelve un array de longitud
    ``n_productos`` con la popularidad^0.75 normalizada (suma a 1).
    Productos sin interacciones tienen popularidad 0 → probabilidad 0.
    """
    col_p = cfg["columnas"]["interaccion_producto_idx"]
    conteos = interacciones_train[col_p].value_counts()
    pop = np.zeros(n_productos, dtype=np.float64)
    for idx, cnt in conteos.items():
        if 0 <= int(idx) < n_productos:
            pop[int(idx)] = float(cnt)

    pop = pop ** 0.75
    total = pop.sum()
    if total > 0:
        pop /= total
    else:
        pop[:] = 1.0 / n_productos
    return pop


def calcular_stats_features_train(
    clientes_train: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Calcula media y desviación estándar de las features en el split de train.

    Estas stats deben aplicarse a val/test para evitar data leakage.

    Parámetros
    ----------
    clientes_train : pd.DataFrame
        DataFrame de clientes del split de entrenamiento.
    cfg : dict
        Configuración completa.

    Devuelve
    -------
    (mean, std) : tupla de arrays shape (n_features,) o (None, None) si
        ``usar_features=False``.
    """
    if not cfg["modelo"]["usar_features"]:
        return None, None

    names = cfg["columnas"]["features_cliente"]
    arr = clientes_train[names].to_numpy(dtype=np.float32)
    mean = arr.mean(axis=0)               # (n_features,)
    std  = arr.std(axis=0) + 1e-8         # (n_features,)
    return mean, std


# ─── Carga completa ───────────────────────────────────────────────────────── #

def cargar_todo(cfg: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    """Función de conveniencia: carga parquets, verifica anti-fuga y calcula stats.

    Devuelve
    -------
    datos:
        Diccionario con todos los DataFrames.
    stats_coldstart:
        Estadísticas de cold-start (clientes/productos nuevos vs. vistos).
    stats_clv:
        Estadísticas de CLV (% de ceros por split).
    """
    datos = cargar_parquets(cfg)
    validar_rangos_idx(datos, cfg)
    verificar_anti_fuga(datos, cfg)
    stats_cs = calcular_estadisticas_coldstart(datos, cfg)
    stats_clv = calcular_estadisticas_clv(datos, cfg)
    return datos, stats_cs, stats_clv
