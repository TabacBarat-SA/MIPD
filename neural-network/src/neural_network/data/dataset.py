"""
Bloque 1 – Dataset con negative sampling para la tarea de recomendación.

Diseño:
    - ``InteractionDataset`` hereda de ``torch.utils.data.Dataset``.
    - Cada muestra es la tupla ``(cliente_idx, producto_idx, etiqueta, clv_target)``
      donde todos son tensores escalares (long, long, float, float).
    - Por cada positivo se generan ``k`` negativos (etiqueta=0) muestreados
      entre productos que el cliente NO ha comprado en el split actual
      (val/test excluye además positivos de train).
    - En entrenamiento: los negativos se regeneran en cada época (se llama a
      ``regenerar_negativos()`` al inicio del epoch).
    - En validación/test: los negativos se fijan con ``semilla_negativos_val``
      para reproducibilidad entre épocas.
    - Muestreo sesgado por popularidad^0.75 (calculado solo en train) si se
      proporciona el array ``popularidad``.
    - ``__len__`` = (nº positivos) × (1 + k), garantizando proporción exacta 1:k.

Optimización del negative sampling:
    - Los negativos se generan agrupando por cliente (no por positivo individual),
      lo que evita la complejidad O(n_positivos × n_productos) del loop original.
    - Para cada cliente único se llama una sola vez a ``np.setdiff1d`` y a
      ``rng.choice``, reduciendo el número de operaciones numpy de
      ``n_positivos`` a ``n_clientes_únicos``.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """Dataset de interacciones con negative sampling.

    Parámetros
    ----------
    clientes_idx : np.ndarray
        Array 1-D con los ``customer_idx`` de cada positivo.
    productos_idx : np.ndarray
        Array 1-D con los ``product_idx`` de cada positivo.
    clv_targets : np.ndarray
        Array 1-D con el target CLV (``clv_log``) para cada cliente positivo.
    positivos_por_cliente : dict[int, set[int]]
        Mapa {customer_idx → set de product_idx comprados}, usado para
        asegurarse de que los negativos no coincidan con positivos.
    n_productos : int
        Número total de productos (tamaño del espacio de muestreo).
    k : int
        Número de negativos por positivo.
    popularidad : np.ndarray | None
        Array de popularidad^0.75 normalizada (longitud = n_productos).
        Si ``None`` se usa muestreo uniforme.
    semilla : int | None
        Si se proporciona, los negativos se fijan con esta semilla (modo val/test).
        Si ``None``, los negativos se regeneran aleatoriamente en cada llamada a
        ``regenerar_negativos()`` (modo train).
    """

    def __init__(
        self,
        clientes_idx: np.ndarray,
        productos_idx: np.ndarray,
        clv_targets: np.ndarray,
        positivos_por_cliente: dict[int, set[int]],
        n_productos: int,
        k: int,
        popularidad: np.ndarray | None = None,
        semilla: int | None = None,
    ) -> None:
        assert len(clientes_idx) == len(productos_idx) == len(clv_targets), (
            "clientes_idx, productos_idx y clv_targets deben tener la misma longitud"
        )
        self._clientes_pos: np.ndarray = np.asarray(clientes_idx, dtype=np.int64)
        self._productos_pos: np.ndarray = np.asarray(productos_idx, dtype=np.int64)
        self._clv_pos: np.ndarray = np.asarray(clv_targets, dtype=np.float32)
        self._positivos_por_cliente = positivos_por_cliente
        self._n_productos = n_productos
        self._k = k
        self._popularidad = popularidad
        self._semilla = semilla
        self._n_pos = len(clientes_idx)

        # Mapa cliente → CLV (para asignar clv a los negativos del mismo cliente)
        self._clv_por_cliente: dict[int, float] = {}
        for c, clv in zip(self._clientes_pos.tolist(), self._clv_pos.tolist()):
            self._clv_por_cliente[int(c)] = float(clv)

        # Índices únicos de clientes presentes en los positivos (para agrupar)
        self._clientes_unicos: np.ndarray = np.unique(self._clientes_pos)

        # Almacena los negativos generados
        self._negativos_clientes: np.ndarray = np.empty(0, dtype=np.int64)
        self._negativos_productos: np.ndarray = np.empty(0, dtype=np.int64)
        self._clv_neg: np.ndarray = np.empty(0, dtype=np.float32)

        # Genera negativos iniciales
        self.regenerar_negativos()

    # ------------------------------------------------------------------ #
    #  Negative sampling (vectorizado por cliente)                        #
    # ------------------------------------------------------------------ #

    def regenerar_negativos(self) -> None:
        """Genera o regenera los negativos de forma vectorizada y ultra rápida.

        En vez de iterar cliente a cliente con np.setdiff1d (que tomaba minutos),
        realiza un muestreo global ponderado en un solo paso vectorizado y resuelve
        las raras colisiones (menos del 0.05% de probabilidad) mediante rechazo.
        """
        rng = (
            np.random.default_rng(self._semilla)
            if self._semilla is not None
            else np.random.default_rng()
        )

        n_neg = self._n_pos * self._k
        if n_neg == 0:
            self._negativos_clientes = np.empty(0, dtype=np.int64)
            self._negativos_productos = np.empty(0, dtype=np.int64)
            self._clv_neg = np.empty(0, dtype=np.float32)
            return

        # Vectorizado: repetir clientes y CLV exactamente k veces por cada positivo
        neg_c = np.repeat(self._clientes_pos, self._k)
        neg_clv = np.repeat(self._clv_pos, self._k)

        # Muestreo global según popularidad (o uniforme) en 1 sola llamada C/NumPy
        neg_p = rng.choice(self._n_productos, size=n_neg, replace=True, p=self._popularidad).astype(np.int64)

        # Filtrar colisiones con los positivos conocidos del cliente
        pos_dict = self._positivos_por_cliente
        for i in range(n_neg):
            c_int = int(neg_c[i])
            pos_c = pos_dict.get(c_int)
            if pos_c and int(neg_p[i]) in pos_c:
                if len(pos_c) < self._n_productos:
                    while int(neg_p[i]) in pos_c:
                        neg_p[i] = rng.choice(self._n_productos, p=self._popularidad)

        self._negativos_clientes = neg_c
        self._negativos_productos = neg_p
        self._clv_neg = neg_clv

    # ------------------------------------------------------------------ #
    #  Dataset interface                                                   #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Devuelve nº positivos × (1 + k)."""
        return self._n_pos + len(self._negativos_clientes)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Devuelve ``(cliente_idx, producto_idx, etiqueta, clv_target)`` como tensores.

        - ``cliente_idx``: long
        - ``producto_idx``: long
        - ``etiqueta``: float (1.0 positivo, 0.0 negativo)
        - ``clv_target``: float (``clv_log`` del cliente)
        """
        if idx < self._n_pos:
            c = self._clientes_pos[idx]
            p = self._productos_pos[idx]
            clv = self._clv_pos[idx]
            etiqueta = 1.0
        else:
            neg_idx = idx - self._n_pos
            c = self._negativos_clientes[neg_idx]
            p = self._negativos_productos[neg_idx]
            clv = self._clv_neg[neg_idx]
            etiqueta = 0.0

        return (
            torch.tensor(c, dtype=torch.long),
            torch.tensor(p, dtype=torch.long),
            torch.tensor(etiqueta, dtype=torch.float32),
            torch.tensor(clv, dtype=torch.float32),
        )


# ─── Función de construcción ────────────────────────────────────────────────── #

def construir_dataset(
    interacciones: Any,  # pd.DataFrame
    clientes: Any,       # pd.DataFrame
    positivos_excluir: dict[int, set[int]] | None,
    n_productos: int,
    cfg: dict[str, Any],
    popularidad: np.ndarray | None = None,
    semilla: int | None = None,
) -> InteractionDataset:
    """Construye un ``InteractionDataset`` desde DataFrames.

    Parámetros
    ----------
    interacciones:
        DataFrame de interacciones del split (interacciones_train/val/test).
    clientes:
        DataFrame de clientes del split (clientes_train/val/test) para obtener clv_log.
    positivos_excluir:
        Positivos adicionales que se excluyen del muestreo negativo
        (en val/test, se excluyen también los positivos de train).
    n_productos:
        Número total de productos.
    cfg:
        Configuración.
    popularidad:
        Array de popularidad^0.75 normalizada (calculada solo en train).
    semilla:
        Semilla fija para val/test (None para train).
    """
    # Importa la función canónica desde preprocessing para evitar duplicación
    from neural_network.data.preprocessing import construir_positivos_por_cliente

    col_c = cfg["columnas"]["interaccion_cliente_idx"]
    col_p = cfg["columnas"]["interaccion_producto_idx"]
    col_clv = cfg["columnas"]["clv_target"]
    col_c_cliente = cfg["columnas"]["cliente_idx"]

    # Mapa customer_idx → clv_log
    clv_map: dict[int, float] = dict(
        zip(clientes[col_c_cliente].tolist(), clientes[col_clv].tolist())
    )

    clientes_arr = interacciones[col_c].to_numpy(dtype=np.int64)
    productos_arr = interacciones[col_p].to_numpy(dtype=np.int64)
    clv_arr = np.array(
        [clv_map.get(int(c), 0.0) for c in clientes_arr], dtype=np.float32
    )

    # Positivos del split actual (usa la función canónica)
    pos_split = construir_positivos_por_cliente(interacciones, col_c, col_p)

    # Unión con positivos a excluir (para val/test)
    if positivos_excluir:
        pos_total: dict[int, set[int]] = {}
        for c, ps in pos_split.items():
            pos_total[c] = ps | positivos_excluir.get(c, set())
        for c, ps in positivos_excluir.items():
            if c not in pos_total:
                pos_total[c] = ps.copy()
    else:
        pos_total = pos_split

    k = cfg["entrenamiento"]["k_negativos"]

    return InteractionDataset(
        clientes_idx=clientes_arr,
        productos_idx=productos_arr,
        clv_targets=clv_arr,
        positivos_por_cliente=pos_total,
        n_productos=n_productos,
        k=k,
        popularidad=popularidad,
        semilla=semilla,
    )
