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

        # Almacena los negativos generados: lista de (cliente_idx, producto_idx)
        self._negativos_clientes: np.ndarray = np.empty(0, dtype=np.int64)
        self._negativos_productos: np.ndarray = np.empty(0, dtype=np.int64)
        self._clv_neg: np.ndarray = np.empty(0, dtype=np.float32)

        # Genera negativos iniciales
        self.regenerar_negativos()

    # ------------------------------------------------------------------ #
    #  Negative sampling                                                   #
    # ------------------------------------------------------------------ #

    def regenerar_negativos(self) -> None:
        """Genera o regenera los negativos.

        En modo ``semilla != None`` (val/test) usa la semilla fija → mismos
        negativos en cada época, garantizando comparabilidad.
        En modo train (``semilla=None``) usa estado aleatorio actual.
        """
        rng = (
            np.random.default_rng(self._semilla)
            if self._semilla is not None
            else np.random.default_rng()
        )

        neg_c_list: list[int] = []
        neg_p_list: list[int] = []
        neg_clv_list: list[float] = []

        for i in range(self._n_pos):
            c = int(self._clientes_pos[i])
            clv = float(self._clv_pos[i])
            positivos_c = self._positivos_por_cliente.get(c, set())

            # Candidatos: todos los productos menos los positivos del cliente
            candidatos = np.arange(self._n_productos, dtype=np.int64)
            mask = np.ones(self._n_productos, dtype=bool)
            for p in positivos_c:
                if 0 <= p < self._n_productos:
                    mask[p] = False
            candidatos = candidatos[mask]

            if len(candidatos) == 0:
                continue  # caso extremo: cliente compró todo

            # Probabilidades de muestreo
            if self._popularidad is not None and len(candidatos) > 0:
                probs = self._popularidad[candidatos]
                s = probs.sum()
                probs = probs / s if s > 0 else np.ones(len(candidatos)) / len(candidatos)
            else:
                probs = None

            n_neg = min(self._k, len(candidatos))
            elegidos = rng.choice(candidatos, size=n_neg, replace=False, p=probs)

            neg_c_list.extend([c] * n_neg)
            neg_p_list.extend(elegidos.tolist())
            neg_clv_list.extend([clv] * n_neg)

        self._negativos_clientes = np.array(neg_c_list, dtype=np.int64)
        self._negativos_productos = np.array(neg_p_list, dtype=np.int64)
        self._clv_neg = np.array(neg_clv_list, dtype=np.float32)

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


# ─── Funciones auxiliares de construcción ──────────────────────────────────── #

def construir_positivos_por_cliente(
    clientes_idx: np.ndarray,
    productos_idx: np.ndarray,
) -> dict[int, set[int]]:
    """Construye el mapa {customer_idx → set(product_idx)} a partir de arrays."""
    positivos: dict[int, set[int]] = {}
    for c, p in zip(clientes_idx, productos_idx):
        positivos.setdefault(int(c), set()).add(int(p))
    return positivos


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

    # Positivos del split actual
    pos_split = construir_positivos_por_cliente(clientes_arr, productos_arr)

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
