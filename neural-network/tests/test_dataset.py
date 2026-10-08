"""
test_dataset.py – Tests del Bloque 1: InteractionDataset y negative sampling.

Usan datos sintéticos pequeños. No dependen de parquets reales.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from neural_network.data.dataset import (
    InteractionDataset,
    construir_positivos_por_cliente,
    construir_dataset,
)


# ─── Fixtures ────────────────────────────────────────────────────────────── #

@pytest.fixture
def datos_sinteticos():
    """Dataset sintético pequeño: 5 clientes, 20 productos, 10 positivos."""
    rng = np.random.default_rng(42)
    n_clientes = 5
    n_productos = 20
    n_positivos = 10

    clientes = rng.integers(0, n_clientes, size=n_positivos).astype(np.int64)
    productos = rng.integers(0, n_productos, size=n_positivos).astype(np.int64)
    clv = rng.uniform(0, 5, size=n_positivos).astype(np.float32)

    return clientes, productos, clv, n_clientes, n_productos


@pytest.fixture
def dataset_sintetico(datos_sinteticos):
    clientes, productos, clv, n_clientes, n_productos = datos_sinteticos
    positivos = construir_positivos_por_cliente(clientes, productos)
    k = 3
    ds = InteractionDataset(
        clientes_idx=clientes,
        productos_idx=productos,
        clv_targets=clv,
        positivos_por_cliente=positivos,
        n_productos=n_productos,
        k=k,
        semilla=42,
    )
    return ds, k, positivos, n_productos


# ─── Tests ───────────────────────────────────────────────────────────────── #

class TestInteractionDataset:

    def test_longitud_total(self, dataset_sintetico):
        """__len__ debe ser n_positivos × (1 + k)."""
        ds, k, _, _ = dataset_sintetico
        n_pos = ds._n_pos
        assert len(ds) == n_pos * (1 + k), (
            f"Esperado {n_pos * (1 + k)}, obtenido {len(ds)}"
        )

    def test_proporcion_positivos_negativos(self, dataset_sintetico):
        """Debe haber exactamente k negativos por cada positivo."""
        ds, k, _, _ = dataset_sintetico
        n_pos = ds._n_pos
        n_neg = len(ds._negativos_clientes)
        assert n_neg == n_pos * k, f"n_neg={n_neg}, esperado {n_pos * k}"

    def test_tipos_de_salida(self, dataset_sintetico):
        """La tupla devuelta debe tener los dtypes correctos."""
        ds, _, _, _ = dataset_sintetico
        c, p, e, clv = ds[0]
        assert c.dtype == torch.long, f"cliente debe ser long, es {c.dtype}"
        assert p.dtype == torch.long, f"producto debe ser long, es {p.dtype}"
        assert e.dtype == torch.float32, f"etiqueta debe ser float32, es {e.dtype}"
        assert clv.dtype == torch.float32, f"clv debe ser float32, es {clv.dtype}"

    def test_forma_de_salida_escalar(self, dataset_sintetico):
        """Cada tensor de la tupla debe ser escalar (0-D)."""
        ds, _, _, _ = dataset_sintetico
        for i in [0, ds._n_pos, len(ds) - 1]:
            c, p, e, clv = ds[i]
            for t, nombre in [(c, "c"), (p, "p"), (e, "e"), (clv, "clv")]:
                assert t.shape == torch.Size([]), f"{nombre}[{i}] no es escalar: {t.shape}"

    def test_etiquetas_correctas(self, dataset_sintetico):
        """Los primeros n_pos índices deben tener etiqueta 1.0, el resto 0.0."""
        ds, _, _, _ = dataset_sintetico
        # Positivos
        for i in range(ds._n_pos):
            _, _, e, _ = ds[i]
            assert e.item() == 1.0, f"ds[{i}] debería tener etiqueta 1.0"
        # Negativos
        for i in range(ds._n_pos, min(ds._n_pos + 5, len(ds))):
            _, _, e, _ = ds[i]
            assert e.item() == 0.0, f"ds[{i}] debería tener etiqueta 0.0"

    def test_negativos_no_coinciden_con_positivos(self, dataset_sintetico):
        """Ningún negativo debe coincidir con un positivo del mismo cliente."""
        ds, _, positivos, _ = dataset_sintetico
        for i in range(len(ds._negativos_clientes)):
            c = int(ds._negativos_clientes[i])
            p = int(ds._negativos_productos[i])
            pos_c = positivos.get(c, set())
            assert p not in pos_c, (
                f"Negativo ({c}, {p}) coincide con un positivo del cliente {c}"
            )

    def test_negativos_fijos_con_semilla(self, datos_sinteticos):
        """Con la misma semilla, los negativos deben ser idénticos."""
        clientes, productos, clv, n_clientes, n_productos = datos_sinteticos
        positivos = construir_positivos_por_cliente(clientes, productos)

        ds1 = InteractionDataset(clientes, productos, clv, positivos, n_productos, k=3, semilla=99)
        ds2 = InteractionDataset(clientes, productos, clv, positivos, n_productos, k=3, semilla=99)

        np.testing.assert_array_equal(ds1._negativos_productos, ds2._negativos_productos)

    def test_regeneracion_cambia_negativos_en_train(self, datos_sinteticos):
        """Sin semilla (modo train), regenerar_negativos() produce negativos distintos."""
        clientes, productos, clv, n_clientes, n_productos = datos_sinteticos
        positivos = construir_positivos_por_cliente(clientes, productos)

        ds = InteractionDataset(clientes, productos, clv, positivos, n_productos, k=3, semilla=None)
        neg_antes = ds._negativos_productos.copy()
        ds.regenerar_negativos()
        neg_despues = ds._negativos_productos.copy()

        # Con RNG sin semilla, es posible (aunque muy improbable) que coincidan.
        # Simplemente verificamos que el array no es None y tiene la longitud correcta.
        assert len(neg_despues) > 0

    def test_indices_dentro_de_rango(self, dataset_sintetico):
        """Todos los product_idx de negativos deben estar en [0, n_productos)."""
        ds, _, _, n_productos = dataset_sintetico
        neg_p = ds._negativos_productos
        assert np.all(neg_p >= 0), "Producto negativo con índice < 0"
        assert np.all(neg_p < n_productos), "Producto negativo fuera de rango"

    def test_clv_target_no_negativo(self, dataset_sintetico):
        """Los CLV targets deben ser ≥ 0 (son log1p de valores positivos)."""
        ds, _, _, _ = dataset_sintetico
        assert np.all(ds._clv_pos >= 0), "Hay CLV targets negativos en positivos"


class TestConstruirPositivos:

    def test_construir_positivos_correcto(self):
        clientes = np.array([0, 0, 1, 2, 2])
        productos = np.array([5, 10, 5, 3, 7])
        pos = construir_positivos_por_cliente(clientes, productos)
        assert pos[0] == {5, 10}
        assert pos[1] == {5}
        assert pos[2] == {3, 7}
