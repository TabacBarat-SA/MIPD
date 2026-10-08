"""
test_metrics.py – Tests de los Bloques 4 y 5: métricas de CLV y recomendación.

Tests con valores calculados a mano para verificar corrección de:
    - MAE, RMSE, Spearman (Bloque 4)
    - Precision@K, NDCG@K (Bloque 5)
    - Carga de embeddings .npy con forma correcta
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


# ─── Bloque 4: CLV Metrics ───────────────────────────────────────────────── #

class TestCLVMetrics:

    def test_mae_correcto(self):
        """MAE manual: sum(|pred - real|) / n."""
        from neural_network.evaluation.clv_metrics import evaluar_clv

        real = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.5, 1.5, 3.5])
        # MAE = (0.5 + 0.5 + 0.5) / 3 = 0.5
        resultado = evaluar_clv(real, pred)
        assert abs(resultado["modelo_log_todos"]["mae"] - 0.5) < 1e-4

    def test_rmse_correcto(self):
        """RMSE manual."""
        from neural_network.evaluation.clv_metrics import evaluar_clv

        real = np.array([0.0, 0.0, 1.0])
        pred = np.array([0.0, 1.0, 0.0])
        # MSE = (0 + 1 + 1) / 3 ≈ 0.6667; RMSE ≈ 0.8165
        resultado = evaluar_clv(real, pred)
        rmse_esperado = float(np.sqrt(np.mean((real - pred) ** 2)))
        assert abs(resultado["modelo_log_todos"]["rmse"] - rmse_esperado) < 1e-3

    def test_spearman_perfecto(self):
        """Correlación de Spearman = 1.0 para ranking perfecto."""
        from neural_network.evaluation.clv_metrics import evaluar_clv

        real = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([0.1, 0.2, 0.3, 0.4])  # mismo orden
        resultado = evaluar_clv(real, pred)
        assert abs(resultado["modelo_log_todos"]["spearman"] - 1.0) < 1e-4

    def test_metricas_clv_positivo_solo_cuando_hay(self):
        """Las métricas para CLV>0 solo se reportan si hay clientes con CLV>0."""
        from neural_network.evaluation.clv_metrics import evaluar_clv

        # Todos cero en escala original → expm1(0) = 0
        real = np.zeros(10)
        pred = np.zeros(10)
        resultado = evaluar_clv(real, pred)
        assert resultado["n_clientes_clv_positivo"] == 0
        assert "modelo_log_clv_positivo" not in resultado

    def test_guarda_json(self, tmp_path):
        """El JSON de métricas debe guardarse en la ruta indicada."""
        from neural_network.evaluation.clv_metrics import evaluar_clv
        import json

        real = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 2.1, 2.9])
        ruta = tmp_path / "metrics_clv.json"
        evaluar_clv(real, pred, ruta_salida=ruta)
        assert ruta.exists()
        with ruta.open() as f:
            data = json.load(f)
        assert "modelo_log_todos" in data

    def test_escala_original_expm1(self):
        """Las métricas en escala original deben ser sobre expm1(pred/real)."""
        from neural_network.evaluation.clv_metrics import evaluar_clv

        real_log = np.array([np.log1p(100.0), np.log1p(200.0)])
        pred_log = real_log.copy()  # predicción perfecta
        resultado = evaluar_clv(real_log, pred_log)
        assert resultado["modelo_original_todos"]["mae"] < 1e-4


# ─── Bloque 5: Rec Metrics ───────────────────────────────────────────────── #

class TestPrecisionNDCG:
    """Cálculo manual de Precision@K y NDCG@K."""

    def _precision_at_k(self, relevantes, ranking, k):
        from neural_network.evaluation.rec_metrics import _precision_at_k
        return _precision_at_k(set(relevantes), ranking, k)

    def _ndcg_at_k(self, relevantes, ranking, k):
        from neural_network.evaluation.rec_metrics import _ndcg_at_k
        return _ndcg_at_k(set(relevantes), ranking, k)

    def test_precision_perfecto(self):
        """Todos los top-K son relevantes → Precision@K = 1.0."""
        relevantes = {0, 1, 2}
        ranking = [0, 1, 2, 5, 6]
        assert self._precision_at_k(relevantes, ranking, k=3) == 1.0

    def test_precision_cero(self):
        """Ninguno de los top-K es relevante → Precision@K = 0.0."""
        relevantes = {10, 11, 12}
        ranking = [0, 1, 2, 3, 4]
        assert self._precision_at_k(relevantes, ranking, k=5) == 0.0

    def test_precision_parcial(self):
        """2 de 4 top-K son relevantes → Precision@4 = 0.5."""
        relevantes = {0, 2}
        ranking = [0, 1, 2, 3]
        p = self._precision_at_k(relevantes, ranking, k=4)
        assert abs(p - 0.5) < 1e-6

    def test_ndcg_perfecto(self):
        """Si el único relevante está en posición 1 → NDCG@K = 1.0."""
        relevantes = {0}
        ranking = [0, 1, 2, 3]
        assert abs(self._ndcg_at_k(relevantes, ranking, k=4) - 1.0) < 1e-6

    def test_ndcg_relevante_en_segunda_posicion(self):
        """Si el relevante está en posición 2, NDCG@K = 1/log2(3) / 1/log2(2)."""
        relevantes = {1}
        ranking = [0, 1, 2, 3]
        dcg = 1.0 / np.log2(3)   # posición i=1 → log2(1+2)=log2(3)
        idcg = 1.0 / np.log2(2)  # ideal: posición 0 → log2(2)
        esperado = dcg / idcg
        obtenido = self._ndcg_at_k(relevantes, ranking, k=4)
        assert abs(obtenido - esperado) < 1e-6

    def test_ndcg_cero(self):
        """Sin relevantes en el ranking → NDCG@K = 0.0."""
        relevantes = {99}
        ranking = [0, 1, 2, 3]
        assert self._ndcg_at_k(relevantes, ranking, k=4) == 0.0

    def test_ndcg_sin_relevantes_set_vacio(self):
        """NDCG con set de relevantes vacío → 0.0."""
        relevantes = set()
        ranking = [0, 1, 2]
        assert self._ndcg_at_k(relevantes, ranking, k=3) == 0.0


# ─── Bloque 5: Exportación de embeddings ─────────────────────────────────── #

class TestExportEmbeddings:

    def test_embeddings_shape_correcto(self, tmp_path):
        """Los .npy deben cargarse con shape (n, dim)."""
        from neural_network.models.multitask_net import MultiTaskNet
        from neural_network.export import exportar_embeddings

        n_c, n_p, dim = 50, 30, 8
        modelo = MultiTaskNet(
            n_clientes=n_c, n_productos=n_p, dim_embedding=dim,
            capas_tronco=[], dropout=0.0, usar_layer_norm=False, usar_features=False,
        )
        ruta_c = tmp_path / "customer_embeddings.npy"
        ruta_p = tmp_path / "product_embeddings.npy"
        emb_c, emb_p = exportar_embeddings(modelo, ruta_c, ruta_p)

        assert emb_c.shape == (n_c, dim), f"customer emb shape: {emb_c.shape}"
        assert emb_p.shape == (n_p, dim), f"product emb shape: {emb_p.shape}"

    def test_embeddings_se_pueden_recargar(self, tmp_path):
        """Los .npy guardados deben recargarse con la misma forma."""
        from neural_network.models.multitask_net import MultiTaskNet
        from neural_network.export import exportar_embeddings

        n_c, n_p, dim = 10, 15, 4
        modelo = MultiTaskNet(
            n_clientes=n_c, n_productos=n_p, dim_embedding=dim,
            capas_tronco=[], dropout=0.0, usar_layer_norm=False, usar_features=False,
        )
        ruta_c = tmp_path / "ce.npy"
        ruta_p = tmp_path / "pe.npy"
        exportar_embeddings(modelo, ruta_c, ruta_p)

        ce = np.load(ruta_c)
        pe = np.load(ruta_p)
        assert ce.shape == (n_c, dim)
        assert pe.shape == (n_p, dim)
