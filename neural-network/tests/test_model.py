"""
test_model.py – Tests del Bloque 2: MultiTaskNet.

Verifican shapes de salida, backward funcional y métodos auxiliares.
"""
from __future__ import annotations

import pytest
import torch

from neural_network.models.multitask_net import MultiTaskNet


# ─── Fixtures ────────────────────────────────────────────────────────────── #

@pytest.fixture
def modelo_pequeño():
    """Modelo pequeño para tests rápidos."""
    return MultiTaskNet(
        n_clientes=10,
        n_productos=15,
        dim_embedding=8,
        capas_tronco=[16, 8],
        dropout=0.0,   # sin dropout para reproducibilidad
        usar_layer_norm=True,
        usar_features=False,
        n_features_cliente=0,
    )


@pytest.fixture
def modelo_con_features():
    """Modelo con features de cliente."""
    return MultiTaskNet(
        n_clientes=10,
        n_productos=15,
        dim_embedding=8,
        capas_tronco=[16, 8],
        dropout=0.0,
        usar_layer_norm=True,
        usar_features=True,
        n_features_cliente=9,
    )


@pytest.fixture
def batch_sintetico():
    """Batch de tamaño 4 para tests."""
    batch_size = 4
    clientes = torch.randint(0, 10, (batch_size,))
    productos = torch.randint(0, 15, (batch_size,))
    return clientes, productos, batch_size


# ─── Tests ───────────────────────────────────────────────────────────────── #

class TestMultiTaskNetShapes:

    def test_forward_shape_sin_features(self, modelo_pequeño, batch_sintetico):
        """forward debe devolver dos tensores de shape (B,)."""
        c, p, B = batch_sintetico
        clv_pred, rec_logit = modelo_pequeño(c, p)
        assert clv_pred.shape == (B,), f"clv_pred.shape={clv_pred.shape}, esperado ({B},)"
        assert rec_logit.shape == (B,), f"rec_logit.shape={rec_logit.shape}, esperado ({B},)"

    def test_forward_shape_con_features(self, modelo_con_features, batch_sintetico):
        """forward con features debe dar las mismas shapes."""
        c, p, B = batch_sintetico
        feats = torch.randn(B, 9)
        clv_pred, rec_logit = modelo_con_features(c, p, feats)
        assert clv_pred.shape == (B,)
        assert rec_logit.shape == (B,)

    def test_get_customer_embeddings_shape(self, modelo_pequeño):
        emb = modelo_pequeño.get_customer_embeddings()
        assert emb.shape == (10, 8), f"Shape esperado (10, 8), obtenido {emb.shape}"

    def test_get_product_embeddings_shape(self, modelo_pequeño):
        emb = modelo_pequeño.get_product_embeddings()
        assert emb.shape == (15, 8), f"Shape esperado (15, 8), obtenido {emb.shape}"

    def test_score_all_products_shape(self, modelo_pequeño):
        scores = modelo_pequeño.score_all_products(customer_idx=0)
        assert scores.shape == (15,), f"Shape esperado (15,), obtenido {scores.shape}"

    def test_embeddings_no_en_cuda(self, modelo_pequeño):
        """get_customer_embeddings debe devolver siempre tensor en CPU."""
        emb = modelo_pequeño.get_customer_embeddings()
        assert emb.device.type == "cpu"


class TestMultiTaskNetBackward:

    def test_backward_sin_features(self, modelo_pequeño, batch_sintetico):
        """El gradiente debe fluir correctamente a todos los parámetros."""
        c, p, B = batch_sintetico
        clv_target = torch.randn(B)
        etiqueta = torch.randint(0, 2, (B,)).float()

        criterio_clv = torch.nn.HuberLoss()
        criterio_rec = torch.nn.BCEWithLogitsLoss()

        clv_pred, rec_logit = modelo_pequeño(c, p)
        loss = criterio_clv(clv_pred, clv_target) + criterio_rec(rec_logit, etiqueta)
        loss.backward()

        # Verificar que los embeddings tienen gradiente
        assert modelo_pequeño.emb_cliente.weight.grad is not None
        assert modelo_pequeño.emb_producto.weight.grad is not None

    def test_backward_con_features(self, modelo_con_features, batch_sintetico):
        """Backward debe funcionar también con features."""
        c, p, B = batch_sintetico
        feats = torch.randn(B, 9)
        clv_target = torch.randn(B)
        etiqueta = torch.randint(0, 2, (B,)).float()

        clv_pred, rec_logit = modelo_con_features(c, p, feats)
        loss = torch.nn.HuberLoss()(clv_pred, clv_target) + \
               torch.nn.BCEWithLogitsLoss()(rec_logit, etiqueta)
        loss.backward()

        assert modelo_con_features.emb_cliente.weight.grad is not None

    def test_backward_no_nan(self, modelo_pequeño, batch_sintetico):
        """La loss y los gradientes no deben ser NaN."""
        c, p, B = batch_sintetico
        clv_target = torch.randn(B)
        etiqueta = torch.randint(0, 2, (B,)).float()

        clv_pred, rec_logit = modelo_pequeño(c, p)
        loss = torch.nn.HuberLoss()(clv_pred, clv_target) + \
               torch.nn.BCEWithLogitsLoss()(rec_logit, etiqueta)
        loss.backward()

        assert not torch.isnan(loss), "La loss es NaN"
        for name, param in modelo_pequeño.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"Gradiente NaN en {name}"


class TestMultiTaskNetConfig:

    def test_desde_config_crea_modelo(self):
        """desde_config debe crear un modelo con los tamaños correctos."""
        cfg = {
            "modelo": {
                "n_clientes": 100,
                "n_productos": 50,
                "dim_embedding": 16,
                "capas_tronco": [32, 16],
                "dropout": 0.1,
                "usar_layer_norm": True,
                "usar_features": False,
                "n_features_cliente": 0,
            }
        }
        modelo = MultiTaskNet.desde_config(cfg)
        assert modelo.n_clientes == 100
        assert modelo.n_productos == 50
        assert modelo.dim_embedding == 16

    def test_sin_tronco(self):
        """capas_tronco=[] debe funcionar (sin tronco compartido)."""
        modelo = MultiTaskNet(
            n_clientes=5, n_productos=10, dim_embedding=4,
            capas_tronco=[], dropout=0.0, usar_layer_norm=False,
            usar_features=False,
        )
        c = torch.tensor([0, 1])
        p = torch.tensor([3, 7])
        clv, rec = modelo(c, p)
        assert clv.shape == (2,)
        assert rec.shape == (2,)
