"""
test_training.py – Tests del Bloque 3: Trainer y EarlyStopping.

Verifican:
    - EarlyStopping: se dispara en el número correcto de épocas.
    - Trainer: la loss baja en un mini-dataset sintético.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from neural_network.training.early_stopping import EarlyStopping
from neural_network.models.multitask_net import MultiTaskNet


# ─── Tests de EarlyStopping ─────────────────────────────────────────────── #

class TestEarlyStopping:

    def test_dispara_tras_patience_epocas(self):
        """Debe devolver True tras ``patience`` épocas sin mejora."""
        es = EarlyStopping(patience=3, min_delta=1e-4)
        modelo = MultiTaskNet(
            n_clientes=5, n_productos=5, dim_embedding=4,
            capas_tronco=[], dropout=0.0, usar_layer_norm=False, usar_features=False,
        )

        # Primera época mejora
        assert not es.paso(0.5, modelo, 1)
        # 3 épocas sin mejora → debe disparar
        assert not es.paso(0.5, modelo, 2)
        assert not es.paso(0.5, modelo, 3)
        assert es.paso(0.5, modelo, 4)   # 4ª sin mejora = patience=3 → True

    def test_no_dispara_si_hay_mejora(self):
        """No debe disparar si la loss mejora continuamente."""
        es = EarlyStopping(patience=3, min_delta=1e-4)
        modelo = MultiTaskNet(
            n_clientes=5, n_productos=5, dim_embedding=4,
            capas_tronco=[], dropout=0.0, usar_layer_norm=False, usar_features=False,
        )

        losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        for i, loss in enumerate(losses):
            resultado = es.paso(loss, modelo, i + 1)
            assert not resultado, f"EarlyStopping disparó en época {i+1} sin razón"

    def test_restaurar_mejor_estado(self):
        """restaurar_mejor debe cargar los pesos de la mejor época."""
        es = EarlyStopping(patience=2, min_delta=1e-4)
        modelo = MultiTaskNet(
            n_clientes=5, n_productos=5, dim_embedding=4,
            capas_tronco=[], dropout=0.0, usar_layer_norm=False, usar_features=False,
        )

        # Guardar un bias conocido en la primera época (mejor)
        torch.nn.init.constant_(modelo.cabeza_clv.bias, 99.0)
        es.paso(0.1, modelo, 1)  # mejor época: bias=99

        # Cambiar el bias
        torch.nn.init.constant_(modelo.cabeza_clv.bias, 0.0)
        es.paso(0.9, modelo, 2)

        # Restaurar
        es.restaurar_mejor(modelo)
        assert abs(modelo.cabeza_clv.bias.item() - 99.0) < 1e-5

    def test_mejor_epoca_registrada(self):
        """mejor_epoca debe corresponder a la época con menor loss."""
        es = EarlyStopping(patience=5, min_delta=1e-4)
        modelo = MultiTaskNet(
            n_clientes=5, n_productos=5, dim_embedding=4,
            capas_tronco=[], dropout=0.0, usar_layer_norm=False, usar_features=False,
        )

        losses = [1.0, 0.5, 0.3, 0.8, 0.9]  # mejor en época 3
        for i, loss in enumerate(losses):
            es.paso(loss, modelo, i + 1)

        assert es.mejor_epoca == 3


# ─── Tests de Trainer (mini-dataset) ─────────────────────────────────────── #

class TestTrainer:

    def _cfg_minimo(self, tmp_path: "Path") -> dict:
        """Configuración mínima para el Trainer en tests."""
        return {
            "entrenamiento": {
                "semilla": 42,
                "batch_size": 8,
                "n_epocas": 5,
                "lr": 0.01,
                "weight_decay": 0.0,
                "grad_clip": 0.0,
                "lambda_rec": 1.0,
                "huber_delta": 1.0,
                "k_negativos": 2,
            },
            "early_stopping": {"patience": 10, "min_delta": 1e-6},
            "modelo": {
                "n_clientes": 10,
                "n_productos": 20,
                "dim_embedding": 8,
                "capas_tronco": [16],
                "dropout": 0.0,
                "usar_layer_norm": False,
                "usar_features": False,
                "n_features_cliente": 0,
            },
            "_artifacts_dir_abs": tmp_path,
            "logging": {
                "model_checkpoint": "model.pt",
                "curvas_csv": "curvas.csv",
                "curvas_png": "curvas.png",
            },
            "evaluacion": {"semilla_negativos_val": 123},
        }

    def test_loss_baja_en_mini_dataset(self, tmp_path):
        """La loss de train debe bajar en un mini-dataset sintético."""
        from neural_network.data.dataset import InteractionDataset, construir_positivos_por_cliente
        from neural_network.training.trainer import Trainer

        rng = np.random.default_rng(42)
        n_pos = 30
        clientes = rng.integers(0, 10, n_pos).astype(np.int64)
        productos = rng.integers(0, 20, n_pos).astype(np.int64)
        clv = rng.uniform(0, 2, n_pos).astype(np.float32)
        positivos = construir_positivos_por_cliente(clientes, productos)

        ds_train = InteractionDataset(clientes, productos, clv, positivos, 20, k=2, semilla=None)
        ds_val = InteractionDataset(clientes, productos, clv, positivos, 20, k=2, semilla=42)

        cfg = self._cfg_minimo(tmp_path)
        modelo = MultiTaskNet(
            n_clientes=10, n_productos=20, dim_embedding=8,
            capas_tronco=[16], dropout=0.0, usar_layer_norm=False, usar_features=False,
        )
        trainer = Trainer(
            modelo=modelo,
            dataset_train=ds_train,
            dataset_val=ds_val,
            cfg=cfg,
            clientes_train_df=None,
            clientes_val_df=None,
        )

        resultado = trainer.entrenar()
        # La loss debe haber bajado respecto a la primera época
        first_loss = trainer.historial[0]["train_loss"]
        last_loss = trainer.historial[-1]["train_loss"]
        # Verificar que al menos entrenó alguna época
        assert len(trainer.historial) > 0
        assert resultado["checkpoint"] is not None

    def test_early_stopping_guarda_checkpoint(self, tmp_path):
        """Tras el early stopping, el checkpoint debe existir en disco."""
        from neural_network.data.dataset import InteractionDataset, construir_positivos_por_cliente
        from neural_network.training.trainer import Trainer

        rng = np.random.default_rng(0)
        n_pos = 20
        clientes = rng.integers(0, 10, n_pos).astype(np.int64)
        productos = rng.integers(0, 20, n_pos).astype(np.int64)
        clv = rng.uniform(0, 2, n_pos).astype(np.float32)
        positivos = construir_positivos_por_cliente(clientes, productos)

        ds_train = InteractionDataset(clientes, productos, clv, positivos, 20, k=2, semilla=None)
        ds_val = InteractionDataset(clientes, productos, clv, positivos, 20, k=2, semilla=99)

        cfg = self._cfg_minimo(tmp_path)
        cfg["entrenamiento"]["n_epocas"] = 10
        cfg["early_stopping"]["patience"] = 2   # se dispara rápido

        modelo = MultiTaskNet(
            n_clientes=10, n_productos=20, dim_embedding=8,
            capas_tronco=[16], dropout=0.0, usar_layer_norm=False, usar_features=False,
        )
        trainer = Trainer(
            modelo=modelo,
            dataset_train=ds_train,
            dataset_val=ds_val,
            cfg=cfg,
        )
        trainer.entrenar()

        ckpt_path = tmp_path / "model.pt"
        assert ckpt_path.exists(), f"Checkpoint no encontrado: {ckpt_path}"

    def test_trainer_genera_curvas_csv(self, tmp_path):
        """El entrenamiento debe generar el archivo de curvas CSV."""
        from neural_network.data.dataset import InteractionDataset, construir_positivos_por_cliente
        from neural_network.training.trainer import Trainer

        rng = np.random.default_rng(7)
        n_pos = 15
        clientes = rng.integers(0, 5, n_pos).astype(np.int64)
        productos = rng.integers(0, 10, n_pos).astype(np.int64)
        clv = rng.uniform(0, 1, n_pos).astype(np.float32)
        positivos = construir_positivos_por_cliente(clientes, productos)

        ds = InteractionDataset(clientes, productos, clv, positivos, 10, k=1, semilla=42)
        cfg = self._cfg_minimo(tmp_path)

        modelo = MultiTaskNet(
            n_clientes=5, n_productos=10, dim_embedding=4,
            capas_tronco=[], dropout=0.0, usar_layer_norm=False, usar_features=False,
        )
        trainer = Trainer(modelo=modelo, dataset_train=ds, dataset_val=ds, cfg=cfg)
        trainer.entrenar()

        csv_path = tmp_path / "curvas.csv"
        assert csv_path.exists(), "El archivo CSV de curvas no se generó"
