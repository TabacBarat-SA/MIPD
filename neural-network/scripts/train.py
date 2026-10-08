"""
train.py – Bloques 1-3: dataset + modelo + entrenamiento.

Ejecuta con Run desde el IDE (sin argumentos).
Genera: artifacts/model.pt, training_curves.csv, training_curves.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR / "src"))

DEFAULT_CONFIG = MODULE_DIR / "configs" / "config.yaml"


def main(config_path: Path | None = None) -> None:
    """Entrena la red neuronal multi-tarea (Bloques 1-3)."""
    import numpy as np
    import torch
    from neural_network.config import cargar_config
    from neural_network.data.preprocessing import cargar_todo, calcular_popularidad_productos
    from neural_network.data.dataset import construir_dataset
    from neural_network.models.multitask_net import MultiTaskNet
    from neural_network.training.trainer import Trainer

    cfg = cargar_config(config_path)
    print("=" * 60)
    print("  TRAIN – Bloques 1-3")
    print("=" * 60)

    # ── Bloque 0: Carga ─────────────────────────────────────────────── #
    print("\n[Bloque 0] Cargando datos...")
    datos, stats_cs, stats_clv = cargar_todo(cfg)

    n_productos = cfg["modelo"]["n_productos"]
    col_c = cfg["columnas"]["interaccion_cliente_idx"]
    col_p = cfg["columnas"]["interaccion_producto_idx"]

    pos_train: dict = {}
    for c, p in zip(datos["interacciones_train"][col_c].tolist(),
                    datos["interacciones_train"][col_p].tolist()):
        pos_train.setdefault(int(c), set()).add(int(p))

    popularidad = calcular_popularidad_productos(datos["interacciones_train"], n_productos, cfg)

    # ── Bloque 1: Datasets ──────────────────────────────────────────── #
    print("\n[Bloque 1] Construyendo datasets...")
    semilla_val = cfg["evaluacion"]["semilla_negativos_val"]
    ds_train = construir_dataset(
        datos["interacciones_train"], datos["clientes_train"],
        None, n_productos, cfg, popularidad, semilla=None,
    )
    ds_val = construir_dataset(
        datos["interacciones_val"], datos["clientes_val"],
        pos_train, n_productos, cfg, popularidad, semilla=semilla_val,
    )
    print(f"  train: {len(ds_train):,} | val: {len(ds_val):,}")

    # ── Bloque 2: Modelo ─────────────────────────────────────────────── #
    print("\n[Bloque 2] Creando MultiTaskNet...")
    modelo = MultiTaskNet.desde_config(cfg)
    print(f"  Parámetros: {sum(p.numel() for p in modelo.parameters() if p.requires_grad):,}")

    # ── Bloque 3: Entrenamiento ──────────────────────────────────────── #
    print("\n[Bloque 3] Entrenando...")
    trainer = Trainer(
        modelo=modelo,
        dataset_train=ds_train,
        dataset_val=ds_val,
        cfg=cfg,
        clientes_train_df=datos["clientes_train"],
        clientes_val_df=datos["clientes_val"],
    )
    resultado = trainer.entrenar()
    print(f"\n  Entrenamiento completado. Checkpoint: {resultado['checkpoint']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    main(config_path=args.config)
