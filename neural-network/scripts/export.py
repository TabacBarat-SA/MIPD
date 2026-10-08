"""
export.py – Exportación de predicciones y embeddings (Bloque 5).

Genera en artifacts/:
    - predictions_clv.csv
    - predictions_rec.csv
    - customer_embeddings.npy
    - product_embeddings.npy
    - customer_idx_to_id.json
    - product_idx_to_id.json

Ejecuta con Run desde el IDE (sin argumentos).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR / "src"))


def main(config_path: Path | None = None) -> None:
    """Exporta predicciones y embeddings del modelo entrenado."""
    import numpy as np
    import torch
    from neural_network.config import cargar_config, get_artifact_path
    from neural_network.data.preprocessing import cargar_parquets
    from neural_network.models.multitask_net import MultiTaskNet
    from neural_network.training.trainer import _construir_features_map
    from neural_network.pipeline import _inferir_clv
    from neural_network.export import (
        exportar_embeddings,
        exportar_predicciones_clv,
        exportar_predicciones_rec,
        exportar_mapeos,
    )

    cfg = cargar_config(config_path)
    print("=" * 60)
    print("  EXPORT – Predicciones y Embeddings")
    print("=" * 60)

    ruta_ckpt = get_artifact_path(cfg, "model_checkpoint")
    if not ruta_ckpt.exists():
        print(f"[ERROR] Checkpoint no encontrado: {ruta_ckpt}")
        print("  Ejecuta primero train.py")
        return

    ckpt = torch.load(ruta_ckpt, map_location="cpu", weights_only=False)
    modelo = MultiTaskNet.desde_config(cfg)
    modelo.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device).eval()

    datos = cargar_parquets(cfg)
    col_c = cfg["columnas"]["cliente_idx"]
    col_c_i = cfg["columnas"]["interaccion_cliente_idx"]
    col_p_i = cfg["columnas"]["interaccion_producto_idx"]
    artifacts_dir = cfg["_artifacts_dir_abs"]

    # Embeddings
    exportar_embeddings(
        modelo=modelo,
        ruta_customer_emb=get_artifact_path(cfg, "customer_embeddings_npy"),
        ruta_product_emb=get_artifact_path(cfg, "product_embeddings_npy"),
    )

    # CLV
    c_test = datos["clientes_test"][col_c].to_numpy()
    features_test = _construir_features_map(datos["clientes_test"], cfg, device)
    clv_pred = _inferir_clv(modelo, c_test, features_test, cfg, device, cfg["modelo"]["n_productos"])
    exportar_predicciones_clv(
        clientes_test=datos["clientes_test"],
        clv_log_pred=clv_pred,
        ruta_salida=get_artifact_path(cfg, "predictions_clv_csv"),
        cfg=cfg,
    )

    # Rec
    pos_train: dict = {}
    for c, p in zip(datos["interacciones_train"][col_c_i].tolist(),
                    datos["interacciones_train"][col_p_i].tolist()):
        pos_train.setdefault(int(c), set()).add(int(p))
    pos_val: dict = {}
    for c, p in zip(datos["interacciones_val"][col_c_i].tolist(),
                    datos["interacciones_val"][col_p_i].tolist()):
        pos_val.setdefault(int(c), set()).add(int(p))

    pos_excluidos = {c: pos_train.get(c, set()) | pos_val.get(c, set())
                     for c in set(pos_train) | set(pos_val)}
    features_train = _construir_features_map(datos["clientes_train"], cfg, device)

    exportar_predicciones_rec(
        modelo=modelo,
        interacciones_test=datos["interacciones_test"],
        positivos_excluidos=pos_excluidos,
        cfg=cfg,
        ruta_salida=get_artifact_path(cfg, "predictions_rec_csv"),
        k_max=max(cfg["evaluacion"]["k_valores"]),
        features_map=features_train,
    )

    exportar_mapeos(
        interacciones_train=datos["interacciones_train"],
        productos=datos["productos"],
        cfg=cfg,
        artifacts_dir=artifacts_dir,
    )

    print(f"\n  Exportación completada. Artifacts en: {artifacts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    main(config_path=args.config)
