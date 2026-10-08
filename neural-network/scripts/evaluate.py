"""
evaluate.py – Bloques 4-5: métricas de CLV y recomendación.

Carga el checkpoint guardado y evalúa sobre el split test.
Genera: artifacts/metrics_clv.json, artifacts/metrics_rec.json
Ejecuta con Run desde el IDE (sin argumentos).
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
    """Evalúa el modelo sobre el split test (Bloques 4-5)."""
    import numpy as np
    import torch
    from neural_network.config import cargar_config, get_artifact_path
    from neural_network.data.preprocessing import cargar_parquets
    from neural_network.models.multitask_net import MultiTaskNet
    from neural_network.evaluation.clv_metrics import evaluar_clv
    from neural_network.evaluation.rec_metrics import evaluar_recomendacion
    from neural_network.training.trainer import _construir_features_map

    cfg = cargar_config(config_path)
    print("=" * 60)
    print("  EVALUATE – Bloques 4-5")
    print("=" * 60)

    # Cargar checkpoint
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
    print(f"  Checkpoint cargado (época {ckpt.get('mejor_epoca', '?')})")

    datos = cargar_parquets(cfg)
    col_c = cfg["columnas"]["cliente_idx"]
    col_c_i = cfg["columnas"]["interaccion_cliente_idx"]
    col_p_i = cfg["columnas"]["interaccion_producto_idx"]

    # ── Bloque 4: CLV ────────────────────────────────────────────────── #
    print("\n[Bloque 4] Métricas CLV...")
    from neural_network.pipeline import _inferir_clv

    c_test = datos["clientes_test"][col_c].to_numpy()
    features_test = _construir_features_map(datos["clientes_test"], cfg, device)
    clv_pred = _inferir_clv(modelo, c_test, features_test, cfg, device, cfg["modelo"]["n_productos"])
    clv_real = datos["clientes_test"][cfg["columnas"]["clv_target"]].to_numpy(dtype=np.float64)

    evaluar_clv(
        clv_log_real=clv_real,
        clv_log_pred=clv_pred,
        clv_real_train=datos["clientes_train"][cfg["columnas"]["clv_target"]].to_numpy(dtype=np.float64),
        gasto_historico=datos["clientes_test"]["importe_total"].to_numpy(dtype=np.float64)
            if "importe_total" in datos["clientes_test"].columns else None,
        cfg=cfg,
        ruta_salida=get_artifact_path(cfg, "metrics_clv_json"),
    )

    # ── Bloque 5: Recomendación ──────────────────────────────────────── #
    print("\n[Bloque 5] Métricas de recomendación...")
    pos_train: dict = {}
    for c, p in zip(datos["interacciones_train"][col_c_i].tolist(),
                    datos["interacciones_train"][col_p_i].tolist()):
        pos_train.setdefault(int(c), set()).add(int(p))

    pos_val: dict = {}
    for c, p in zip(datos["interacciones_val"][col_c_i].tolist(),
                    datos["interacciones_val"][col_p_i].tolist()):
        pos_val.setdefault(int(c), set()).add(int(p))

    train_vistos = set(datos["interacciones_train"][col_c_i].unique())
    features_train = _construir_features_map(datos["clientes_train"], cfg, device)

    evaluar_recomendacion(
        modelo=modelo,
        interacciones_test=datos["interacciones_test"],
        positivos_train=pos_train,
        positivos_val=pos_val,
        clientes_train_vistos=train_vistos,
        k_valores=cfg["evaluacion"]["k_valores"],
        cfg=cfg,
        interacciones_train=datos["interacciones_train"],
        features_map=features_train,
        ruta_salida=get_artifact_path(cfg, "metrics_rec_json"),
    )

    print("\n  Evaluación completada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    main(config_path=args.config)
