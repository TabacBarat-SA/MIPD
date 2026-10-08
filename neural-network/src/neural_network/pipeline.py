"""
Pipeline principal – orquesta los 5 bloques en orden.

Uso:
    from neural_network.pipeline import run_pipeline
    run_pipeline()                          # usa config por defecto
    run_pipeline(config_path=Path("..."))   # config personalizada

Bloques:
    0. Carga parquets, verifica anti-fuga, estadísticas cold-start y CLV
    1. Construcción de datasets con negative sampling
    2. Construcción del modelo MultiTaskNet
    3. Entrenamiento con early stopping
    4. Evaluación de CLV
    5. Evaluación de recomendación + exportación
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def run_pipeline(config_path: Path | None = None) -> dict[str, Any]:
    """Ejecuta el pipeline completo de entrenamiento y evaluación.

    Parámetros
    ----------
    config_path : Path | None
        Ruta al config.yaml. Si None, usa el valor por defecto del módulo.

    Devuelve
    -------
    dict
        Resumen con rutas de artefactos generados y métricas finales.
    """
    from neural_network.config import cargar_config, get_artifact_path
    from neural_network.data.preprocessing import (
        cargar_todo,
        calcular_popularidad_productos,
    )
    from neural_network.data.dataset import construir_dataset
    from neural_network.models.multitask_net import MultiTaskNet
    from neural_network.training.trainer import Trainer, _construir_features_map
    from neural_network.evaluation.clv_metrics import evaluar_clv
    from neural_network.evaluation.rec_metrics import evaluar_recomendacion
    from neural_network.export import (
        exportar_embeddings,
        exportar_predicciones_clv,
        exportar_predicciones_rec,
        exportar_mapeos,
    )

    print("=" * 60)
    print("  PIPELINE – MultiTaskNet (CLV + Recomendación)")
    print("=" * 60)

    # ─── BLOQUE 0: Carga y verificación ─────────────────────────────── #
    print("\n[BLOQUE 0] Carga de datos y verificaciones...")
    cfg = cargar_config(config_path)
    datos, stats_cs, stats_clv = cargar_todo(cfg)

    n_productos = cfg["modelo"]["n_productos"]
    col_c = cfg["columnas"]["interaccion_cliente_idx"]
    col_p = cfg["columnas"]["interaccion_producto_idx"]

    # Positivos por cliente en cada split (para exclusión en negative sampling)
    pos_train = _construir_positivos_desde_df(datos["interacciones_train"], col_c, col_p)
    pos_val   = _construir_positivos_desde_df(datos["interacciones_val"], col_c, col_p)
    pos_test  = _construir_positivos_desde_df(datos["interacciones_test"], col_c, col_p)

    # Popularidad para muestreo sesgado (solo train)
    popularidad = calcular_popularidad_productos(
        datos["interacciones_train"], n_productos, cfg
    )

    # ─── BLOQUE 1: Datasets ──────────────────────────────────────────── #
    print("\n[BLOQUE 1] Construcción de datasets...")
    semilla_val = cfg["evaluacion"]["semilla_negativos_val"]

    ds_train = construir_dataset(
        interacciones=datos["interacciones_train"],
        clientes=datos["clientes_train"],
        positivos_excluir=None,
        n_productos=n_productos,
        cfg=cfg,
        popularidad=popularidad,
        semilla=None,   # negativos dinámicos en train
    )
    ds_val = construir_dataset(
        interacciones=datos["interacciones_val"],
        clientes=datos["clientes_val"],
        positivos_excluir=pos_train,
        n_productos=n_productos,
        cfg=cfg,
        popularidad=popularidad,
        semilla=semilla_val,
    )
    print(f"  train: {len(ds_train)} muestras | val: {len(ds_val)} muestras")

    # ─── BLOQUE 2: Modelo ────────────────────────────────────────────── #
    print("\n[BLOQUE 2] Construcción del modelo MultiTaskNet...")
    modelo = MultiTaskNet.desde_config(cfg)
    n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"  Parámetros entrenables: {n_params:,}")

    # ─── BLOQUE 3: Entrenamiento ─────────────────────────────────────── #
    print("\n[BLOQUE 3] Entrenamiento con early stopping...")
    trainer = Trainer(
        modelo=modelo,
        dataset_train=ds_train,
        dataset_val=ds_val,
        cfg=cfg,
        clientes_train_df=datos["clientes_train"],
        clientes_val_df=datos["clientes_val"],
    )
    resultado_entrenamiento = trainer.entrenar()
    print(f"  Mejor época: {resultado_entrenamiento['mejor_epoca']} "
          f"(val_loss={resultado_entrenamiento['mejor_val_loss']:.4f})")

    # ─── BLOQUE 4: Evaluación CLV ────────────────────────────────────── #
    print("\n[BLOQUE 4] Evaluación de CLV sobre test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.eval().to(device)

    col_c_cliente = cfg["columnas"]["cliente_idx"]
    col_clv_log = cfg["columnas"]["clv_target"]
    col_clv_orig = cfg["columnas"]["clv_futuro"]
    col_feats = cfg["columnas"]["features_cliente"]

    clv_log_test = datos["clientes_test"][col_clv_log].to_numpy(dtype=np.float64)
    c_test_idxs = datos["clientes_test"][col_c_cliente].to_numpy(dtype=np.int64)

    # Inferencia CLV
    features_test_map = _construir_features_map(datos["clientes_test"], cfg, device)
    clv_log_pred = _inferir_clv(modelo, c_test_idxs, features_test_map, cfg, device, n_productos)

    metricas_clv = evaluar_clv(
        clv_log_real=clv_log_test,
        clv_log_pred=clv_log_pred,
        clv_real_train=datos["clientes_train"][col_clv_log].to_numpy(dtype=np.float64),
        gasto_historico=datos["clientes_test"]["importe_total"].to_numpy(dtype=np.float64)
            if "importe_total" in datos["clientes_test"].columns else None,
        cfg=cfg,
        ruta_salida=get_artifact_path(cfg, "metrics_clv_json"),
    )

    # ─── BLOQUE 5: Evaluación Recomendación + Exportación ────────────── #
    print("\n[BLOQUE 5] Evaluación de recomendación...")
    features_train_map = _construir_features_map(datos["clientes_train"], cfg, device)

    metricas_rec = evaluar_recomendacion(
        modelo=modelo,
        interacciones_test=datos["interacciones_test"],
        positivos_train=pos_train,
        positivos_val=pos_val,
        clientes_train_vistos=stats_cs["clientes_train_vistos"],
        k_valores=cfg["evaluacion"]["k_valores"],
        cfg=cfg,
        interacciones_train=datos["interacciones_train"],
        features_map=features_train_map,
        ruta_salida=get_artifact_path(cfg, "metrics_rec_json"),
    )

    print("\n[BLOQUE 5] Exportando predicciones y embeddings...")
    artifacts_dir: Path = cfg["_artifacts_dir_abs"]

    exportar_embeddings(
        modelo=modelo,
        ruta_customer_emb=get_artifact_path(cfg, "customer_embeddings_npy"),
        ruta_product_emb=get_artifact_path(cfg, "product_embeddings_npy"),
    )

    exportar_predicciones_clv(
        clientes_test=datos["clientes_test"],
        clv_log_pred=clv_log_pred,
        ruta_salida=get_artifact_path(cfg, "predictions_clv_csv"),
        cfg=cfg,
    )

    pos_excluidos = {c: pos_train.get(c, set()) | pos_val.get(c, set())
                     for c in set(pos_train) | set(pos_val)}

    exportar_predicciones_rec(
        modelo=modelo,
        interacciones_test=datos["interacciones_test"],
        positivos_excluidos=pos_excluidos,
        cfg=cfg,
        ruta_salida=get_artifact_path(cfg, "predictions_rec_csv"),
        k_max=max(cfg["evaluacion"]["k_valores"]),
        features_map=features_train_map,
    )

    exportar_mapeos(
        interacciones_train=datos["interacciones_train"],
        productos=datos["productos"],
        cfg=cfg,
        artifacts_dir=artifacts_dir,
    )

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETADO")
    print(f"  Artifacts en: {artifacts_dir}")
    print("=" * 60)

    return {
        "entrenamiento": resultado_entrenamiento,
        "metricas_clv": metricas_clv,
        "metricas_rec": metricas_rec,
        "artifacts_dir": str(artifacts_dir),
    }


# ─── Funciones auxiliares privadas ───────────────────────────────────────── #

def _construir_positivos_desde_df(
    df: Any,
    col_c: str,
    col_p: str,
) -> dict[int, set[int]]:
    """Helper para construir positivos_por_cliente desde un DataFrame."""
    pos: dict[int, set[int]] = {}
    for c, p in zip(df[col_c].tolist(), df[col_p].tolist()):
        pos.setdefault(int(c), set()).add(int(p))
    return pos


def _construir_features_map(df: Any, cfg: dict, device: torch.device) -> dict[int, torch.Tensor] | None:
    """Helper que delega a la función del trainer."""
    from neural_network.training.trainer import _construir_features_map as _cm
    return _cm(df, cfg, device)


@torch.no_grad()
def _inferir_clv(
    modelo: "MultiTaskNet",
    c_idxs: np.ndarray,
    features_map: dict[int, torch.Tensor] | None,
    cfg: dict,
    device: torch.device,
    n_productos: int,
) -> np.ndarray:
    """Infiere clv_log para todos los clientes del array c_idxs."""
    modelo.eval()
    batch_size = 512
    preds = []
    n_feat = cfg["modelo"].get("n_features_cliente", 9) if cfg["modelo"]["usar_features"] else 0

    for i in range(0, len(c_idxs), batch_size):
        batch_c = c_idxs[i:i + batch_size]
        c_t = torch.tensor(batch_c, dtype=torch.long, device=device)
        # Producto ficticio (cualquiera, la cabeza CLV no lo usa)
        p_t = torch.zeros(len(batch_c), dtype=torch.long, device=device)

        feats = None
        if features_map is not None and n_feat > 0:
            rows = [features_map.get(int(c), torch.zeros(n_feat)) for c in batch_c]
            feats = torch.stack(rows).to(device)

        clv_pred, _ = modelo(c_t, p_t, feats)
        preds.append(clv_pred.cpu().numpy())

    return np.concatenate(preds)
