"""
Bloque 5 – Exportación de predicciones y embeddings.

Genera en artifacts/:
    - predictions_clv.csv      (customer_idx, clv_log_real, clv_log_pred,
                                 clv_original_real, clv_original_pred)
    - predictions_rec.csv      (customer_idx, product_idx, score, rank)
    - customer_embeddings.npy  shape (n_clientes, dim_embedding)
    - product_embeddings.npy   shape (n_productos, dim_embedding)
    - customer_idx_to_id.json  {str(idx): customer_unique_id | str(idx)}
    - product_idx_to_id.json   {str(idx): product_id}
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_network.models.multitask_net import MultiTaskNet


def exportar_embeddings(
    modelo: MultiTaskNet,
    ruta_customer_emb: Path,
    ruta_product_emb: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Exporta embeddings de clientes y productos como .npy.

    Devuelve
    -------
    customer_emb : np.ndarray shape (n_clientes, dim)
    product_emb  : np.ndarray shape (n_productos, dim)
    """
    customer_emb = modelo.get_customer_embeddings().numpy()
    product_emb = modelo.get_product_embeddings().numpy()
    np.save(ruta_customer_emb, customer_emb)
    np.save(ruta_product_emb, product_emb)
    print(f"[export] customer_embeddings.npy: {customer_emb.shape} → {ruta_customer_emb}")
    print(f"[export] product_embeddings.npy:  {product_emb.shape} → {ruta_product_emb}")
    return customer_emb, product_emb


def exportar_predicciones_clv(
    clientes_test: Any,    # pd.DataFrame
    clv_log_pred: np.ndarray,
    ruta_salida: Path,
    cfg: dict[str, Any],
) -> None:
    """Guarda predictions_clv.csv con predicciones y valores reales."""
    import csv as csv_mod
    import numpy as np

    col_c = cfg["columnas"]["cliente_idx"]
    col_clv_log = cfg["columnas"]["clv_target"]
    col_clv_orig = cfg["columnas"]["clv_futuro"]

    c_idxs = clientes_test[col_c].tolist()
    clv_log_real = clientes_test[col_clv_log].to_numpy(dtype=np.float64)
    clv_orig_real = clientes_test[col_clv_orig].to_numpy(dtype=np.float64)
    clv_orig_pred = np.expm1(np.asarray(clv_log_pred, dtype=np.float64))

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with ruta_salida.open("w", newline="", encoding="utf-8") as f:
        w = csv_mod.writer(f)
        w.writerow(["customer_idx", "clv_log_real", "clv_log_pred",
                    "clv_original_real", "clv_original_pred"])
        for i, c in enumerate(c_idxs):
            w.writerow([
                c,
                round(float(clv_log_real[i]), 6),
                round(float(clv_log_pred[i]), 6),
                round(float(clv_orig_real[i]), 4),
                round(float(clv_orig_pred[i]), 4),
            ])
    print(f"[export] predictions_clv.csv ({len(c_idxs)} filas) → {ruta_salida}")


def exportar_predicciones_rec(
    modelo: MultiTaskNet,
    interacciones_test: Any,    # pd.DataFrame
    positivos_excluidos: dict[int, set[int]],
    cfg: dict[str, Any],
    ruta_salida: Path,
    k_max: int = 20,
    features_map: dict[int, torch.Tensor] | None = None,
) -> None:
    """Genera top-K recomendaciones por cliente y las guarda en CSV.

    Solo para los clientes con al menos una interacción en test.
    Excluye productos vistos en train o val.
    """
    device = next(modelo.parameters()).device
    n_productos = modelo.n_productos
    col_c = cfg["columnas"]["interaccion_cliente_idx"]
    col_p = cfg["columnas"]["interaccion_producto_idx"]
    batch_size_scoring: int = cfg["evaluacion"].get("batch_size_scoring", 4096)

    clientes_test_list = list(interacciones_test[col_c].unique())

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with ruta_salida.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["customer_idx", "product_idx", "score", "rank"])

        modelo.eval()
        with torch.no_grad():
            for c in clientes_test_list:
                excluidos = positivos_excluidos.get(int(c), set())

                feat = None
                if features_map is not None:
                    feat = features_map.get(int(c))  # shape (n_features,) o None

                # Usa el método centralizado del modelo
                logits = modelo.score_todos_los_productos(
                    customer_idx=int(c),
                    device=device,
                    features_cliente=feat,
                    batch_size=batch_size_scoring,
                ).numpy()

                for ex in excluidos:
                    if 0 <= ex < n_productos:
                        logits[ex] = -np.inf

                top_idx = np.argsort(-logits)[:k_max]
                for rank, p_idx in enumerate(top_idx, start=1):
                    score = float(logits[p_idx])
                    if not np.isfinite(score):
                        break
                    w.writerow([int(c), int(p_idx), round(score, 6), rank])

    print(f"[export] predictions_rec.csv ({len(clientes_test_list)} clientes, top-{k_max}) → {ruta_salida}")


def exportar_mapeos(
    interacciones_train: Any,  # pd.DataFrame
    productos: Any,            # pd.DataFrame
    cfg: dict[str, Any],
    artifacts_dir: Path,
) -> None:
    """Guarda customer_idx_to_id.json y product_idx_to_id.json."""
    import json

    col_c = cfg["columnas"]["interaccion_cliente_idx"]
    col_p = cfg["columnas"]["interaccion_producto_idx"]

    # customer_idx → customer_unique_id (vectorizado, sin iterrows)
    if "customer_unique_id" in interacciones_train.columns:
        df_c = interacciones_train[[col_c, "customer_unique_id"]].drop_duplicates()
        c_map = dict(zip(df_c[col_c].astype(int).astype(str), df_c["customer_unique_id"]))
    else:
        c_idxs = interacciones_train[col_c].unique()
        c_map = {str(int(ci)): str(int(ci)) for ci in c_idxs}

    # product_idx → product_id (vectorizado, sin iterrows)
    p_map = dict(zip(productos[col_p].astype(int).astype(str), productos["product_id"]))

    for nombre, mapa in [("customer_idx_to_id.json", c_map), ("product_idx_to_id.json", p_map)]:
        ruta = artifacts_dir / nombre
        with ruta.open("w", encoding="utf-8") as f:
            json.dump(mapa, f, ensure_ascii=False)
        print(f"[export] {nombre} ({len(mapa)} entradas) → {ruta}")
