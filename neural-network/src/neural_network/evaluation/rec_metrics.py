"""
Bloque 5 – Métricas de evaluación de recomendación.

Métricas:
    - Precision@K
    - NDCG@K
    para K configurable (por defecto: 5, 10, 20).

Protocolo:
    - Por cada cliente en interacciones_test con al menos una compra relevante,
      se rankean todos los productos candidatos (excluidos los vistos en train y val)
      usando el modelo.
    - Se reportan métricas separadas para clientes vistos vs. nuevos en train.
    - Clientes sin compras en test: excluidos del cálculo (se documenta cuántos).

Baseline:
    - Popularidad: recomienda los productos más populares en train
      (independiente del cliente).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_network.models.multitask_net import MultiTaskNet


# ─── Métricas elementales ────────────────────────────────────────────────── #

def _precision_at_k(relevantes: set[int], ranking: list[int], k: int) -> float:
    """Precision@K: fracción de los top-K que son relevantes."""
    top_k = ranking[:k]
    hits = sum(1 for p in top_k if p in relevantes)
    return hits / k


def _ndcg_at_k(relevantes: set[int], ranking: list[int], k: int) -> float:
    """NDCG@K: ganancia acumulada descontada normalizada."""
    top_k = ranking[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, p in enumerate(top_k)
        if p in relevantes
    )
    # IDCG: todos los relevantes en las primeras posiciones
    n_rel = min(len(relevantes), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


# ─── Evaluación del modelo ────────────────────────────────────────────────── #

def evaluar_recomendacion(
    modelo: MultiTaskNet,
    interacciones_test: Any,          # pd.DataFrame
    positivos_train: dict[int, set[int]],
    positivos_val: dict[int, set[int]],
    clientes_train_vistos: set[int],
    k_valores: list[int],
    cfg: dict[str, Any],
    interacciones_train: Any = None,   # para baseline popularidad
    features_map: dict[int, torch.Tensor] | None = None,
    ruta_salida: Path | None = None,
) -> dict[str, Any]:
    """Calcula Precision@K y NDCG@K sobre interacciones_test.

    Parámetros
    ----------
    modelo : MultiTaskNet
        Modelo entrenado.
    interacciones_test : pd.DataFrame
        Interacciones del split test.
    positivos_train : dict
        {customer_idx → set(product_idx)} en train.
    positivos_val : dict
        {customer_idx → set(product_idx)} en val.
    clientes_train_vistos : set[int]
        customer_idx que aparecen en interacciones_train.
    k_valores : list[int]
        Valores de K para las métricas.
    cfg : dict
        Configuración.
    interacciones_train : pd.DataFrame | None
        Para calcular la baseline de popularidad.
    features_map : dict | None
        {customer_idx → Tensor de features} para el modelo.
    ruta_salida : Path | None
        Si se proporciona, guarda las métricas en JSON.

    Devuelve
    -------
    dict
        Métricas por K, segmentadas por "vistos" / "nuevos" / "todos".
    """
    device = next(modelo.parameters()).device
    n_productos = modelo.n_productos

    col_c = cfg["columnas"]["interaccion_cliente_idx"]
    col_p = cfg["columnas"]["interaccion_producto_idx"]

    # Relevantes por cliente en test
    relevantes_test: dict[int, set[int]] = {}
    for c, p in zip(interacciones_test[col_c].tolist(), interacciones_test[col_p].tolist()):
        relevantes_test.setdefault(int(c), set()).add(int(p))

    # Popularidad baseline (calculada en train)
    ranking_popular: list[int] = []
    if interacciones_train is not None:
        from collections import Counter
        conteos = Counter(interacciones_train[col_p].tolist())
        ranking_popular = [p for p, _ in conteos.most_common()]
    else:
        ranking_popular = list(range(n_productos))

    metricas: dict[str, Any] = {
        "n_clientes_con_test": len(relevantes_test),
        "n_clientes_excluidos_sin_test": 0,
    }

    # Acumuladores por grupo y K
    grupos = ["todos", "vistos", "nuevos"]
    acum: dict[str, dict[int, dict[str, list[float]]]] = {
        g: {k: {"prec": [], "ndcg": []} for k in k_valores}
        for g in grupos
    }
    acum_base: dict[str, dict[int, dict[str, list[float]]]] = {
        g: {k: {"prec": [], "ndcg": []} for k in k_valores}
        for g in grupos
    }

    modelo.eval()
    with torch.no_grad():
        for c, rel in relevantes_test.items():
            # Excluidos: vistos en train + val
            excluidos = (positivos_train.get(c, set()) | positivos_val.get(c, set()))

            # Grupo del cliente
            grupo = "vistos" if c in clientes_train_vistos else "nuevos"

            # ── Ranking del modelo ────────────────────────────────────── #
            feat = None
            if features_map is not None:
                feat = features_map.get(c)
                if feat is not None:
                    feat = feat.unsqueeze(0)  # (1, n_feat)

            c_tensor = torch.tensor([c], dtype=torch.long, device=device)
            # Puntuar todos los productos
            all_products = torch.arange(n_productos, dtype=torch.long, device=device)

            # Batch para evitar OOM en GPU con muchos productos
            batch_size = 4096
            logits_list = []
            for i in range(0, n_productos, batch_size):
                p_batch = all_products[i:i + batch_size]
                c_batch = c_tensor.expand(len(p_batch))
                f_batch = None
                if feat is not None:
                    f_batch = feat.expand(len(p_batch), -1)
                _, logit = modelo(c_batch, p_batch, f_batch)
                logits_list.append(logit.cpu())

            logits = torch.cat(logits_list).numpy()  # (n_productos,)

            # Mask excluidos
            for ex in excluidos:
                if 0 <= ex < n_productos:
                    logits[ex] = -np.inf

            ranking_modelo = list(np.argsort(-logits))

            # Ranking baseline
            ranking_pop_filtrado = [p for p in ranking_popular if p not in excluidos]

            for k in k_valores:
                prec_m = _precision_at_k(rel, ranking_modelo, k)
                ndcg_m = _ndcg_at_k(rel, ranking_modelo, k)
                prec_b = _precision_at_k(rel, ranking_pop_filtrado, k)
                ndcg_b = _ndcg_at_k(rel, ranking_pop_filtrado, k)

                for g in ("todos", grupo):
                    acum[g][k]["prec"].append(prec_m)
                    acum[g][k]["ndcg"].append(ndcg_m)
                    acum_base[g][k]["prec"].append(prec_b)
                    acum_base[g][k]["ndcg"].append(ndcg_b)

    # Compilar resultados
    resultados: dict[str, Any] = {}
    for g in grupos:
        resultados[g] = {}
        for k in k_valores:
            pres = acum[g][k]["prec"]
            ndcgs = acum[g][k]["ndcg"]
            if pres:
                resultados[g][f"precision_at_{k}"] = round(float(np.mean(pres)), 4)
                resultados[g][f"ndcg_at_{k}"] = round(float(np.mean(ndcgs)), 4)
                resultados[g][f"n_clientes"] = len(pres)
            # baseline
            pb = acum_base[g][k]["prec"]
            nb = acum_base[g][k]["ndcg"]
            if pb:
                resultados[g][f"baseline_precision_at_{k}"] = round(float(np.mean(pb)), 4)
                resultados[g][f"baseline_ndcg_at_{k}"] = round(float(np.mean(nb)), 4)

    metricas["resultados"] = resultados

    # Imprimir resumen
    print("[rec_metrics] Resumen Precision@K / NDCG@K:")
    for g in grupos:
        for k in k_valores:
            p = resultados[g].get(f"precision_at_{k}", "N/A")
            n = resultados[g].get(f"ndcg_at_{k}", "N/A")
            nc = resultados[g].get("n_clientes", 0)
            print(f"  [{g:6s}] @{k:2d}: P={p}, NDCG={n} (n={nc})")

    if ruta_salida is not None:
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        with ruta_salida.open("w", encoding="utf-8") as f:
            json.dump(metricas, f, indent=2, ensure_ascii=False)
        print(f"[rec_metrics] Guardado: {ruta_salida}")

    return metricas
