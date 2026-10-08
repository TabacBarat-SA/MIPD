"""
Bloque 4 – Métricas de evaluación del CLV.

Métricas:
    - MAE (error absoluto medio)
    - RMSE (raíz del error cuadrático medio)
    - Correlación de Spearman (scipy.stats.spearmanr)

Reportadas en:
    1. Escala log (directamente sobre clv_log predicho/real).
    2. Escala original (expm1 aplicado a pred y real).
    3. Solo sobre clientes con CLV > 0 (subconjunto no cero).

Baselines:
    - Media del train (predicción constante = media(clv_log_train)).
    - Gasto histórico del cliente (importe_total_train → transformado a log1p).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr) if not np.isnan(corr) else float("nan")


def evaluar_clv(
    clv_log_real: np.ndarray,
    clv_log_pred: np.ndarray,
    clv_real_train: np.ndarray | None = None,
    gasto_historico: np.ndarray | None = None,
    cfg: dict[str, Any] | None = None,
    ruta_salida: Path | None = None,
) -> dict[str, Any]:
    """Calcula métricas de CLV en escala log y original.

    Parámetros
    ----------
    clv_log_real : np.ndarray
        Valores reales de clv_log (log1p(clv_futuro)) para el split test.
    clv_log_pred : np.ndarray
        Predicciones del modelo en escala log.
    clv_real_train : np.ndarray | None
        Valores de clv_log del split train, para la baseline de media.
    gasto_historico : np.ndarray | None
        importe_total de cada cliente test, para baseline de gasto histórico.
    cfg : dict | None
        Configuración (no usada actualmente, reservada para extensión).
    ruta_salida : Path | None
        Si se proporciona, guarda las métricas en JSON.

    Devuelve
    -------
    dict
        Métricas completas anidadas por escala y subconjunto.
    """
    clv_log_real = np.asarray(clv_log_real, dtype=np.float64)
    clv_log_pred = np.asarray(clv_log_pred, dtype=np.float64)

    # Escala original
    clv_orig_real = np.expm1(clv_log_real)
    clv_orig_pred = np.expm1(clv_log_pred)

    # Máscara de clientes con CLV > 0
    mask_pos = clv_orig_real > 0
    n_pos = int(mask_pos.sum())
    n_total = len(clv_log_real)
    pct_cero = 100.0 * (1 - n_pos / max(n_total, 1))

    metricas: dict[str, Any] = {
        "n_clientes_test": n_total,
        "n_clientes_clv_positivo": n_pos,
        "pct_clv_cero": round(pct_cero, 2),
    }

    # ── Métricas del modelo ──────────────────────────────────────────── #
    for escala, real, pred in [
        ("log", clv_log_real, clv_log_pred),
        ("original", clv_orig_real, clv_orig_pred),
    ]:
        metricas[f"modelo_{escala}_todos"] = {
            "mae": round(_mae(real, pred), 4),
            "rmse": round(_rmse(real, pred), 4),
            "spearman": round(_spearman(real, pred), 4),
        }
        if n_pos > 1:
            metricas[f"modelo_{escala}_clv_positivo"] = {
                "mae": round(_mae(real[mask_pos], pred[mask_pos]), 4),
                "rmse": round(_rmse(real[mask_pos], pred[mask_pos]), 4),
                "spearman": round(_spearman(real[mask_pos], pred[mask_pos]), 4),
            }

    # ── Baseline: media del train ────────────────────────────────────── #
    if clv_real_train is not None:
        media_log = float(np.mean(clv_real_train))
        pred_base_log = np.full_like(clv_log_real, media_log)
        pred_base_orig = np.expm1(pred_base_log)

        for escala, real, pred_b in [
            ("log", clv_log_real, pred_base_log),
            ("original", clv_orig_real, pred_base_orig),
        ]:
            metricas[f"baseline_media_{escala}"] = {
                "mae": round(_mae(real, pred_b), 4),
                "rmse": round(_rmse(real, pred_b), 4),
                "spearman": round(_spearman(real, pred_b), 4),
            }

    # ── Baseline: gasto histórico del cliente ────────────────────────── #
    if gasto_historico is not None:
        # gasto histórico → log1p para comparar con clv_log
        gasto_log = np.log1p(np.asarray(gasto_historico, dtype=np.float64))
        for escala, real, pred_g in [
            ("log", clv_log_real, gasto_log),
            ("original", clv_orig_real, np.asarray(gasto_historico, dtype=np.float64)),
        ]:
            metricas[f"baseline_gasto_historico_{escala}"] = {
                "mae": round(_mae(real, pred_g), 4),
                "rmse": round(_rmse(real, pred_g), 4),
                "spearman": round(_spearman(real, pred_g), 4),
            }

    # ── Imprimir resumen ─────────────────────────────────────────────── #
    print(f"[clv_metrics] n_test={n_total}, CLV=0: {pct_cero:.1f}%")
    m_log = metricas.get("modelo_log_todos", {})
    m_orig = metricas.get("modelo_original_todos", {})
    print(
        f"  [log]      MAE={m_log.get('mae'):.4f} RMSE={m_log.get('rmse'):.4f} "
        f"Spearman={m_log.get('spearman'):.4f}"
    )
    print(
        f"  [original] MAE={m_orig.get('mae'):.4f} RMSE={m_orig.get('rmse'):.4f} "
        f"Spearman={m_orig.get('spearman'):.4f}"
    )
    if n_pos > 1:
        m_pos = metricas.get("modelo_original_clv_positivo", {})
        print(
            f"  [original, CLV>0] MAE={m_pos.get('mae'):.4f} "
            f"RMSE={m_pos.get('rmse'):.4f} Spearman={m_pos.get('spearman'):.4f}"
        )

    if ruta_salida is not None:
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        with ruta_salida.open("w", encoding="utf-8") as f:
            json.dump(metricas, f, indent=2, ensure_ascii=False)
        print(f"[clv_metrics] Guardado: {ruta_salida}")

    return metricas
