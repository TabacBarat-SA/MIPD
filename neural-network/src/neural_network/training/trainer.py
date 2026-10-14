"""
Bloque 3 – Entrenador multi-tarea.

Pérdida combinada:
    L = L_clv + λ · L_rec
    - L_clv: HuberLoss sobre clv_log (ya en escala log1p).
    - L_rec: BCEWithLogitsLoss sobre el logit de recomendación.
    - λ (lambda_rec) configurable en config.yaml.

Flujo por época:
    1. Llama a ``dataset_train.regenerar_negativos()`` para nuevos negativos.
    2. Itera batches de train, backward, gradient clipping, optimizer step.
    3. Valida sobre dataset_val con torch.no_grad().
    4. Llama a EarlyStopping; si dispara, restaura mejores pesos y guarda checkpoint.
    5. Registra curvas (CSV + PNG) en artifacts/.

Normalización de features:
    Las stats (media, std) se calculan SOLO sobre el split de train y se
    aplican a val/test para evitar data leakage. Se persisten en el checkpoint
    para que evaluate.py pueda reproducirlas sin acceso al train.

Soporte CPU/GPU automático.
"""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # sin GUI
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_network.data.dataset import InteractionDataset
from neural_network.models.multitask_net import MultiTaskNet
from neural_network.training.early_stopping import EarlyStopping, ModelCheckpoint


def _fijar_semillas(semilla: int) -> None:
    """Fija semillas de torch, numpy y random para reproducibilidad."""
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    np.random.seed(semilla)
    random.seed(semilla)
    # benchmark=True deja que cuDNN elija los kernels más rápidos para los
    # tamaños de entrada fijos de este modelo (gran ganancia en GPU).
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def calcular_stats_normalizacion(
    clientes_train_df: Any,  # pd.DataFrame
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Calcula media y std de las features SOLO sobre el split de train.

    Estas stats deben aplicarse a todos los splits (train, val, test) para
    evitar data leakage. Se devuelven como arrays numpy para persistirse
    en el checkpoint y usarse en inferencia posterior.

    Parámetros
    ----------
    clientes_train_df : pd.DataFrame
        DataFrame de clientes del split de entrenamiento.
    cfg : dict
        Configuración completa.

    Devuelve
    -------
    (mean, std) : arrays shape (n_features,), o (None, None) si usar_features=False.
    """
    if not cfg["modelo"]["usar_features"] or clientes_train_df is None:
        return None, None

    names = cfg["columnas"]["features_cliente"]
    arr = clientes_train_df[names].to_numpy(dtype=np.float32)
    mean = arr.mean(axis=0)        # (n_features,)
    std  = arr.std(axis=0) + 1e-8  # (n_features,)
    return mean, std


def construir_features_map(
    clientes_df: Any,          # pd.DataFrame
    cfg: dict[str, Any],
    mean: np.ndarray | None,
    std: np.ndarray | None,
) -> dict[int, torch.Tensor] | None:
    """Construye mapa {customer_idx → tensor de features} con normalización consistente.

    Parámetros
    ----------
    clientes_df : pd.DataFrame
        DataFrame de clientes del split a procesar.
    cfg : dict
        Configuración completa.
    mean : np.ndarray | None
        Media de las features calculada sobre TRAIN. Si None, se calcula localmente
        (solo válido para el split de entrenamiento en el primer ciclo).
    std : np.ndarray | None
        Desviación estándar de las features calculada sobre TRAIN.

    Devuelve
    -------
    dict[int, Tensor] | None
        Mapa {customer_idx → tensor normalizado}, o None si usar_features=False.
    """
    if not cfg["modelo"]["usar_features"] or clientes_df is None:
        return None

    col_c = cfg["columnas"]["cliente_idx"]
    names = cfg["columnas"]["features_cliente"]

    arr = clientes_df[names].to_numpy(dtype=np.float32)

    if mean is None or std is None:
        mean = arr.mean(axis=0, keepdims=True)
        std  = arr.std(axis=0, keepdims=True) + 1e-8

    arr = (arr - mean) / std

    # Devuelve un tensor denso shape (n_clientes_max+1, n_features) indexable
    # directamente con customer_idx en GPU, evitando el dict lookup por batch.
    idxs = clientes_df[col_c].to_numpy(dtype=np.int64)
    n_max = int(idxs.max()) + 1
    n_feat = arr.shape[1]
    tabla = np.zeros((n_max, n_feat), dtype=np.float32)
    tabla[idxs] = arr
    return torch.from_numpy(tabla)  # shape (n_max, n_feat)


# Alias privado para compatibilidad con pipeline.py
_construir_features_map = construir_features_map

# Tipo del mapa de features (ahora es un Tensor, antes era dict)
FeaturesMap = torch.Tensor | None


def _calcular_perdida_batch(
    modelo: MultiTaskNet,
    batch: tuple[torch.Tensor, ...],
    criterio_clv: nn.Module,
    criterio_rec: nn.Module,
    lambda_rec: float,
    device: torch.device,
    features_tabla: torch.Tensor | None,  # shape (n_clientes_max, n_features)
    n_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calcula la pérdida combinada para un batch.

    Devuelve (loss_total, loss_clv, loss_rec).
    """
    c_idx, p_idx, etiqueta, clv_target = [t.to(device) for t in batch]

    # Lookup vectorizado en GPU: features_tabla[c_idx] en una sola operación
    feats = None
    if features_tabla is not None and n_features > 0:
        feats = features_tabla.to(device)[c_idx]  # (batch, n_features)

    clv_pred, rec_logit = modelo(c_idx, p_idx, feats)

    loss_clv = criterio_clv(clv_pred, clv_target)
    loss_rec = criterio_rec(rec_logit, etiqueta)
    loss_total = loss_clv + lambda_rec * loss_rec

    return loss_total, loss_clv, loss_rec


class Trainer:
    """Entrena MultiTaskNet con early stopping y registro de curvas.

    Parámetros
    ----------
    modelo : MultiTaskNet
    dataset_train : InteractionDataset
    dataset_val : InteractionDataset
    cfg : dict
        Configuración completa.
    clientes_train_df : pd.DataFrame
        Usado para calcular las stats de normalización de features (SOLO train).
    clientes_val_df : pd.DataFrame
        Usado para construir el mapa de features de validación con stats de train.
    """

    def __init__(
        self,
        modelo: MultiTaskNet,
        dataset_train: InteractionDataset,
        dataset_val: InteractionDataset,
        cfg: dict[str, Any],
        clientes_train_df: Any = None,
        clientes_val_df: Any = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  device={self.device}")

        # Semilla
        _fijar_semillas(cfg["entrenamiento"]["semilla"])

        self.modelo = modelo.to(self.device)
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val

        # ── Normalización: stats calculadas SOLO en train ─────────────── #
        # Se calculan una vez y se aplican a train, val y test consistentemente.
        self.n_features = cfg["modelo"].get("n_features_cliente", 9) if cfg["modelo"]["usar_features"] else 0
        self.features_mean: np.ndarray | None
        self.features_std: np.ndarray | None
        self.features_mean, self.features_std = calcular_stats_normalizacion(clientes_train_df, cfg)

        self.features_train = construir_features_map(
            clientes_train_df, cfg, self.features_mean, self.features_std
        ) if clientes_train_df is not None else None
        self.features_val = construir_features_map(
            clientes_val_df, cfg, self.features_mean, self.features_std
        ) if clientes_val_df is not None else None

        # DataLoaders con workers para carga as\u00edncrona
        n_workers = 2
        e = cfg["entrenamiento"]
        self.loader_train = DataLoader(
            dataset_train,
            batch_size=e["batch_size"],
            shuffle=True,
            num_workers=n_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(n_workers > 0),
        )
        self.loader_val = DataLoader(
            dataset_val,
            batch_size=e["batch_size"] * 2,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(n_workers > 0),
        )

        # ── Optimizador ───────────────────────────────────────────────── #
        self.optimizer = torch.optim.AdamW(
            modelo.parameters(),
            lr=e["lr"],
            weight_decay=e["weight_decay"],
        )

        # ── Criterios de pérdida ──────────────────────────────────────── #
        self.criterio_clv = nn.HuberLoss(delta=e["huber_delta"])
        self.criterio_rec = nn.BCEWithLogitsLoss()
        self.lambda_rec: float = float(e["lambda_rec"])
        self.grad_clip: float = float(e["grad_clip"])

        # ── Rutas de artefactos ───────────────────────────────────────── #
        self.ruta_checkpoint: Path = cfg["_artifacts_dir_abs"] / cfg["logging"]["model_checkpoint"]
        self.ruta_curvas_csv: Path = cfg["_artifacts_dir_abs"] / cfg["logging"]["curvas_csv"]
        self.ruta_curvas_png: Path = cfg["_artifacts_dir_abs"] / cfg["logging"]["curvas_png"]

        # ── Callbacks: EarlyStopping y ModelCheckpoint (val_loss, mode='min') ── #
        es_cfg = cfg["early_stopping"]
        self.early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=es_cfg["patience"],
            min_delta=es_cfg["min_delta"],
            restore_best_weights=True,
        )
        self.checkpoint = ModelCheckpoint(
            filepath=self.ruta_checkpoint,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=True,
        )

        # Historial de curvas
        self.historial: list[dict[str, float]] = []

    # ------------------------------------------------------------------ #
    #  Bucle principal                                                     #
    # ------------------------------------------------------------------ #

    def entrenar(self) -> dict[str, Any]:
        """Ejecuta el bucle de entrenamiento completo.

        Devuelve
        -------
        dict
            Resumen con mejor época, mejor val_loss y ruta del checkpoint.
        """
        n_epocas = self.cfg["entrenamiento"]["n_epocas"]
        print(f"  {n_epocas} épocas máx.  patience={self.early_stopping.patience}")

        for epoca in range(1, n_epocas + 1):
            t0 = time.time()

            # ── Regenerar negativos de train ─────────────────────────────── #
            self.dataset_train.regenerar_negativos()

            # ── Fase train ───────────────────────────────────────────────── #
            train_loss, train_clv, train_rec = self._una_epoca_train()

            # ── Fase val ─────────────────────────────────────────────────── #
            val_loss, val_clv, val_rec = self._una_epoca_val()

            elapsed = time.time() - t0
            self.historial.append({
                "epoca": epoca,
                "train_loss": train_loss,
                "train_clv": train_clv,
                "train_rec": train_rec,
                "val_loss": val_loss,
                "val_clv": val_clv,
                "val_rec": val_rec,
            })

            print(
                f"  E{epoca:03d} {elapsed:.1f}s | "
                f"train {train_loss:.4f} (clv={train_clv:.4f} rec={train_rec:.4f}) | "
                f"val {val_loss:.4f} (clv={val_clv:.4f} rec={val_rec:.4f})"
            )

            # ── ModelCheckpoint (guarda si val_loss mejora en modo min) ── #
            self.checkpoint.paso(
                valor=val_loss,
                modelo=self.modelo,
                epoca=epoca,
                extra_datos={
                    "config": self.cfg,
                    "features_mean": self.features_mean,
                    "features_std": self.features_std,
                },
            )

            # ── EarlyStopping (detiene si val_loss no mejora en modo min) ─ #
            if self.early_stopping.paso(valor=val_loss, modelo=self.modelo, epoca=epoca):
                print(
                    f"  [early-stop] época {epoca}  "
                    f"mejor=época {self.early_stopping.mejor_epoca} "
                    f"val_loss={self.early_stopping.mejor_loss:.4f}"
                )
                break

        # Restaurar mejores pesos del modelo y guardar curvas
        self.early_stopping.restaurar_mejor(self.modelo)
        self._guardar_curvas()

        mejor_epoca = self.checkpoint.mejor_epoca or self.early_stopping.mejor_epoca
        mejor_val_loss = self.checkpoint.mejor_score if self.checkpoint.mejor_score != float("inf") else self.early_stopping.mejor_loss

        return {
            "mejor_epoca": mejor_epoca,
            "mejor_val_loss": mejor_val_loss,
            "checkpoint": str(self.ruta_checkpoint),
        }

    def _una_epoca_train(self) -> tuple[float, float, float]:
        """Itera todos los batches de train y devuelve (loss, clv, rec) medios."""
        self.modelo.train()
        total_loss = total_clv = total_rec = 0.0
        n = 0

        for batch in self.loader_train:
            self.optimizer.zero_grad()
            loss, lc, lr = _calcular_perdida_batch(
                self.modelo, batch,
                self.criterio_clv, self.criterio_rec,
                self.lambda_rec, self.device,
                self.features_train, self.n_features,
            )
            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.modelo.parameters(), self.grad_clip)

            self.optimizer.step()

            bs = batch[0].size(0)
            total_loss += loss.item() * bs
            total_clv += lc.item() * bs
            total_rec += lr.item() * bs
            n += bs

        return total_loss / n, total_clv / n, total_rec / n

    @torch.no_grad()
    def _una_epoca_val(self) -> tuple[float, float, float]:
        """Valida sobre el loader de validación."""
        self.modelo.eval()
        total_loss = total_clv = total_rec = 0.0
        n = 0

        for batch in self.loader_val:
            loss, lc, lr = _calcular_perdida_batch(
                self.modelo, batch,
                self.criterio_clv, self.criterio_rec,
                self.lambda_rec, self.device,
                self.features_val, self.n_features,
            )
            bs = batch[0].size(0)
            total_loss += loss.item() * bs
            total_clv += lc.item() * bs
            total_rec += lr.item() * bs
            n += bs

        return total_loss / n, total_clv / n, total_rec / n

    # ------------------------------------------------------------------ #
    #  Persistencia                                                        #
    # ------------------------------------------------------------------ #

    def _guardar_checkpoint(self) -> None:
        """Guarda el mejor modelo en disco con la config, el estado y las stats de normalización.

        Las stats de normalización (mean, std) se incluyen para que evaluate.py
        y cualquier script de inferencia puedan reproducir la misma normalización
        sin acceder al split de train.
        """
        torch.save({
            "model_state_dict": self.modelo.state_dict(),
            "config": self.cfg,
            "mejor_epoca": self.early_stopping.mejor_epoca,
            "mejor_val_loss": self.early_stopping.mejor_loss,
            # Stats de normalización calculadas sobre train (evita data leakage en inferencia)
            "features_mean": self.features_mean,
            "features_std": self.features_std,
        }, self.ruta_checkpoint)
        print(f"  checkpoint -> {self.ruta_checkpoint}")

    def _guardar_curvas(self) -> None:
        """Guarda historial de curvas en CSV y PNG."""
        if not self.historial:
            return

        keys = list(self.historial[0].keys())
        with self.ruta_curvas_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.historial)

        # ─── PNG con estilo moderno ──────────────────────────────────────────── #
        epocas = [h["epoca"] for h in self.historial]
        mejor_e = self.early_stopping.mejor_epoca
        mejor_v = self.early_stopping.mejor_loss

        metricas = [
            ("train_loss", "val_loss",  "Loss total",      "#4C9BE8", "#FF6B6B"),
            ("train_clv",  "val_clv",   "Loss CLV (Huber)", "#52D68A", "#FFB347"),
            ("train_rec",  "val_rec",   "Loss Rec (BCE)",   "#BB8FCE", "#F9CA24"),
        ]

        with plt.style.context("dark_background"):
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            fig.patch.set_facecolor("#1A1A2E")

            for ax, (lk_tr, lk_val, titulo, c_tr, c_val) in zip(axes, metricas):
                ax.set_facecolor("#16213E")
                tr_vals = [h[lk_tr] for h in self.historial]
                vl_vals = [h[lk_val] for h in self.historial]

                ax.plot(epocas, tr_vals, color=c_tr, linewidth=2,
                        label="train", alpha=0.9)
                ax.plot(epocas, vl_vals, color=c_val, linewidth=2,
                        label="val", alpha=0.9)

                # Marca la mejor época con línea vertical y anotación
                if mejor_e in epocas:
                    ax.axvline(x=mejor_e, color="white", linewidth=1,
                               linestyle="--", alpha=0.5)
                    best_val = vl_vals[epocas.index(mejor_e)]
                    ax.annotate(
                        f"  ép.{mejor_e}\n  {best_val:.4f}",
                        xy=(mejor_e, best_val),
                        color="white", fontsize=7, alpha=0.85,
                    )

                ax.set_title(titulo, fontsize=11, fontweight="bold",
                             color="white", pad=8)
                ax.set_xlabel("Época", fontsize=9, color="#AAAAAA")
                ax.tick_params(colors="#AAAAAA", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333355")
                ax.grid(True, alpha=0.15, linestyle=":")
                ax.legend(fontsize=8, framealpha=0.3)

            fig.suptitle(
                f"Curvas de entrenamiento – MultiTaskNet"
                f"   |   mejor época {mejor_e}  val_loss={mejor_v:.4f}",
                fontsize=12, color="white", fontweight="bold", y=1.02,
            )
            fig.tight_layout()
            fig.savefig(self.ruta_curvas_png, dpi=130,
                        bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)

        print(f"  curvas -> {self.ruta_curvas_png}")
