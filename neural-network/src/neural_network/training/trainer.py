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
from neural_network.training.early_stopping import EarlyStopping


def _fijar_semillas(semilla: int) -> None:
    """Fija semillas de torch, numpy y random para reproducibilidad."""
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    np.random.seed(semilla)
    random.seed(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _calcular_perdida_batch(
    modelo: MultiTaskNet,
    batch: tuple[torch.Tensor, ...],
    criterio_clv: nn.Module,
    criterio_rec: nn.Module,
    lambda_rec: float,
    device: torch.device,
    features_map: dict[int, torch.Tensor] | None,
    n_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calcula la pérdida combinada para un batch.

    Devuelve (loss_total, loss_clv, loss_rec).
    """
    c_idx, p_idx, etiqueta, clv_target = [t.to(device) for t in batch]

    # Construir features_cliente si procede
    feats = None
    if features_map is not None and n_features > 0:
        rows = []
        for ci in c_idx.cpu().tolist():
            rows.append(features_map.get(ci, torch.zeros(n_features)))
        feats = torch.stack(rows).to(device)

    clv_pred, rec_logit = modelo(c_idx, p_idx, feats)

    loss_clv = criterio_clv(clv_pred, clv_target)
    loss_rec = criterio_rec(rec_logit, etiqueta)
    loss_total = loss_clv + lambda_rec * loss_rec

    return loss_total, loss_clv, loss_rec


def _construir_features_map(
    clientes_df: Any,  # pd.DataFrame
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[int, torch.Tensor] | None:
    """Construye mapa {customer_idx → tensor de features} para lookup rápido."""
    if not cfg["modelo"]["usar_features"]:
        return None

    col_c = cfg["columnas"]["cliente_idx"]
    names = cfg["columnas"]["features_cliente"]
    feats_map: dict[int, torch.Tensor] = {}

    arr = clientes_df[names].to_numpy(dtype=np.float32)
    # Normalización simple: z-score por columna (solo stats del propio split)
    # Nota: en producción habría que usar stats del train también en val/test.
    # Aquí usamos la media/std del DataFrame recibido (aceptable para Semana 4).
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-8
    arr = (arr - mean) / std

    for i, c_idx in enumerate(clientes_df[col_c].tolist()):
        feats_map[int(c_idx)] = torch.tensor(arr[i], dtype=torch.float32)

    return feats_map


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
        Usado para construir el mapa de features de clientes.
    clientes_val_df : pd.DataFrame
        Usado para el mapa de features en validación.
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
        print(f"[trainer] Dispositivo: {self.device}")

        # Semilla
        _fijar_semillas(cfg["entrenamiento"]["semilla"])

        self.modelo = modelo.to(self.device)
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val

        # Features maps
        self.n_features = cfg["modelo"].get("n_features_cliente", 9) if cfg["modelo"]["usar_features"] else 0
        self.features_train = _construir_features_map(clientes_train_df, cfg, self.device) if clientes_train_df is not None else None
        self.features_val = _construir_features_map(clientes_val_df, cfg, self.device) if clientes_val_df is not None else None

        # DataLoaders
        e = cfg["entrenamiento"]
        self.loader_train = DataLoader(
            dataset_train,
            batch_size=e["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        self.loader_val = DataLoader(
            dataset_val,
            batch_size=e["batch_size"] * 2,
            shuffle=False,
            num_workers=0,
        )

        # Optimizador
        self.optimizer = torch.optim.AdamW(
            modelo.parameters(),
            lr=e["lr"],
            weight_decay=e["weight_decay"],
        )

        # Criterios de pérdida
        self.criterio_clv = nn.HuberLoss(delta=e["huber_delta"])
        self.criterio_rec = nn.BCEWithLogitsLoss()
        self.lambda_rec: float = float(e["lambda_rec"])
        self.grad_clip: float = float(e["grad_clip"])

        # Early stopping
        es_cfg = cfg["early_stopping"]
        self.early_stopping = EarlyStopping(
            patience=es_cfg["patience"],
            min_delta=es_cfg["min_delta"],
        )

        # Rutas de artefactos
        self.ruta_checkpoint: Path = cfg["_artifacts_dir_abs"] / cfg["logging"]["model_checkpoint"]
        self.ruta_curvas_csv: Path = cfg["_artifacts_dir_abs"] / cfg["logging"]["curvas_csv"]
        self.ruta_curvas_png: Path = cfg["_artifacts_dir_abs"] / cfg["logging"]["curvas_png"]

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
        print(f"[trainer] Entrenando {n_epocas} épocas máx. con patience={self.early_stopping.patience}")

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
                f"[E{epoca:03d}] {elapsed:.1f}s | "
                f"train: {train_loss:.4f} (clv={train_clv:.4f} rec={train_rec:.4f}) | "
                f"val: {val_loss:.4f} (clv={val_clv:.4f} rec={val_rec:.4f})"
            )

            # ── Early stopping ────────────────────────────────────────────── #
            if self.early_stopping.paso(val_loss, self.modelo, epoca):
                print(
                    f"[early stopping] Detenido en época {epoca}. "
                    f"Mejor: época {self.early_stopping.mejor_epoca} "
                    f"(val_loss={self.early_stopping.mejor_loss:.4f})"
                )
                break

        # Restaurar mejores pesos y guardar checkpoint
        self.early_stopping.restaurar_mejor(self.modelo)
        self._guardar_checkpoint()
        self._guardar_curvas()

        return {
            "mejor_epoca": self.early_stopping.mejor_epoca,
            "mejor_val_loss": self.early_stopping.mejor_loss,
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
        """Guarda el mejor modelo en disco con la config y el estado."""
        torch.save({
            "model_state_dict": self.modelo.state_dict(),
            "config": self.cfg,
            "mejor_epoca": self.early_stopping.mejor_epoca,
            "mejor_val_loss": self.early_stopping.mejor_loss,
        }, self.ruta_checkpoint)
        print(f"[trainer] Checkpoint guardado: {self.ruta_checkpoint}")

    def _guardar_curvas(self) -> None:
        """Guarda historial de curvas en CSV y PNG."""
        # CSV
        if self.historial:
            keys = list(self.historial[0].keys())
            with self.ruta_curvas_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.historial)

            # PNG
            epocas = [h["epoca"] for h in self.historial]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, (llave_train, llave_val, titulo) in zip(axes, [
                ("train_loss", "val_loss", "Loss total"),
                ("train_clv", "val_clv", "Loss CLV (Huber)"),
                ("train_rec", "val_rec", "Loss Rec (BCE)"),
            ]):
                ax.plot(epocas, [h[llave_train] for h in self.historial], label="train")
                ax.plot(epocas, [h[llave_val] for h in self.historial], label="val")
                ax.set_title(titulo)
                ax.set_xlabel("Época")
                ax.legend()
                ax.grid(True, alpha=0.3)
            fig.suptitle("Curvas de entrenamiento – MultiTaskNet")
            fig.tight_layout()
            fig.savefig(self.ruta_curvas_png, dpi=120)
            plt.close(fig)
            print(f"[trainer] Curvas guardadas: {self.ruta_curvas_csv}, {self.ruta_curvas_png}")
