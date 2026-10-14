"""
Early stopping y ModelCheckpoint para el entrenamiento de MultiTaskNet.

Diseñados exactamente para monitorear val_loss en modo min:
- ModelCheckpoint: guarda en disco el mejor checkpoint cada vez que val_loss alcanza un nuevo mínimo.
- EarlyStopping: monitorea val_loss en modo min, cuenta épocas de paciencia y detiene el entrenamiento.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class ModelCheckpoint:
    """Guarda automáticamente el checkpoint del modelo en disco cuando val_loss mejora.

    Parámetros
    ----------
    filepath : str | Path
        Ruta del archivo donde se guardará el checkpoint.
    monitor : str
        Métrica a monitorear (por defecto: 'val_loss').
    mode : str
        'min' (menor es mejor) o 'max' (mayor es mejor).
    save_best_only : bool
        Si es True, guarda únicamente cuando se supera el mejor valor histórico.
    verbose : bool
        Si es True, imprime aviso cada vez que se guarda un nuevo mejor modelo.
    """

    def __init__(
        self,
        filepath: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        verbose: bool = True,
    ) -> None:
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.mejor_score: float = float("inf") if mode == "min" else float("-inf")
        self.mejor_epoca: int = 0

    def _es_mejora(self, valor: float) -> bool:
        if self.mode == "min":
            return valor < self.mejor_score
        return valor > self.mejor_score

    def paso(
        self,
        valor: float,
        modelo: nn.Module,
        epoca: int,
        extra_datos: dict[str, Any] | None = None,
    ) -> bool:
        """Evalúa la métrica actual y guarda el checkpoint si es la mejor hasta ahora.

        Devuelve
        -------
        bool
            True si se guardó un nuevo checkpoint.
        """
        if not self.save_best_only or self._es_mejora(valor):
            self.mejor_score = valor
            self.mejor_epoca = epoca

            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "epoch": epoca,
                "model_state_dict": modelo.state_dict(),
                self.monitor: valor,
                "mejor_epoca": self.mejor_epoca,
                "mejor_val_loss": self.mejor_score,
            }
            if extra_datos:
                checkpoint.update(extra_datos)

            torch.save(checkpoint, self.filepath)
            if self.verbose:
                print(f"  [ModelCheckpoint] Epoca {epoca}: {self.monitor} mejoro a {valor:.4f} -> guardado en {self.filepath.name}")
            return True
        return False


class EarlyStopping:
    """Monitorea val_loss en modo min y detiene el entrenamiento si no mejora.

    Parámetros
    ----------
    monitor : str
        Métrica a monitorear (por defecto: 'val_loss').
    mode : str
        'min' para pérdidas o 'max' para métricas crecientes (por defecto: 'min').
    patience : int
        Número de épocas sin mejora antes de detener.
    min_delta : float
        Mejora mínima para considerar una época como mejor.
    restore_best_weights : bool
        Si es True, restaura en memoria los pesos del mejor modelo al detener.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 5,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self._mejor_loss: float = float("inf") if mode == "min" else float("-inf")
        self._epocas_sin_mejora: int = 0
        self._mejor_estado: dict | None = None
        self.mejor_epoca: int = 0
        self.detener: bool = False

    def paso(self, valor: float, modelo: nn.Module, epoca: int) -> bool:
        """Evalúa si hay mejora y actualiza el contador de paciencia."""
        es_mejora = (
            valor < self._mejor_loss - self.min_delta
            if self.mode == "min"
            else valor > self._mejor_loss + self.min_delta
        )

        if es_mejora:
            self._mejor_loss = valor
            self._epocas_sin_mejora = 0
            if self.restore_best_weights:
                self._mejor_estado = copy.deepcopy(modelo.state_dict())
            self.mejor_epoca = epoca
            self.detener = False
        else:
            self._epocas_sin_mejora += 1
            if self._epocas_sin_mejora >= self.patience:
                self.detener = True

        return self.detener

    def restaurar_mejor(self, modelo: nn.Module) -> None:
        """Restaura en el modelo los mejores pesos guardados en memoria."""
        if self._mejor_estado is not None:
            modelo.load_state_dict(self._mejor_estado)

    @property
    def mejor_loss(self) -> float:
        """Mejor pérdida de validación registrada."""
        return self._mejor_loss
