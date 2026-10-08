"""
Early stopping con restauración del mejor checkpoint en memoria.

Criterio: pérdida de validación total (L_clv + λ·L_rec).
Se detiene si la mejora es menor que ``min_delta`` durante ``patience`` épocas.
Los mejores pesos se guardan en memoria (no en disco) para restauración rápida;
el checkpoint en disco se escribe por el Trainer.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EarlyStopping:
    """Monitorea la pérdida de validación y detiene el entrenamiento si no mejora.

    Parámetros
    ----------
    patience : int
        Número de épocas sin mejora antes de detener.
    min_delta : float
        Mejora mínima para considerar una época como mejor.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._mejor_loss: float = float("inf")
        self._epocas_sin_mejora: int = 0
        self._mejor_estado: dict | None = None
        self.mejor_epoca: int = 0
        self.detener: bool = False

    def paso(self, val_loss: float, modelo: nn.Module, epoca: int) -> bool:
        """Evalúa si hay mejora y actualiza el estado interno.

        Parámetros
        ----------
        val_loss : float
            Pérdida de validación de la época actual.
        modelo : nn.Module
            Modelo actual. Se guarda una copia en memoria si mejora.
        epoca : int
            Índice de la época actual (para reportar).

        Devuelve
        -------
        bool
            ``True`` si el entrenamiento debe detenerse.
        """
        if val_loss < self._mejor_loss - self.min_delta:
            self._mejor_loss = val_loss
            self._epocas_sin_mejora = 0
            self._mejor_estado = copy.deepcopy(modelo.state_dict())
            self.mejor_epoca = epoca
            self.detener = False
        else:
            self._epocas_sin_mejora += 1
            if self._epocas_sin_mejora >= self.patience:
                self.detener = True

        return self.detener

    def restaurar_mejor(self, modelo: nn.Module) -> None:
        """Carga los mejores pesos guardados en el modelo.

        Parámetros
        ----------
        modelo : nn.Module
            Modelo al que se restauran los pesos.

        Lanza
        -----
        RuntimeError
            Si nunca se guardó ningún estado (no se llamó a ``paso``).
        """
        if self._mejor_estado is None:
            raise RuntimeError("No hay estado guardado. Llama a paso() primero.")
        modelo.load_state_dict(self._mejor_estado)

    @property
    def mejor_loss(self) -> float:
        """Mejor pérdida de validación registrada."""
        return self._mejor_loss
