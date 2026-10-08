"""
Módulo neural_network – Red neuronal multi-tarea para CLV y recomendación de productos.

Exportaciones principales:
    - run_pipeline: orquesta los 5 bloques completos
    - MultiTaskNet: arquitectura de la red neuronal
"""
from neural_network.pipeline import run_pipeline
from neural_network.models.multitask_net import MultiTaskNet

__all__ = ["run_pipeline", "MultiTaskNet"]
