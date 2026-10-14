"""
run_pipeline.py – Ejecuta el pipeline completo (Bloques 0-5).

Ejecuta con Run desde el IDE (sin argumentos).
Genera todos los artefactos en neural-network/artifacts/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]   # neural-network/
PROJECT_ROOT = MODULE_DIR.parent                   # raíz del proyecto
sys.path.insert(0, str(MODULE_DIR / "src"))

DEFAULT_CONFIG = MODULE_DIR / "configs" / "config.yaml"

# ==============================================================================
# CONFIGURACIÓN RÁPIDA (modifica estas variables para experimentar)
# ==============================================================================
MAX_EPOCHS: int = 30      # Cuántas épocas hacer como máximo (ej. 30, 50, 100)
PATIENCE: int = 5          # En cuántas épocas parar si val_loss no mejora (early stopping)
MIN_DELTA: float = 1e-4    # Mejora mínima en val_loss para considerar mejora
BATCH_SIZE: int = 512      # Tamaño de batch
LR: float = 1e-3           # Learning rate (tasa de aprendizaje del optimizador)
# ==============================================================================


def main(
    config_path: Path | None = None,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    min_delta: float = MIN_DELTA,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> None:
    """Orquesta el pipeline completo de 5 bloques con hiperparámetros configurables."""
    from neural_network.pipeline import run_pipeline

    resultado = run_pipeline(
        config_path=config_path,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        batch_size=batch_size,
        lr=lr,
    )
    print("\nResumen final:")
    print(f"  Mejor época: {resultado['entrenamiento']['mejor_epoca']}")
    print(f"  Mejor val_loss: {resultado['entrenamiento']['mejor_val_loss']:.4f}")
    print(f"  Artifacts: {resultado['artifacts_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline completo CLV + Recomendación")
    parser.add_argument("--config", type=Path, default=None,
                        help="Ruta al config.yaml (por defecto: neural-network/configs/config.yaml)")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS,
                        help=f"Número máx de épocas (por defecto: {MAX_EPOCHS})")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help=f"Paciencia early stopping (por defecto: {PATIENCE})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Tamaño de batch (por defecto: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"Learning rate (por defecto: {LR})")
    args = parser.parse_args()

    main(
        config_path=args.config,
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
    )
