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


def main(config_path: Path | None = None) -> None:
    """Orquesta el pipeline completo de 5 bloques."""
    from neural_network.pipeline import run_pipeline
    resultado = run_pipeline(config_path=config_path)
    print("\nResumen final:")
    print(f"  Mejor época: {resultado['entrenamiento']['mejor_epoca']}")
    print(f"  Mejor val_loss: {resultado['entrenamiento']['mejor_val_loss']:.4f}")
    print(f"  Artifacts: {resultado['artifacts_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline completo CLV + Recomendación")
    parser.add_argument("--config", type=Path, default=None,
                        help="Ruta al config.yaml (por defecto: neural-network/configs/config.yaml)")
    args = parser.parse_args()
    main(config_path=args.config)
