"""
PASO 0 – Inspección de los parquets y metadata.json.

Ejecuta con Run desde el IDE (sin argumentos).
Imprime columnas, dtypes, nº de filas, 5 filas de muestra y estadísticas
de cold-start y CLV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ─── Configuración de rutas independiente del directorio de trabajo ──────── #
MODULE_DIR = Path(__file__).resolve().parents[1]   # neural-network/
PROJECT_ROOT = MODULE_DIR.parent                   # raíz del proyecto
sys.path.insert(0, str(MODULE_DIR / "src"))

DEFAULT_CONFIG = MODULE_DIR / "configs" / "config.yaml"


def main(config_path: Path | None = None) -> None:
    """Punto de entrada principal del script de inspección."""
    import pandas as pd
    import numpy as np
    from neural_network.config import cargar_config
    from neural_network.data.preprocessing import (
        cargar_parquets,
        cargar_metadata,
        verificar_anti_fuga,
        calcular_estadisticas_coldstart,
        calcular_estadisticas_clv,
    )

    cfg = cargar_config(config_path)

    print("=" * 70)
    print("  PASO 0 – Inspección de datos")
    print(f"  processed_dir: {cfg['datos']['_processed_dir_abs']}")
    print("=" * 70)

    # ── metadata.json ────────────────────────────────────────────────── #
    meta = cargar_metadata(cfg)
    print("\n=== metadata.json ===")
    # Solo las claves de primer nivel (omite el mapa completo de IDs)
    for k, v in meta.items():
        if k != "mapas":
            print(f"  {k}: {v}")
        else:
            print(f"  mapas: {{clientes: {len(v.get('clientes', {}))} entradas, ...}}")

    # ── Parquets ─────────────────────────────────────────────────────── #
    datos = cargar_parquets(cfg)
    archivos_ordenados = [
        "clientes_train", "clientes_val", "clientes_test",
        "interacciones_train", "interacciones_val", "interacciones_test",
        "productos",
    ]

    for nombre in archivos_ordenados:
        df = datos[nombre]
        print(f"\n=== {nombre}.parquet ===")
        print(f"  Filas: {len(df):,}")
        print("  Columnas y dtypes:")
        for col, dtype in df.dtypes.items():
            print(f"    {col}: {dtype}")
        print("  Muestra (5 filas):")
        print(df.head(5).to_string(index=True))

    # ── Verificaciones ────────────────────────────────────────────────── #
    print("\n=== Verificación anti-fuga ===")
    verificar_anti_fuga(datos, cfg)

    print("\n=== Estadísticas cold-start ===")
    calcular_estadisticas_coldstart(datos, cfg)

    print("\n=== Estadísticas CLV ===")
    calcular_estadisticas_clv(datos, cfg)

    # ── Distribución de interacciones por cliente ─────────────────────── #
    print("\n=== Distribución de interacciones por cliente (train) ===")
    col_c = cfg["columnas"]["interaccion_cliente_idx"]
    ints_por_cliente = datos["interacciones_train"].groupby(col_c).size()
    print(ints_por_cliente.describe().to_string())
    print(f"  Clientes con 1 interacción: {(ints_por_cliente==1).sum():,}")
    print(f"  Clientes con >1 interacción: {(ints_por_cliente>1).sum():,}")

    print("\n  Inspección completada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PASO 0: Inspección de datos")
    parser.add_argument("--config", type=Path, default=None,
                        help="Ruta al config.yaml (por defecto: neural-network/configs/config.yaml)")
    args = parser.parse_args()
    main(config_path=args.config)
