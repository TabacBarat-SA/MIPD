"""Genera las tablas train/val/test a partir de los CSV de Olist.

Uso (desde la raíz del proyecto):
    python -m src.data.preparar
    python -m src.data.preparar --train 2017-10-01 --val 2018-01-01 --test 2018-04-01
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .features import FEATURES, TARGETS, construir_splits, etiquetar_interacciones
from .pedidos import (anadir_indices, cargar_tablas, construir_interacciones,
                      construir_pedidos, construir_productos)

RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ / "data" / "raw"
OUT = RAIZ / "data" / "processed"
CORTES = {"train": "2017-10-01", "val": "2018-01-01", "test": "2018-04-01"}  # ejemplo: ajustar
MESES = 3


def validar(pedidos, clientes, interacciones) -> dict:
    """Comprobaciones anti-fuga y de cordura. Devuelve estadísticas por split."""
    assert pedidos["order_id"].is_unique, "order_id duplicado"
    assert not clientes.duplicated(["customer_idx", "split"]).any(), "cliente repetido en un split"
    assert clientes[FEATURES + TARGETS].notna().all().all(), "hay NaN en variables o target"
    assert (clientes["recencia_dias"] >= 0).all(), "recencia negativa: fuga de información"
    assert (clientes["antiguedad_dias"] >= clientes["recencia_dias"]).all(), "antigüedad < recencia"

    stats = clientes.groupby("split").agg(
        clientes=("customer_idx", "size"),
        tasa_recompra=("recompra", "mean"),
        pct_clv_cero=("clv_futuro", lambda s: (s == 0).mean()),
        clv_medio=("clv_futuro", "mean"),
    ).round(4)
    assert (stats["tasa_recompra"] < 0.25).all(), "recompra > 25 %: ¿algún merge duplicó filas?"
    assert interacciones["split"].isin(["train", "val", "test"]).all()
    return stats.to_dict("index")


def guardar(df, nombre, out, csv=False):
    out.mkdir(parents=True, exist_ok=True)
    if csv:
        df.to_csv(out / f"{nombre}.csv", index=False)
    else:
        df.to_parquet(out / f"{nombre}.parquet", index=False)


def preparar(raw=RAW, out=OUT, cortes=CORTES, meses=MESES, csv=False) -> dict:
    t = cargar_tablas(raw)
    pedidos = construir_pedidos(t["orders"], t["customers"], t["payments"])
    productos = construir_productos(t["products"], t["translation"])
    interacciones = construir_interacciones(t["items"], pedidos, productos)
    pedidos, interacciones, productos, mapas = anadir_indices(pedidos, interacciones, productos)
    print(f"Pedidos entregados: {len(pedidos):,} | clientes únicos: {len(mapas['clientes']):,}")

    clientes = construir_splits(pedidos, interacciones, cortes, meses)
    interacciones = etiquetar_interacciones(interacciones, cortes, meses)
    stats = validar(pedidos, clientes, interacciones)

    for s in ("train", "val", "test"):
        guardar(clientes[clientes["split"] == s].reset_index(drop=True), f"clientes_{s}", out, csv)
        guardar(interacciones[interacciones["split"] == s].reset_index(drop=True),
                f"interacciones_{s}", out, csv)
    guardar(productos, "productos", out, csv)

    meta = {"cortes": cortes, "meses": meses, "features": FEATURES, "targets": TARGETS,
            "mapas": mapas, "stats": stats}
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str),
                                       encoding="utf-8")

    print("\nResumen por split:")
    print(pd.DataFrame(stats).T.to_string())
    print("\nInteracciones por split:", interacciones["split"].value_counts().to_dict())
    print("Listo. Ficheros en", out)
    return meta


def main():
    p = argparse.ArgumentParser(description="Prepara los datos train/val/test de Olist")
    p.add_argument("--raw", type=Path, default=RAW)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--meses", type=int, default=MESES, help="horizonte del CLV en meses")
    p.add_argument("--csv", action="store_true", help="guardar CSV en vez de parquet")
    for nombre, fecha in CORTES.items():
        p.add_argument(f"--{nombre}", default=fecha, help=f"fecha de corte de {nombre}")
    a = p.parse_args()
    preparar(a.raw, a.out, {"train": a.train, "val": a.val, "test": a.test}, a.meses, a.csv)


if __name__ == "__main__":
    main()
