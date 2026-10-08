import numpy as np
import pandas as pd

FEATURES = ["frecuencia", "importe_total", "ticket_medio", "cuotas_media",
            "recencia_dias", "antiguedad_dias", "dias_entre_pedidos", "n_categorias"]
TARGETS = ["clv_futuro", "clv_log", "recompra"]


def comprobar_cortes(cortes: dict, meses: int) -> None:
    assert list(cortes) == ["train", "val", "test"], "cortes: claves train, val y test, en ese orden"
    fechas = [pd.Timestamp(f) for f in cortes.values()]
    for a, b in zip(fechas, fechas[1:]):
        assert a + pd.DateOffset(months=meses) <= b, \
            f"La ventana de {a.date()} (+{meses} meses) se solapa con el corte {b.date()}"


def construir_tabla(pedidos, interacciones, corte, meses=3) -> pd.DataFrame:
    corte = pd.Timestamp(corte)
    fin = corte + pd.DateOffset(months=meses)

    pasado = pedidos[pedidos["fecha"] < corte]
    futuro = pedidos[(pedidos["fecha"] >= corte) & (pedidos["fecha"] < fin)]

    t = pasado.groupby("customer_idx").agg(
        frecuencia=("order_id", "size"),
        importe_total=("importe", "sum"),
        ticket_medio=("importe", "mean"),
        cuotas_media=("cuotas", "mean"),
        primera=("fecha", "min"),
        ultima=("fecha", "max"),
        estado_idx=("estado_idx", "last"),
    )
    t["recencia_dias"] = (corte - t["ultima"]).dt.days
    t["antiguedad_dias"] = (corte - t["primera"]).dt.days
    intervalo = (t["ultima"] - t["primera"]).dt.days
    t["dias_entre_pedidos"] = (intervalo / (t["frecuencia"] - 1).replace(0, np.nan)).fillna(0.0)

    cats = (interacciones[interacciones["fecha"] < corte]
            .groupby("customer_idx")["categoria_idx"].nunique().rename("n_categorias"))
    t = t.join(cats).fillna({"n_categorias": 0}).astype({"n_categorias": int})

    t["clv_futuro"] = futuro.groupby("customer_idx")["importe"].sum().reindex(t.index).fillna(0.0)
    t["clv_log"] = np.log1p(t["clv_futuro"])
    t["recompra"] = (t["clv_futuro"] > 0).astype(int)

    return t.drop(columns=["primera", "ultima"]).reset_index()


def construir_splits(pedidos, interacciones, cortes, meses) -> pd.DataFrame:
    comprobar_cortes(cortes, meses)
    partes = []
    for nombre, corte in cortes.items():
        t = construir_tabla(pedidos, interacciones, corte, meses)
        t["split"], t["corte"] = nombre, pd.Timestamp(corte)
        partes.append(t)
    return pd.concat(partes, ignore_index=True)


def etiquetar_interacciones(interacciones, cortes, meses) -> pd.DataFrame:

    comprobar_cortes(cortes, meses)
    c_val, c_test = pd.Timestamp(cortes["val"]), pd.Timestamp(cortes["test"])
    c_fin = c_test + pd.DateOffset(months=meses)

    f = interacciones["fecha"]
    split = np.select([f < c_val, f < c_test, f < c_fin], ["train", "val", "test"], default="")
    out = interacciones.assign(split=split)
    return out[out["split"] != ""].reset_index(drop=True)
