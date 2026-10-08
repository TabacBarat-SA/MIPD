"""Carga de los CSV de Olist y construcción de pedidos, productos e interacciones."""
from pathlib import Path

import pandas as pd

ARCHIVOS = {
    "customers": ("olist_customers_dataset.csv",
                  ["customer_id", "customer_unique_id", "customer_state"]),
    "orders": ("olist_orders_dataset.csv",
               ["order_id", "customer_id", "order_status", "order_purchase_timestamp"]),
    "items": ("olist_order_items_dataset.csv",
              ["order_id", "order_item_id", "product_id", "price"]),
    "payments": ("olist_order_payments_dataset.csv",
                 ["order_id", "payment_installments", "payment_value"]),
    "products": ("olist_products_dataset.csv",
                 ["product_id", "product_category_name"]),
    "translation": ("product_category_name_translation.csv", None),
}


def cargar_tablas(raw_dir) -> dict:
    """Lee los 6 CSV y devuelve un diccionario nombre -> DataFrame."""
    raw_dir = Path(raw_dir)
    faltan = [f for f, _ in ARCHIVOS.values() if not (raw_dir / f).exists()]
    if faltan:
        raise FileNotFoundError(f"Faltan ficheros en {raw_dir}: {faltan}")

    tablas = {}
    for nombre, (fichero, columnas) in ARCHIVOS.items():
        fechas = ["order_purchase_timestamp"] if nombre == "orders" else []
        tablas[nombre] = pd.read_csv(raw_dir / fichero, usecols=columnas, parse_dates=fechas)
    return tablas


def construir_pedidos(orders, customers, payments) -> pd.DataFrame:
    entregados = orders.loc[orders["order_status"] == "delivered",
                            ["order_id", "customer_id", "order_purchase_timestamp"]]

    # un pedido puede tener varios pagos: se suman ANTES de unir para no duplicar filas
    pagos = payments.groupby("order_id", as_index=False).agg(
        importe=("payment_value", "sum"),
        cuotas=("payment_installments", "max"),
    )

    pedidos = (
        entregados
        .merge(customers, on="customer_id", how="left")
        .merge(pagos, on="order_id", how="left")
        .rename(columns={"order_purchase_timestamp": "fecha"})
        .dropna(subset=["importe", "customer_unique_id"])
        .query("importe > 0")
        .drop(columns="customer_id")
        .sort_values(["fecha", "order_id"])
        .reset_index(drop=True)
    )
    pedidos["cuotas"] = pedidos["cuotas"].fillna(1)
    assert pedidos["order_id"].is_unique, "order_id duplicado: algún merge duplicó filas"
    return pedidos


def construir_productos(products, translation) -> pd.DataFrame:
    p = products.merge(translation, on="product_category_name", how="left")
    p["categoria"] = (p["product_category_name_english"]
                      .fillna(p["product_category_name"])
                      .fillna("unknown"))
    return p[["product_id", "categoria"]].drop_duplicates("product_id")


def construir_interacciones(items, pedidos, productos) -> pd.DataFrame:
    por_producto = items.groupby(["order_id", "product_id"], as_index=False).agg(
        cantidad=("order_item_id", "size"),
        precio=("price", "sum"),
    )
    return (
        por_producto
        .merge(pedidos[["order_id", "customer_unique_id", "fecha"]], on="order_id", how="inner")
        .merge(productos, on="product_id", how="left")
        .fillna({"categoria": "unknown"})
        .sort_values(["fecha", "order_id"])
        .reset_index(drop=True)
    )


def _indices(valores) -> dict:
    return {v: i for i, v in enumerate(sorted(set(valores)))}


def anadir_indices(pedidos, interacciones, productos):
    mapas = {
        "clientes": _indices(pedidos["customer_unique_id"]),
        "productos": _indices(interacciones["product_id"]),
        "categorias": _indices(interacciones["categoria"]),
        "estados": _indices(pedidos["customer_state"].dropna()),
    }
    pedidos, interacciones = pedidos.copy(), interacciones.copy()
    productos = productos[productos["product_id"].isin(mapas["productos"])].copy()

    pedidos["customer_idx"] = pedidos["customer_unique_id"].map(mapas["clientes"])
    pedidos["estado_idx"] = pedidos["customer_state"].map(mapas["estados"]).fillna(-1).astype(int)
    interacciones["customer_idx"] = interacciones["customer_unique_id"].map(mapas["clientes"])
    interacciones["product_idx"] = interacciones["product_id"].map(mapas["productos"])
    interacciones["categoria_idx"] = interacciones["categoria"].map(mapas["categorias"])
    productos["product_idx"] = productos["product_id"].map(mapas["productos"])
    productos["categoria_idx"] = productos["categoria"].map(mapas["categorias"]).fillna(-1).astype(int)

    return pedidos, interacciones, productos.sort_values("product_idx").reset_index(drop=True), mapas
