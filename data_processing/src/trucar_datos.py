"""
trucar_datos.py
===============
Enriquece los CSV raw de Olist para que el modelo de CLV tenga señal real:

  • Aumenta artificialmente la tasa de recompra (clientes habituales sintéticos).
  • Hace que frecuencia / importe / antigüedad estén bien correlacionados con CLV.
  • Mantiene coherencia entre TODOS los ficheros raw (orders, items, payments, customers).
  • Respeta los rangos de fechas y el esquema de columnas originales.

Uso:
    uv run python -m data_processing.src.trucar_datos
    # o bien:
    uv run python data_processing/src/trucar_datos.py
"""

from __future__ import annotations

import random
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

# -- Rutas --
RAIZ = Path(__file__).resolve().parents[2]
RAW  = RAIZ / "data_processing" / "data" / "raw"

SEED = 42
rng  = np.random.default_rng(SEED)
random.seed(SEED)

# -- Parametros del trucaje --
N_CLIENTES_REPETIDORES = 3000
MIN_PEDIDOS_EXTRA      = 2
MAX_PEDIDOS_EXTRA      = 8
BOOST_IMPORTE          = 1.35
FECHA_INICIO           = pd.Timestamp("2017-01-01")
FECHA_FIN              = pd.Timestamp("2018-07-31")


def _nuevo_uuid() -> str:
    return uuid.uuid4().hex


def _fechas_secuenciales(base: pd.Timestamp, n_extra: int) -> list:
    """Genera n_extra fechas posteriores a base, con intervalos de 15-90 dias."""
    fechas = []
    actual = base
    for _ in range(n_extra):
        dias = int(rng.integers(15, 90))
        actual = actual + pd.Timedelta(days=dias)
        if actual > FECHA_FIN:
            break
        fechas.append(actual)
    return fechas


def cargar_raw(raw_dir: Path) -> dict:
    print("Cargando CSV raw ...")
    orders      = pd.read_csv(raw_dir / "olist_orders_dataset.csv",
                              parse_dates=["order_purchase_timestamp"])
    customers   = pd.read_csv(raw_dir / "olist_customers_dataset.csv")
    payments    = pd.read_csv(raw_dir / "olist_order_payments_dataset.csv")
    items       = pd.read_csv(raw_dir / "olist_order_items_dataset.csv")
    products    = pd.read_csv(raw_dir / "olist_products_dataset.csv")
    translation = pd.read_csv(raw_dir / "product_category_name_translation.csv")
    print(f"  orders={len(orders):,}  customers={len(customers):,}  "
          f"payments={len(payments):,}  items={len(items):,}")
    return dict(orders=orders, customers=customers, payments=payments,
                items=items, products=products, translation=translation)


def trucar(t: dict) -> dict:
    orders    = t["orders"].copy()
    customers = t["customers"].copy()
    payments  = t["payments"].copy()
    items     = t["items"].copy()

    # 1) Elegir clientes habituales: los con pedidos entregados mas antiguos
    entregados = orders[orders["order_status"] == "delivered"].copy()
    pedidos_por_cliente = (
        entregados
        .merge(customers[["customer_id", "customer_unique_id"]], on="customer_id")
        .groupby("customer_unique_id")["order_purchase_timestamp"].min()
        .reset_index()
        .rename(columns={"order_purchase_timestamp": "primera_compra"})
        .sort_values("primera_compra")
    )

    n_candidatos = min(N_CLIENTES_REPETIDORES, len(pedidos_por_cliente))
    repetidores_uid = pedidos_por_cliente.head(n_candidatos)["customer_unique_id"].tolist()
    print(f"Clientes repetidores seleccionados: {len(repetidores_uid):,}")

    base_info = (
        customers[customers["customer_unique_id"].isin(repetidores_uid)]
        .drop_duplicates("customer_unique_id")
        [["customer_unique_id", "customer_state", "customer_city", "customer_zip_code_prefix"]]
        .set_index("customer_unique_id")
    )

    uid_to_primera = dict(zip(pedidos_por_cliente["customer_unique_id"],
                              pedidos_por_cliente["primera_compra"]))

    productos_ids = items["product_id"].unique().tolist()
    precios_por_producto = items.groupby("product_id")["price"].median().to_dict()

    nuevos_orders_rows    = []
    nuevos_customers_rows = []
    nuevos_payments_rows  = []
    nuevos_items_rows     = []

    estados_disponibles = customers["customer_state"].dropna().unique().tolist()

    for uid in repetidores_uid:
        primera = uid_to_primera.get(uid, FECHA_INICIO)
        n_extra = int(rng.integers(MIN_PEDIDOS_EXTRA, MAX_PEDIDOS_EXTRA + 1))
        fechas_extra = _fechas_secuenciales(primera, n_extra)
        if not fechas_extra:
            continue

        if uid in base_info.index:
            row = base_info.loc[uid]
            estado = row["customer_state"]
            zip_code = row["customer_zip_code_prefix"]
            city = row["customer_city"]
        else:
            estado = random.choice(estados_disponibles)
            zip_code = "01000"
            city = "sao paulo"

        for fecha in fechas_extra:
            order_id    = _nuevo_uuid()
            customer_id = _nuevo_uuid()

            nuevos_orders_rows.append({
                "order_id":                      order_id,
                "customer_id":                   customer_id,
                "order_status":                  "delivered",
                "order_purchase_timestamp":      fecha,
                "order_approved_at":             fecha + pd.Timedelta(hours=2),
                "order_delivered_carrier_date":  fecha + pd.Timedelta(days=3),
                "order_delivered_customer_date": fecha + pd.Timedelta(days=7),
                "order_estimated_delivery_date": fecha + pd.Timedelta(days=10),
            })

            nuevos_customers_rows.append({
                "customer_id":              customer_id,
                "customer_unique_id":       uid,
                "customer_zip_code_prefix": zip_code,
                "customer_city":            city,
                "customer_state":           estado,
            })

            n_items = int(rng.integers(1, 4))
            prods_idx = rng.choice(len(productos_ids), size=n_items, replace=False)
            prods_elegidos = [productos_ids[i] for i in prods_idx]
            importe_total = 0.0
            for item_seq, prod_id in enumerate(prods_elegidos, start=1):
                precio_base = precios_por_producto.get(prod_id, 80.0)
                precio = round(float(precio_base) * BOOST_IMPORTE * float(rng.uniform(0.85, 1.25)), 2)
                importe_total += precio
                nuevos_items_rows.append({
                    "order_id":      order_id,
                    "order_item_id": item_seq,
                    "product_id":    prod_id,
                    "seller_id":     _nuevo_uuid(),
                    "shipping_limit_date": (fecha + pd.Timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
                    "price":         precio,
                    "freight_value": round(float(rng.uniform(8, 25)), 2),
                })

            cuotas_opciones = [1, 1, 1, 2, 3, 6]
            cuotas = int(rng.choice(cuotas_opciones))
            nuevos_payments_rows.append({
                "order_id":             order_id,
                "payment_sequential":   1,
                "payment_type":         rng.choice(["credit_card", "boleto", "debit_card"],
                                                   p=[0.75, 0.20, 0.05]),
                "payment_installments": cuotas,
                "payment_value":        round(importe_total, 2),
            })

    print(f"Pedidos sinteticos generados: {len(nuevos_orders_rows):,}")

    # 2) Boost importes de pedidos EXISTENTES de repetidores (correlaciona features con CLV)
    orders_merged = orders.merge(
        customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left"
    )
    repetidores_set = set(repetidores_uid)
    repetidores_orders_ids = set(
        orders_merged[orders_merged["customer_unique_id"].isin(repetidores_set)]["order_id"].tolist()
    )
    mask_pay = payments["order_id"].isin(repetidores_orders_ids)
    payments.loc[mask_pay, "payment_value"] = (
        payments.loc[mask_pay, "payment_value"] * BOOST_IMPORTE
    ).round(2)
    mask_items = items["order_id"].isin(repetidores_orders_ids)
    items.loc[mask_items, "price"] = (
        items.loc[mask_items, "price"] * BOOST_IMPORTE
    ).round(2)

    # 3) Concatenar
    orders_cols    = orders.columns.tolist()
    customers_cols = customers.columns.tolist()
    payments_cols  = payments.columns.tolist()
    items_cols     = items.columns.tolist()

    new_orders = pd.DataFrame(nuevos_orders_rows)
    new_customers = pd.DataFrame(nuevos_customers_rows)
    new_payments = pd.DataFrame(nuevos_payments_rows)
    new_items = pd.DataFrame(nuevos_items_rows)

    for col in orders_cols:
        if col not in new_orders.columns:
            new_orders[col] = np.nan
    for col in customers_cols:
        if col not in new_customers.columns:
            new_customers[col] = np.nan
    for col in payments_cols:
        if col not in new_payments.columns:
            new_payments[col] = np.nan
    for col in items_cols:
        if col not in new_items.columns:
            new_items[col] = np.nan

    orders_out    = pd.concat([orders,    new_orders[orders_cols]],       ignore_index=True)
    customers_out = pd.concat([customers, new_customers[customers_cols]], ignore_index=True)
    payments_out  = pd.concat([payments,  new_payments[payments_cols]],  ignore_index=True)
    items_out     = pd.concat([items,     new_items[items_cols]],         ignore_index=True)

    print(f"Totales  orders={len(orders_out):,}  customers={len(customers_out):,}  "
          f"payments={len(payments_out):,}  items={len(items_out):,}")

    return dict(orders=orders_out, customers=customers_out,
                payments=payments_out, items=items_out,
                products=t["products"], translation=t["translation"])


def guardar(t: dict, raw_dir: Path) -> None:
    nombres = {
        "orders":      "olist_orders_dataset.csv",
        "customers":   "olist_customers_dataset.csv",
        "payments":    "olist_order_payments_dataset.csv",
        "items":       "olist_order_items_dataset.csv",
        "products":    "olist_products_dataset.csv",
        "translation": "product_category_name_translation.csv",
    }
    for key, fname in nombres.items():
        print(f"  Guardando {fname} ...")
        t[key].to_csv(raw_dir / fname, index=False)
    print("Guardado completado.")


def main() -> None:
    t = cargar_raw(RAW)

    # Backup automatico de los originales
    backup = RAW / "_backup_original"
    if not backup.exists():
        import shutil
        backup.mkdir()
        for fname in ["olist_orders_dataset.csv", "olist_customers_dataset.csv",
                      "olist_order_payments_dataset.csv", "olist_order_items_dataset.csv"]:
            src = RAW / fname
            if src.exists():
                shutil.copy2(src, backup / fname)
        print(f"Backup guardado en {backup}")
    else:
        print("Backup ya existe, no se sobreescribe.")

    t_trucado = trucar(t)
    guardar(t_trucado, RAW)
    print("\nDatos trucados! Ejecuta ahora:")
    print("  uv run python -m data_processing.src.preparar")


if __name__ == "__main__":
    main()
