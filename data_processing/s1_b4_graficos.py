import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RUTA_DATOS = Path("../datos")
REPORTS = Path("../reports")


def cargar_pedidos(ruta_datos=RUTA_DATOS):
    customers = pd.read_csv(ruta_datos / "olist_customers_dataset.csv")
    orders = pd.read_csv(ruta_datos / "olist_orders_dataset.csv",
                         parse_dates=["order_purchase_timestamp"])
    payments = pd.read_csv(ruta_datos / "olist_order_payments_dataset.csv")

    orders = orders[orders["order_status"] == "delivered"]

    pago_por_pedido = (payments
                       .groupby("order_id", as_index=False)["payment_value"]
                       .sum())

    pedidos = (
        orders[["order_id", "customer_id", "order_purchase_timestamp"]]
        .merge(customers[["customer_id", "customer_unique_id", "customer_state"]],
               on="customer_id", how="left")
        .merge(pago_por_pedido, on="order_id", how="left")
        .rename(columns={"order_purchase_timestamp": "fecha",
                         "payment_value": "importe"})
        .dropna(subset=["importe"])
    )
    print(pedidos.shape)
    return pedidos