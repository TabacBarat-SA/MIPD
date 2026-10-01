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


def grafico_pedidos_por_cliente(pedidos):
    pedidos_por_cliente = pedidos.groupby("customer_unique_id")["order_id"].nunique()

    distribucion = pedidos_por_cliente.value_counts().sort_index()
    pct_una_vez = (pedidos_por_cliente == 1).mean() * 100
    print(distribucion)
    print(f"Clientes que compran una sola vez: {pct_una_vez:.1f} %")

    plt.figure(figsize=(8, 5))
    plt.bar(distribucion.index, distribucion.values)
    plt.yscale("log")
    plt.xticks(distribucion.index)
    plt.xlabel("Nº de pedidos del cliente")
    plt.ylabel("Nº de clientes (escala log)")
    plt.title("Distribución de pedidos por cliente")
    plt.tight_layout()
    plt.savefig(REPORTS / "s1_b4_pedidos_por_cliente.png", dpi=150)
    plt.show()


def grafico_importes(pedidos):
    print(pedidos["importe"].describe())

    limite = pedidos["importe"].quantile(0.99)
    importes = pedidos.loc[pedidos["importe"] <= limite, "importe"]
    print(f"Percentil 99: {limite:.2f} | pedidos fuera del gráfico: "
          f"{(pedidos['importe'] > limite).sum()}")

    plt.figure(figsize=(8, 5))
    plt.hist(importes, bins=50)
    plt.xlabel("Importe del pedido")
    plt.ylabel("Nº de pedidos")
    plt.title("Distribución de importes (hasta el percentil 99)")
    plt.tight_layout()
    plt.savefig(REPORTS / "s1_b4_importes.png", dpi=150)
    plt.show()


def grafico_pedidos_por_mes(pedidos):
    por_mes = pedidos.groupby(pedidos["fecha"].dt.to_period("M")).size()
    print(por_mes)

    plt.figure(figsize=(10, 5))
    plt.bar(por_mes.index.astype(str), por_mes.values)
    plt.xticks(rotation=90)
    plt.xlabel("Mes")
    plt.ylabel("Nº de pedidos")
    plt.title("Pedidos entregados por mes")
    plt.tight_layout()
    plt.savefig(REPORTS / "s1_b4_pedidos_por_mes.png", dpi=150)
    plt.show()


def generar_graficos():
    REPORTS.mkdir(parents=True, exist_ok=True)
    pedidos = cargar_pedidos()
    grafico_pedidos_por_cliente(pedidos)
    grafico_importes(pedidos)
    grafico_pedidos_por_mes(pedidos)


if __name__ == "__main__":
    generar_graficos()