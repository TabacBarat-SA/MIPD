import pandas as pd
def cargar_cliente(ruta_datos="../datos/"):
    customers = pd.read_csv(f"{ruta_datos}olist_customers_dataset.csv")
    orders = pd.read_csv(f"{ruta_datos}datos/olist_orders_dataset.csv",
                     parse_dates=["order_purchase_timestamp"])