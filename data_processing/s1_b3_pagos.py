import pandas as pd

def cargar_pagos(ruta_datos="../datos/"):
    payments = pd.read_csv(f"{ruta_datos}olist_order_payments_dataset.csv")

