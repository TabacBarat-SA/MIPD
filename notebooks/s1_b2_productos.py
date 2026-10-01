import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RUTA_DATOS = Path("../datos")
REPORTS = Path("../reports")


def cargar_productos(ruta_datos=RUTA_DATOS):
    items = pd.read_csv(ruta_datos / "olist_order_items_dataset.csv")
    products = pd.read_csv(ruta_datos / "olist_products_dataset.csv")
    translation = pd.read_csv(ruta_datos / "product_category_name_translation.csv")

    filas_antes = products.shape[0]
    sin_categoria = products["product_category_name"].isna().sum()

    products = products.merge(translation, on="product_category_name", how="left")

    sin_traduccion = (products["product_category_name"].notna()
                      & products["product_category_name_english"].isna()).sum()

    assert products.shape[0] == filas_antes

    products["categoria"] = products["product_category_name_english"].fillna("unknown")

    print(f"Productos: {filas_antes}")
    print(f"Sin categoría (vacía en products): {sin_categoria}")
    print(f"Con categoría pero sin traducción: {sin_traduccion}")

    return items, products[["product_id", "categoria"]]


def ranking_categorias(items, productos, n=15):
    ventas = items[["order_id", "product_id", "price"]].merge(
        productos, on="product_id", how="left")

    # Cada artículo debe conservar su fila
    assert ventas.shape[0] == items.shape[0]

    por_categoria = (ventas
                     .groupby("categoria")
                     .agg(articulos=("product_id", "size"),
                          ingresos=("price", "sum"))
                     .sort_values("articulos", ascending=False))

    print(f"\nTop {n} por nº de artículos vendidos:")
    print(por_categoria.head(n))
    print("\nTop 5 por ingresos (para comparar):")
    print(por_categoria.sort_values("ingresos", ascending=False).head(5))

    print("\nPrecio de los artículos:")
    print(ventas["price"].describe())

    return por_categoria.head(n)


def grafico_categorias(top):
    plt.figure(figsize=(9, 6))
    plt.barh(top.index, top["articulos"])
    plt.gca().invert_yaxis()  # la categoría con más ventas arriba
    plt.xlabel("Nº de artículos vendidos")
    plt.title("Las 15 categorías con más ventas")
    plt.tight_layout()
    plt.savefig(REPORTS / "s1_b2_categorias.png", dpi=150)
    plt.show()


def generar_analisis_productos():
    REPORTS.mkdir(parents=True, exist_ok=True)
    items, productos = cargar_productos()
    top = ranking_categorias(items, productos)
    grafico_categorias(top)


if __name__ == "__main__":
    generar_analisis_productos()