"""Genera CSV sintéticos con las mismas columnas que Olist (solo para tests)."""
from pathlib import Path

import numpy as np
import pandas as pd


def make_fake_olist(raw_dir: Path, n_people: int = 4000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    n_orders = int(n_people * 1.04)
    person = np.concatenate([np.arange(n_people), rng.choice(n_people, n_orders - n_people)])
    rng.shuffle(person)
    states = np.array(["SP", "RJ", "MG", "RS", "BA"])
    person_state = rng.choice(states, n_people)

    order_ids = [f"o{i:06d}" for i in range(n_orders)]
    customer_ids = [f"c{i:06d}" for i in range(n_orders)]   # uno nuevo por pedido
    days = pd.to_datetime("2017-01-01") + pd.to_timedelta(rng.integers(0, 600, n_orders), unit="D")
    days = days + pd.to_timedelta(rng.integers(0, 86400, n_orders), unit="s")
    status = rng.choice(["delivered", "canceled", "shipped"], n_orders, p=[0.96, 0.03, 0.01])

    pd.DataFrame({
        "customer_id": customer_ids,
        "customer_unique_id": [f"u{p:06d}" for p in person],
        "customer_zip_code_prefix": 1000,
        "customer_city": "x",
        "customer_state": person_state[person],
    }).to_csv(raw_dir / "olist_customers_dataset.csv", index=False)

    pd.DataFrame({
        "order_id": order_ids, "customer_id": customer_ids, "order_status": status,
        "order_purchase_timestamp": days.strftime("%Y-%m-%d %H:%M:%S"),
    }).to_csv(raw_dir / "olist_orders_dataset.csv", index=False)

    n_prod = 300
    cats = ["beleza_saude", "cama_mesa_banho", "esporte_lazer", "informatica", "pc_gamer", None]
    pd.DataFrame({
        "product_id": [f"p{i:04d}" for i in range(n_prod)],
        "product_category_name": rng.choice(cats, n_prod),
        "product_weight_g": rng.integers(100, 5000, n_prod).astype(float),
        "product_length_cm": rng.integers(10, 50, n_prod).astype(float),
        "product_height_cm": rng.integers(5, 40, n_prod).astype(float),
        "product_width_cm": rng.integers(5, 40, n_prod).astype(float),
        "product_photos_qty": rng.integers(1, 6, n_prod).astype(float),
    }).to_csv(raw_dir / "olist_products_dataset.csv", index=False)

    pd.DataFrame({
        "product_category_name": cats[:4],
        "product_category_name_english": ["health_beauty", "bed_bath_table", "sports_leisure", "computers_accessories"],
    }).to_csv(raw_dir / "product_category_name_translation.csv", index=False)

    rows = []
    for oid in order_ids:
        for k in range(1, rng.integers(1, 4) + 1):
            rows.append((oid, k, f"p{rng.integers(0, n_prod):04d}", round(float(rng.gamma(2, 50)) + 5, 2)))
    pd.DataFrame(rows, columns=["order_id", "order_item_id", "product_id", "price"]).to_csv(
        raw_dir / "olist_order_items_dataset.csv", index=False)

    prows = []
    for oid in order_ids:
        for k in range(1, rng.integers(1, 3) + 1):     # a veces 2 pagos
            prows.append((oid, k, "credit_card", int(rng.integers(1, 6)), round(float(rng.gamma(2, 40)) + 5, 2)))
    pd.DataFrame(prows, columns=["order_id", "payment_sequential", "payment_type",
                                 "payment_installments", "payment_value"]).to_csv(
        raw_dir / "olist_order_payments_dataset.csv", index=False)
