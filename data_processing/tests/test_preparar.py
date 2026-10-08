import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_olist import make_fake_olist  # noqa: E402
from src.features import comprobar_cortes  # noqa: E402
from src.preparar import preparar  # noqa: E402

CORTES = {"train": "2017-07-01", "val": "2017-10-01", "test": "2018-01-01"}


@pytest.fixture(scope="module")
def salida(tmp_path_factory):
    base = tmp_path_factory.mktemp("olist")
    make_fake_olist(base / "raw")
    preparar(base / "raw", base / "processed", CORTES)
    return base / "processed"


def test_se_crean_los_ficheros(salida):
    for s in CORTES:
        assert (salida / f"clientes_{s}.parquet").exists()
        assert (salida / f"interacciones_{s}.parquet").exists()
    assert (salida / "productos.parquet").exists()
    assert (salida / "metadata.json").exists()


def test_sin_fuga_en_variables(salida):
    for s, corte in CORTES.items():
        c = pd.read_parquet(salida / f"clientes_{s}.parquet")
        assert (c["recencia_dias"] >= 0).all()
        assert (c["corte"] == pd.Timestamp(corte)).all()


def test_interacciones_dentro_de_su_ventana(salida):
    tr = pd.read_parquet(salida / "interacciones_train.parquet")
    va = pd.read_parquet(salida / "interacciones_val.parquet")
    te = pd.read_parquet(salida / "interacciones_test.parquet")
    assert tr["fecha"].max() < pd.Timestamp(CORTES["val"])
    assert pd.Timestamp(CORTES["val"]) <= va["fecha"].min() and va["fecha"].max() < pd.Timestamp(CORTES["test"])
    assert te["fecha"].min() >= pd.Timestamp(CORTES["test"])


def test_cortes_solapados_dan_error():
    with pytest.raises(AssertionError):
        comprobar_cortes({"train": "2017-07-01", "val": "2017-08-01", "test": "2018-01-01"}, 3)
