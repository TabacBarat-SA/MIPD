"""
conftest.py – Configuración común de pytest.

Inserta neural-network/src en sys.path para que los tests
puedan importar neural_network sin instalarlo.
"""
from __future__ import annotations

import sys
from pathlib import Path

# neural-network/tests/../src  →  neural-network/src
TESTS_DIR = Path(__file__).resolve().parent
MODULE_DIR = TESTS_DIR.parent
sys.path.insert(0, str(MODULE_DIR / "src"))
