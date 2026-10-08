"""
Carga y validación de config.yaml.

Todas las rutas definidas en config.yaml son relativas a PROJECT_ROOT y
se resuelven aquí en rutas absolutas para que el resto del código no
necesite saber nada de rutas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _resolver_rutas(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Convierte la sección ``datos`` y ``artifacts_dir`` a rutas absolutas."""
    processed_dir = project_root / cfg["datos"]["processed_dir"]
    cfg["datos"]["_processed_dir_abs"] = processed_dir

    # Archivos individuales dentro de processed/
    for clave in (
        "clientes_train",
        "clientes_val",
        "clientes_test",
        "interacciones_train",
        "interacciones_val",
        "interacciones_test",
        "productos",
        "metadata",
    ):
        nombre = cfg["datos"][clave]
        cfg["datos"][f"_{clave}_abs"] = processed_dir / nombre

    cfg["_artifacts_dir_abs"] = project_root / cfg["artifacts_dir"]
    return cfg


def cargar_config(config_path: Path | None = None) -> dict[str, Any]:
    """Carga ``config.yaml`` y resuelve rutas relativas a PROJECT_ROOT.

    Parámetros
    ----------
    config_path:
        Ruta al archivo YAML. Si es ``None`` se usa el valor por defecto
        ``neural-network/configs/config.yaml`` relativo a PROJECT_ROOT.

    Devuelve
    -------
    dict
        Configuración completa con rutas absolutas añadidas bajo claves
        con prefijo ``_``.
    """
    # PROJECT_ROOT = raíz del repositorio (dos niveles sobre este fichero)
    # Estructura: neural-network/src/neural_network/config.py
    module_dir: Path = Path(__file__).resolve().parents[2]  # neural-network/
    project_root: Path = module_dir.parent  # raíz del proyecto

    if config_path is None:
        config_path = module_dir / "configs" / "config.yaml"

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml no encontrado: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    cfg["_project_root"] = project_root
    cfg["_module_dir"] = module_dir
    cfg = _resolver_rutas(cfg, project_root)

    # Crea el directorio de artifacts si no existe
    artifacts_dir: Path = cfg["_artifacts_dir_abs"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return cfg


def get_artifact_path(cfg: dict[str, Any], nombre_clave: str) -> Path:
    """Devuelve la ruta absoluta de un artefacto definido en ``logging``."""
    return cfg["_artifacts_dir_abs"] / cfg["logging"][nombre_clave]
