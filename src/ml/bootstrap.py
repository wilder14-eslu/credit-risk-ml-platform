"""Entrena el modelo automáticamente si no existe ningún artefacto todavía.

Pensado para un despliegue "clone-and-run" (Render, Streamlit Cloud) donde
el modelo entrenado nunca se commitea a git (ver `.gitignore`: `*.joblib`,
`data/processed/*`), pero el dataset crudo sí (`DATA/GiveMeSomeCredit/`).
`app.py` ya hacía este entrenamiento perezoso para el dashboard de
Streamlit; este módulo extrae esa lógica para que la API (`src/api/main.py`)
también pueda usarla al arrancar, sin duplicar el código.

No hace nada si ya existe un artefacto en `model_path` (o si `on_step` no
se usa, no imprime nada por sí solo salvo logging estándar).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.predict import DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)


def train_model_from_raw_dataset(
    model_path: Path,
    on_step: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Descarga (o reutiliza) el dataset crudo, lo valida y entrena el modelo.

    ``on_step`` es un callback opcional para reportar progreso (p. ej. una
    UI de Streamlit); si no se pasa, solo queda el logging estándar.
    """

    def _step(message: str) -> None:
        logger.info(message)
        if on_step is not None:
            on_step(message)

    from src.data_pipeline.ingest import download_give_me_some_credit
    from src.data_pipeline.validate import clean_out_of_range_rows, validate_input_data
    from src.ml.train import train_model

    _step("Descargando y preparando el dataset...")
    data_dir = download_give_me_some_credit()
    csv_files = sorted(Path(data_dir).glob("cs-training.csv")) or sorted(
        Path(data_dir).glob("*.csv")
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No se encontró ningún CSV en {data_dir} para entrenar el modelo."
        )

    _step("Limpiando y validando datos crudos...")
    raw_data = pd.read_csv(csv_files[0], index_col=0)
    raw_data = clean_out_of_range_rows(raw_data)
    validate_input_data(raw_data, require_target=True)

    _step("Entrenando el modelo de Machine Learning (puede tomar unos segundos)...")
    return train_model(raw_data, model_path=model_path, register=False)


def ensure_model_trained(
    model_path: Path | str | None = None,
    on_step: Callable[[str], None] | None = None,
) -> Path:
    """Devuelve la ruta de un modelo entrenado, entrenándolo si hace falta."""
    resolved = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    if resolved.is_file():
        return resolved
    logger.warning(
        "No hay modelo entrenado en %s; entrenando uno ahora (primer arranque).",
        resolved,
    )
    train_model_from_raw_dataset(resolved, on_step=on_step)
    return resolved
