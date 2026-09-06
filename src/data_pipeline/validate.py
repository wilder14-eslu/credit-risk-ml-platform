"""Data quality gates used before training and batch inference."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.feature_store.features import load_feature_schema

logger = logging.getLogger(__name__)


def clean_out_of_range_rows(
    data: pd.DataFrame,
    schema_path: Path = Path("config/data_schema.yaml"),
) -> pd.DataFrame:
    """Drop rows that violate a feature's min/max bounds.

    "Give Me Some Credit" ships with a handful of known bad rows (e.g. an
    applicant with age 0). Rather than let one bad row block training on
    150,000 good ones, this filters them out up front so
    `validate_input_data` (the strict, fail-fast gate used for
    single-record API requests) sees only well-formed bulk data.
    """
    schema = load_feature_schema(schema_path)
    definitions = schema["features"]

    keep_mask = pd.Series(True, index=data.index)
    for name, definition in definitions.items():
        source = definition["source"]
        column = source if source in data.columns else name
        if column not in data.columns:
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        if "min" in definition:
            keep_mask &= values.isna() | (values >= definition["min"])
        if "max" in definition:
            keep_mask &= values.isna() | (values <= definition["max"])

    cleaned = data.loc[keep_mask]
    dropped = len(data) - len(cleaned)
    if dropped:
        logger.warning("Se descartaron %d filas fuera de rango antes de entrenar.", dropped)
    return cleaned


def validate_input_data(
    data: pd.DataFrame,
    schema_path: Path = Path("config/data_schema.yaml"),
    require_target: bool = False,
) -> bool:
    schema = load_feature_schema(schema_path)
    definitions = schema["features"]

    missing = [
        definition["source"]
        for name, definition in definitions.items()
        if definition["source"] not in data.columns and name not in data.columns
    ]
    if missing:
        raise ValueError(f"Schema error: missing columns {missing}")

    for name, definition in definitions.items():
        source = definition["source"]
        column = source if source in data.columns else name
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().all():
            raise ValueError(f"Quality error: feature {column} has no numeric values")
        if "min" in definition and (values.dropna() < definition["min"]).any():
            raise ValueError(f"Quality error: {column} is below minimum")
        if "max" in definition and (values.dropna() > definition["max"]).any():
            raise ValueError(f"Quality error: {column} is above maximum")

    target = schema["target"]
    target_source = schema.get("target_source", target)
    if require_target and target not in data.columns and target_source not in data.columns:
        raise ValueError(f"Schema error: missing target column {target_source}")
    return True
