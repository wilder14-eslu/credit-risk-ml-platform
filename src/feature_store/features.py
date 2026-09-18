"""Canonical feature transformations shared by training and inference."""

from pathlib import Path

import pandas as pd
import yaml

SCHEMA_PATH = Path("config/data_schema.yaml")


def load_feature_schema(schema_path: Path = SCHEMA_PATH) -> dict:
    with schema_path.open(encoding="utf-8") as schema_file:
        return yaml.safe_load(schema_file)


def _feature_definition(schema_path: Path = SCHEMA_PATH) -> dict:
    return load_feature_schema(schema_path)["features"]


def build_features(
    data: pd.DataFrame,
    schema_path: Path = SCHEMA_PATH,
) -> pd.DataFrame:
    """Return the canonical model matrix from raw or canonical column names."""
    definitions = _feature_definition(schema_path)
    result = {}
    for name, definition in definitions.items():
        source = definition["source"]
        column = source if source in data.columns else name
        if column not in data.columns:
            raise ValueError(f"Missing feature column: {source}")
        result[name] = pd.to_numeric(data[column], errors="coerce")
    df = pd.DataFrame(result, index=data.index)

    # --- FEATURE ENGINEERING AVANZADO ---
    # 1. Total de atrasos históricos
    df["total_past_due"] = (
        df.get("number_of_time_30_59_days_past_due", 0) +
        df.get("number_of_time_60_89_days_past_due", 0) +
        df.get("number_of_times_90_days_late", 0)
    )

    # 2. Ingreso disponible estimado (restringiendo el debt_ratio a un máximo de 1 para el cálculo)
    monthly_income = df.get("monthly_income", 0)
    debt_ratio = df.get("debt_ratio", 0)
    df["disposable_income"] = monthly_income * (1.0 - debt_ratio.clip(upper=1.0))

    # 3. Ingreso mensual por dependiente económico (asumiendo al menos 1: el propio solicitante)
    dependents = df.get("number_dependents", 0)
    df["income_per_dependent"] = monthly_income / (dependents + 1.0)

    return df


# FEATURE_NAMES incluye tanto las características base del esquema como las derivadas calculadas arriba
DERIVED_FEATURES = ("total_past_due", "disposable_income", "income_per_dependent")
FEATURE_NAMES = tuple(_feature_definition()) + DERIVED_FEATURES