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
    return pd.DataFrame(result, index=data.index)


FEATURE_NAMES = tuple(_feature_definition())