"""Feature preprocessing entry point.

Turns raw, validated Kaggle rows into the train/test split the model
training step expects, applying the same canonical feature transformation
(`src.feature_store.features.build_features`) used at inference time so
training and serving never drift apart.

Imputation is fitted **only on the training split** (median per feature)
and then applied to both splits. Computing the medians on the full dataset
before splitting would leak test-set information into training (a small
but real form of data leakage). The fitted values are persisted next to
the model (`imputation_values.json`, see `src.ml.train`) so inference
imputes a missing field with exactly the same value training used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_store.features import build_features, load_feature_schema

IMPUTATION_FILENAME = "imputation_values.json"


def build_features_and_target(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Canonical feature matrix (missing values kept as NaN) and binary target."""
    schema = load_feature_schema()
    target_source = schema.get("target_source", schema["target"])
    target_column = target_source if target_source in raw_data.columns else schema["target"]

    features = build_features(raw_data)
    target = pd.to_numeric(raw_data[target_column], errors="coerce")

    valid_target = target.notna()
    return features.loc[valid_target], target.loc[valid_target].astype(int)


def fit_imputation_values(features: pd.DataFrame) -> dict[str, float]:
    """Median per feature, computed on the data passed in (the training split)."""
    medians = features.median(numeric_only=True)
    return {column: float(medians.get(column, 0.0)) for column in features.columns}


def apply_imputation(features: pd.DataFrame, values: dict[str, float]) -> pd.DataFrame:
    """Fill NaNs with the given per-feature values (0.0 for unknown columns)."""
    return features.fillna(value=values).fillna(0.0)


def save_imputation_values(values: dict[str, float], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(values, indent=2), encoding="utf-8")
    return output_path


def load_imputation_values(path: Path | str) -> dict[str, float] | None:
    path = Path(path)
    if not path.is_file():
        return None
    return {key: float(value) for key, value in json.loads(path.read_text("utf-8")).items()}


def split_raw_features(
    raw_data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split **before** imputation (NaNs preserved).

    Used by `src.ml.evaluation` to impute inside each cross-validation fold.
    """
    features, target = build_features_and_target(raw_data)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def prepare_training_data(
    raw_data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into stratified train/test sets and impute with train medians.

    Median imputation keeps training robust to the missing income/dependents
    values that are common in "Give Me Some Credit" (~20% missing income).
    """
    x_train, x_test, y_train, y_test = split_raw_features(raw_data, test_size, random_state)
    values = fit_imputation_values(x_train)
    return apply_imputation(x_train, values), apply_imputation(x_test, values), y_train, y_test
