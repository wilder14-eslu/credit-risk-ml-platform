"""Feature preprocessing entry point.

Turns raw, validated Kaggle rows into the train/test split the model
training step expects, applying the same canonical feature transformation
(`src.feature_store.features.build_features`) used at inference time so
training and serving never drift apart.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_store.features import build_features, load_feature_schema


def prepare_training_data(
    raw_data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Clean, impute and split raw data into stratified train/test sets."""
    schema = load_feature_schema()
    target_source = schema.get("target_source", schema["target"])
    target_column = target_source if target_source in raw_data.columns else schema["target"]

    features = build_features(raw_data)
    target = pd.to_numeric(raw_data[target_column], errors="coerce")

    valid_target = target.notna()
    features = features.loc[valid_target]
    target = target.loc[valid_target].astype(int)

    # Median imputation keeps training robust to the missing income/dependents
    # values that are common in "Give Me Some Credit" (~20% missing income).
    features = features.fillna(features.median(numeric_only=True))

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
