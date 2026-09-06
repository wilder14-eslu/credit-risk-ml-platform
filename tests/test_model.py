from pathlib import Path

import pandas as pd

from src.feature_store.features import build_features

DATA_FILE = Path("DATA/GiveMeSomeCredit/cs-training.csv")


def test_data_file_exists_and_features_are_not_empty() -> None:
    assert DATA_FILE.is_file(), f"Data file not found: {DATA_FILE}"

    raw_data = pd.read_csv(DATA_FILE, index_col=0)
    features = build_features(raw_data)

    assert isinstance(features, pd.DataFrame)
    assert not features.empty
