import json

import numpy as np
import pandas as pd

from src.pipelines.full_report import _jsonable, profile_dataset


def test_jsonable_handles_numpy_and_nan() -> None:
    payload = {"a": np.float64(0.5), "b": np.int64(3), "c": np.array([1.0, np.nan]),
               "d": float("nan"), "e": (np.float32(1.5),)}
    out = _jsonable(payload)
    assert out == {"a": 0.5, "b": 3, "c": [1.0, None], "d": None, "e": [1.5]}
    json.dumps(out)


def test_profile_dataset_counts_sentinels_and_missing() -> None:
    raw = pd.read_csv("DATA/GiveMeSomeCredit/cs-training.csv", index_col=0, nrows=5000)
    profile = profile_dataset(raw)
    assert profile["rows"] == 5000
    assert 0 < profile["default_prevalence"] < 0.2
    assert profile["features"]["monthly_income"]["missing_rate"] > 0.1
    assert profile["temporal_variable"] is None  # no hay variable temporal: OOT imposible
    assert set(profile["sentinel_codes_96_98"]) == {
        "number_of_time_30_59_days_past_due",
        "number_of_time_60_89_days_past_due",
        "number_of_times_90_days_late",
    }
