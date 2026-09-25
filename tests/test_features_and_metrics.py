import numpy as np
import pandas as pd
import pytest

from credit_risk.features.engineering import DERIVED_FEATURES, MedianImputer, build_features, feature_names
from credit_risk.models import metrics as M


def test_build_features_derived_columns(canonical):
    feats = build_features(canonical)
    assert list(feats.columns) == list(feature_names())
    row = canonical.iloc[0]
    expected = (
        row["number_of_time_30_59_days_past_due"]
        + row["number_of_time_60_89_days_past_due"]
        + row["number_of_times_90_days_late"]
    )
    assert feats.iloc[0]["total_past_due"] == expected
    assert set(DERIVED_FEATURES) <= set(feats.columns)


def test_build_features_requires_base_columns():
    with pytest.raises(ValueError, match="Faltan"):
        build_features(pd.DataFrame({"age": [30]}))


def test_imputer_fills_with_train_medians(canonical):
    feats = build_features(canonical)
    imputer = MedianImputer().fit(feats)
    filled = imputer.transform(feats)
    assert not filled.isna().any().any()
    assert imputer.medians_["monthly_income"] == pytest.approx(feats["monthly_income"].median())
    with pytest.raises(RuntimeError):
        MedianImputer().transform(feats)


def test_ks_and_auc_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = M.classification_metrics(y, p)
    assert m["roc_auc"] == 1.0
    assert m["gini"] == 1.0
    assert m["ks"] == 1.0


def test_calibration_error_is_small_for_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20_000)
    y = (rng.uniform(0, 1, 20_000) < p).astype(int)
    assert M.expected_calibration_error(y, p) < 0.02
    assert M.expected_calibration_error(y, np.clip(p + 0.3, 0, 1)) > 0.1


def test_optimal_threshold_moves_down_when_false_negatives_are_costly():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    p = np.clip(y * 0.3 + rng.normal(0.35, 0.15, 5000), 0, 1)
    cheap = M.optimal_threshold(y, p, cost_fn=1, cost_fp=1)
    costly = M.optimal_threshold(y, p, cost_fn=10, cost_fp=1)
    assert costly < cheap


@pytest.mark.parametrize(
    "train,test,label",
    [(0.95, 0.80, "sobreajuste"), (0.60, 0.58, "subajuste"), (0.86, 0.85, "buen_ajuste")],
)
def test_diagnose_fit(train, test, label):
    assert M.diagnose_fit(train, test)["fit_diagnosis"] == label
