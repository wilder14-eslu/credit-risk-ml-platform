import numpy as np
import pandas as pd
import pytest

from src.ml.nested_cv import (
    final_tuning_and_holdout,
    nested_cross_validation,
    tune_hyperparameters,
)
from src.ml.stability import seed_stability, summarize_distribution


def _data(n: int = 1500, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    x.loc[rng.choice(n, 100, replace=False), "b"] = np.nan
    logit = -2.0 + 1.2 * x["a"] - 0.8 * x["c"].fillna(0)
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-logit))))
    return x, y


def test_summarize_distribution_fields() -> None:
    s = summarize_distribution([0.80, 0.82, 0.84, 0.86, 0.88])
    assert s["mean"] == pytest.approx(0.84) and s["median"] == pytest.approx(0.84)
    assert s["iqr"] == s["q3"] - s["q1"]
    assert s["ci_mean_low"] < s["mean"] < s["ci_mean_high"]


def test_tuning_returns_params_and_reasonable_auc() -> None:
    x, y = _data()
    params, auc = tune_hyperparameters("xgboost", x, y.to_numpy(), n_trials=3)
    assert "max_depth" in params and 0.6 < auc < 1.0


def test_nested_cv_structure_and_isolation() -> None:
    x, y = _data()
    result = nested_cross_validation(
        "logistic_regression", x, y, outer_splits=3, inner_splits=2, n_trials=2
    )
    assert len(result["folds"]) == 3
    summary = result["summary"]
    assert summary["outer_tuned_roc_auc"]["n"] == 3
    assert "p_value" in summary["tuned_minus_default"]
    # El mismo procedimiento con la misma semilla es reproducible.
    again = nested_cross_validation(
        "logistic_regression", x, y, outer_splits=3, inner_splits=2, n_trials=2
    )
    assert [f["outer_tuned"]["roc_auc"] for f in result["folds"]] == [
        f["outer_tuned"]["roc_auc"] for f in again["folds"]
    ]


def test_final_holdout_uses_untouched_data() -> None:
    x, y = _data(2000)
    out = final_tuning_and_holdout(
        "logistic_regression", x.iloc[:1500], y.iloc[:1500], x.iloc[1500:], y.iloc[1500:],
        n_trials=2, inner_splits=2,
    )
    assert out["holdout_tuned"]["roc_auc"] > 0.6
    assert "p_value" in out["delong_tuned_vs_default"]


def test_seed_stability_reports_spread() -> None:
    x, y = _data()
    out = seed_stability(x, y, ["logistic_regression"], seeds=(0, 1, 2))
    summary = out["summary"]["logistic_regression"]["roc_auc"]
    assert summary["n"] == 3 and summary["std"] > 0
