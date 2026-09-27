"""Unit tests for the statistical evaluation toolkit (`src.ml.stats`)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.data_pipeline.preprocess import (
    apply_imputation,
    fit_imputation_values,
    prepare_training_data,
)
from src.ml import stats as st


@pytest.fixture
def scored() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.1, 4000)
    strong = y * 1.2 + rng.normal(0, 1, len(y))
    weak = y * 0.3 + rng.normal(0, 1, len(y))
    return y, strong, weak


def test_delong_auc_matches_sklearn(scored) -> None:
    y, strong, _ = scored
    result = st.delong_auc_ci(y, strong)
    assert result["auc"] == pytest.approx(roc_auc_score(y, strong), abs=1e-10)
    assert result["ci_low"] < result["auc"] < result["ci_high"]


def test_delong_test_detects_difference_and_not_self(scored) -> None:
    y, strong, weak = scored
    assert st.delong_roc_test(y, strong, weak)["p_value"] < 1e-6
    assert st.delong_roc_test(y, strong, strong)["p_value"] == pytest.approx(1.0)


def test_bootstrap_ci_contains_estimate(scored) -> None:
    y, strong, _ = scored
    ci = st.bootstrap_ci(y, strong, roc_auc_score, n_boot=200)
    assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"]


def test_corrected_ttest_is_more_conservative_than_naive() -> None:
    a = np.array([0.86, 0.87, 0.865, 0.862, 0.868])
    b = a - np.array([0.002, 0.001, 0.003, 0.002, 0.001])
    corrected = st.corrected_resampled_ttest(a, b, n_train=80, n_test=20)
    diff = a - b
    naive_t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
    assert abs(corrected["t"]) < abs(naive_t)


def test_corrected_ci_is_wider_than_naive() -> None:
    values = [0.86, 0.87, 0.865, 0.862, 0.868]
    naive = st.t_confidence_interval(values)
    corrected = st.t_confidence_interval(values, test_train_ratio=0.25)
    assert corrected["ci_high"] - corrected["ci_low"] > naive["ci_high"] - naive["ci_low"]


def test_calibration_metrics_on_perfectly_calibrated_scores() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.5, 50_000)
    y = rng.binomial(1, p)
    assert st.expected_calibration_error(y, p) < 0.01
    calib = st.calibration_slope_intercept(y, p)
    assert calib["slope"] == pytest.approx(1.0, abs=0.08)


def test_decile_table_capture_reaches_one(scored) -> None:
    y, strong, _ = scored
    table = st.decile_table(y, strong)
    assert len(table) == 10
    assert table[-1]["cum_bad_capture"] == pytest.approx(1.0)
    assert table[0]["bad_rate"] > table[-1]["bad_rate"]


def test_threshold_analysis_reports_bayes_threshold(scored) -> None:
    y, strong, _ = scored
    p = 1 / (1 + np.exp(-strong))
    result = st.threshold_analysis(y, p, cost_fn=5, cost_fp=1)
    assert result["bayes_threshold"] == pytest.approx(1 / 6)
    assert 0 < result["min_cost"]["threshold"] < 1


def test_imputation_is_fitted_on_train_split_only() -> None:
    rng = np.random.default_rng(2)
    n = 1000
    raw = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 1, n),
            "age": rng.integers(20, 80, n).astype(float),
            "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 3, n).astype(float),
            "DebtRatio": rng.uniform(0, 1, n),
            "MonthlyIncome": np.where(rng.uniform(size=n) < 0.2, np.nan, rng.uniform(1e3, 1e4, n)),
            "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 10, n).astype(float),
            "NumberOfTimes90DaysLate": rng.integers(0, 2, n).astype(float),
            "NumberRealEstateLoansOrLines": rng.integers(0, 3, n).astype(float),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 2, n).astype(float),
            "NumberOfDependents": rng.integers(0, 4, n).astype(float),
            "SeriousDlqin2yrs": rng.binomial(1, 0.2, n),
        }
    )
    x_train, x_test, _, _ = prepare_training_data(raw)
    assert not x_train.isna().any().any()
    assert not x_test.isna().any().any()
    # The median of a median-imputed column equals the original train median.
    values = fit_imputation_values(x_train)
    assert apply_imputation(pd.DataFrame([{"monthly_income": np.nan}]), values)[
        "monthly_income"
    ].iloc[0] == pytest.approx(values["monthly_income"])
