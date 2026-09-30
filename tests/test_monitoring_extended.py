import numpy as np
import pandas as pd
import pytest

from src.monitoring.business import business_drift, score_drift
from src.monitoring.performance import performance_drift, performance_panel
from src.monitoring.quality import data_quality_report

FEATURES = [
    "revolving_utilization_unsecured", "age", "number_of_time_30_59_days_past_due",
    "debt_ratio", "monthly_income", "number_open_credit_lines",
    "number_of_times_90_days_late", "number_real_estate_loans",
    "number_of_time_60_89_days_past_due", "number_dependents",
]


def _frame(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({c: rng.uniform(1, 50, n) for c in FEATURES})
    frame["age"] = rng.uniform(25, 80, n)  # dentro de [18, 120]
    return frame


def test_quality_clean_data_passes() -> None:
    report = data_quality_report(_frame())
    assert report["quality_ok"] and not report["schema"]["schema_drift"]


def test_quality_detects_schema_missingness_and_invalid_values() -> None:
    frame = _frame().drop(columns=["debt_ratio"])
    frame["surprise"] = 1
    frame.loc[:150, "monthly_income"] = np.nan
    frame.loc[0, "age"] = 5  # por debajo del mínimo (18)
    report = data_quality_report(frame, reference_missing_rates={"monthly_income": 0.02})
    assert report["schema"]["missing_columns"] == ["debt_ratio"]
    assert report["schema"]["unexpected_columns"] == ["surprise"]
    assert report["missingness"]["monthly_income"]["alert"]
    assert report["invalid_values"]["age"]["below_min"] == 1
    assert not report["quality_ok"]


def test_score_drift_stable_vs_shifted() -> None:
    rng = np.random.default_rng(1)
    ref = rng.beta(1, 14, 20000)
    same = rng.beta(1, 14, 20000)
    shifted = rng.beta(2, 10, 20000)
    assert not score_drift(ref, same)["drift_detected"]
    assert score_drift(ref, shifted)["drift_detected"]


def test_business_drift_flags_rejection_shift_and_reports_composition() -> None:
    rng = np.random.default_rng(2)
    ref = rng.beta(1, 14, 20000)
    worse = np.clip(ref * 2.0, 0, 1)
    report = business_drift(ref, worse, 0.16, current_outcomes=rng.binomial(1, worse))
    assert report["rejection_drift"] and report["rejection_rate_change_pp"] > 0
    assert sum(report["portfolio_composition_by_reference_decile"]) == pytest.approx(1.0)
    assert report["observed_default_rate"] > 0


def test_performance_panel_and_drift_flags() -> None:
    rng = np.random.default_rng(3)
    p = rng.beta(1, 14, 20000)
    y = rng.binomial(1, p)
    reference = performance_panel(y, p)
    assert 0.5 < reference["roc_auc"] < 1 and 0.8 < reference["calibration_slope"] < 1.2
    noisy = performance_panel(y, rng.permutation(p))  # scores sin relación con y
    drift = performance_drift(reference, noisy)
    assert drift["flags"]["auc_degraded"] and drift["drift_detected"]
    assert not performance_drift(reference, reference)["drift_detected"]


def test_performance_panel_requires_both_classes() -> None:
    with pytest.raises(ValueError):
        performance_panel([0, 0, 0], [0.1, 0.2, 0.3])
