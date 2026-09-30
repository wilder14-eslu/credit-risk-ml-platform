import numpy as np
import pandas as pd
import pytest

from src.risk.stress import apply_scenario, load_scenarios, run_stress_test


class _Monotone:
    """Modelo de juguete: PD creciente con utilización y atrasos."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        z = -3 + 1.5 * x["revolving_utilization_unsecured"] + 0.5 * x[
            "number_of_time_30_59_days_past_due"
        ] - 0.0001 * x["monthly_income"]
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])


def _features(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "revolving_utilization_unsecured": rng.uniform(0, 1, n),
            "number_of_time_30_59_days_past_due": rng.poisson(0.2, n).astype(float),
            "number_of_time_60_89_days_past_due": rng.poisson(0.1, n).astype(float),
            "debt_ratio": rng.uniform(0, 1, n),
            "monthly_income": rng.uniform(1000, 8000, n),
        }
    )


def test_config_defines_three_named_scenarios() -> None:
    config = load_scenarios()
    assert set(config["scenarios"]) == {"base", "moderate_stress", "severe_stress"}


def test_apply_scenario_does_not_mutate_input_and_is_reproducible() -> None:
    x = _features()
    original = x.copy()
    scenario = load_scenarios()["scenarios"]["severe_stress"]
    a = apply_scenario(x, scenario, seed=1)
    b = apply_scenario(x, scenario, seed=1)
    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(a, b)
    assert a["monthly_income"].mean() == pytest.approx(0.75 * x["monthly_income"].mean())


def test_base_is_identity_and_stress_is_monotone() -> None:
    x = _features()
    rows = {r["scenario"]: r for r in run_stress_test(_Monotone(), x, threshold=0.16)}
    base = rows["base"]
    assert base["mean_pd_change_vs_base"] == pytest.approx(0.0)
    assert rows["severe_stress"]["mean_pd"] > rows["moderate_stress"]["mean_pd"] > base["mean_pd"]
    assert rows["severe_stress"]["rejection_rate"] >= rows["moderate_stress"]["rejection_rate"]
    assert "expected_loss" not in base  # sin LGD/EAD no hay pérdida esperada


def test_expected_loss_only_with_explicit_lgd_and_ead() -> None:
    x = _features(100)
    rows = run_stress_test(_Monotone(), x, 0.16, lgd=0.45, ead=1000.0)
    assert all("expected_loss" in r for r in rows)
    assert rows[-1]["expected_loss"] > rows[0]["expected_loss"]


def test_unknown_variable_raises() -> None:
    with pytest.raises(KeyError):
        apply_scenario(_features(10), {"multiply": {"nope": 2.0}})
