import numpy as np
import pandas as pd
import pytest

from src.risk.cost_analysis import cost_ratio_table
from src.risk.expected_loss import (
    LgdEadUnavailableError,
    expected_loss,
    portfolio_expected_loss,
)
from src.risk.fairness import age_groups, group_fairness_report, wilson_interval
from src.risk.oot import (
    OOT_UNAVAILABLE_MESSAGE,
    OutOfTimeValidationUnavailable,
    find_temporal_column,
    out_of_time_split,
)
from src.risk.portfolio import (
    approval_scenarios,
    gains_curve,
    risk_concentration,
    segment_performance,
)
from src.risk.reject_inference import (
    fuzzy_augmentation,
    parceling,
    reweight_by_acceptance_propensity,
)


def _scored(n: int = 20000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(1, 14, n), 0.001, 0.99)
    return rng.binomial(1, p), p


# ---- expected loss ------------------------------------------------------
def test_expected_loss_formula_and_aggregation() -> None:
    loss = expected_loss([0.1, 0.5], lgd=0.4, ead=[1000.0, 200.0])
    assert loss == pytest.approx([40.0, 40.0])
    summary = portfolio_expected_loss([0.1, 0.5], 0.4, [1000.0, 200.0])
    assert summary["expected_loss"] == pytest.approx(80.0)
    assert summary["expected_loss_rate"] == pytest.approx(80.0 / 1200.0)


def test_expected_loss_refuses_to_invent_lgd_or_ead() -> None:
    with pytest.raises(LgdEadUnavailableError):
        expected_loss([0.1], lgd=None, ead=[100.0])
    with pytest.raises(LgdEadUnavailableError):
        expected_loss([0.1], lgd=0.4, ead=None)


def test_expected_loss_validates_ranges() -> None:
    with pytest.raises(ValueError):
        expected_loss([1.2], 0.4, 100.0)
    with pytest.raises(ValueError):
        expected_loss([0.1], 1.4, 100.0)
    with pytest.raises(ValueError):
        expected_loss([0.1, 0.2], 0.4, [100.0, 100.0, 100.0])


# ---- costos -------------------------------------------------------------
def test_cost_ratio_table_thresholds_fall_as_fn_cost_rises() -> None:
    y, p = _scored()
    rows = cost_ratio_table(y[:12000], p[:12000], y[12000:], p[12000:])
    assert [r["fn_fp_ratio"] for r in rows] == [1, 2, 3, 5, 10, 20]
    thresholds = [r["threshold"] for r in rows]
    assert thresholds == sorted(thresholds, reverse=True)
    rejection = [r["rejection_rate"] for r in rows]
    assert rejection == sorted(rejection)
    for row in rows:
        assert row["approval_rate"] + row["rejection_rate"] == pytest.approx(1.0)
        assert row["expected_cost_per_applicant"] <= row["approve_all_cost_per_applicant"] + 1e-9


# ---- cartera ------------------------------------------------------------
def test_gains_curve_and_concentration() -> None:
    y, p = _scored()
    curve = gains_curve(y, p)
    assert curve["captured_defaults"][-1] == pytest.approx(1.0)
    assert np.all(np.diff(curve["captured_defaults"]) >= 0)
    conc = risk_concentration(y, p)
    assert conc["top_segments"]["top_10pct"]["defaults_captured"] > 0.10
    assert conc["top_segments"]["top_10pct"]["lift"] > 1.0


def test_approval_scenarios_reduce_risk() -> None:
    y, p = _scored()
    rows = approval_scenarios(y, p)
    assert all(r["default_rate_reduction_vs_approve_all"] > 0 for r in rows)
    bad = [r["bad_rate_approved"] for r in rows]
    assert bad == sorted(bad)


def test_segment_performance_flags_small_groups() -> None:
    y, p = _scored(6000)
    seg = np.where(np.arange(6000) < 5900, "big", "tiny")
    rows = {r["segment"]: r for r in segment_performance(y, p, pd.Series(seg), n_boot=50)}
    assert "roc_auc" in rows["big"] and "note" in rows["tiny"]


# ---- equidad ------------------------------------------------------------
def test_wilson_interval_is_valid_for_small_n() -> None:
    low, high = wilson_interval(0, 10)
    assert 0.0 == pytest.approx(low, abs=1e-12) and 0 < high < 0.4
    assert all(np.isnan(v) for v in wilson_interval(0, 0))


def test_fairness_report_shape_and_small_sample_flag() -> None:
    y, p = _scored(10000)
    ages = pd.Series(np.random.default_rng(1).integers(18, 90, 10000))
    groups = age_groups(ages)
    report = group_fairness_report(y, p, groups, threshold=0.16)
    assert report["reference_group"] in {g["group"] for g in report["groups"]}
    assert "causa" in report["caveat"]
    for group in report["groups"]:
        assert 0 <= group["rejection_rate"]["ci_low"] <= group["rejection_rate"]["ci_high"] <= 1
        assert group["n"] > 0
    assert len(report["disparities_vs_reference"]) == len(report["groups"]) - 1


# ---- OOT ----------------------------------------------------------------
def test_oot_unavailable_without_temporal_variable() -> None:
    data = pd.DataFrame({"age": [30, 40], "income": [1.0, 2.0]})
    assert find_temporal_column(data) is None
    with pytest.raises(OutOfTimeValidationUnavailable, match="cannot be performed"):
        out_of_time_split(data)
    assert "current dataset" in OOT_UNAVAILABLE_MESSAGE


def test_oot_split_is_chronological_when_dates_exist() -> None:
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    data = pd.DataFrame({"date": dates, "x": range(100)})
    dev, oot = out_of_time_split(data, oot_fraction=0.2)
    assert dev["date"].max() < oot["date"].min()
    assert len(dev) + len(oot) == 100


# ---- reject inference ---------------------------------------------------
def test_reject_inference_tools_on_synthetic_data() -> None:
    rng = np.random.default_rng(3)
    xa = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    xr = pd.DataFrame({"a": rng.normal(1.0, 1, 500), "b": rng.normal(0, 1, 500)})
    weights = reweight_by_acceptance_propensity(xa, xr)
    assert weights.mean() == pytest.approx(1.0)
    assert weights.min() > 0

    sa = rng.uniform(0, 1, 2000)
    ya = rng.binomial(1, sa * 0.2)
    simulated = parceling(sa, ya, rng.uniform(0, 1, 500), bad_rate_multiplier=2.0)
    assert set(np.unique(simulated)) <= {0, 1}
    assert simulated.mean() > ya.mean()  # el multiplicador endurece a los rechazados

    x_aug, y_aug, w = fuzzy_augmentation(xr, np.full(500, 0.3))
    assert len(x_aug) == 1000 and w[:500].sum() == pytest.approx(150.0)
    assert set(np.unique(y_aug)) == {0, 1}
