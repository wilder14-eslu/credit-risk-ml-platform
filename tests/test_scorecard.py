import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.risk.scorecard import (
    MISSING_CODE,
    CreditScorecard,
    fit_binning,
    information_value_label,
)


def _synthetic(n: int = 6000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    util = rng.uniform(0, 1.5, n)
    age = rng.integers(21, 75, n).astype(float)
    late = rng.choice([0, 1, 2, 96], n, p=[0.85, 0.09, 0.05, 0.01]).astype(float)
    noise = rng.normal(size=n)
    logit = -3.2 + 2.2 * util - 0.03 * (age - 45) + 0.8 * (late > 0) + 0.3 * (late == 96)
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    x = pd.DataFrame({"util": util, "age": age, "late": late, "noise": noise})
    x.loc[rng.choice(n, 300, replace=False), "age"] = np.nan
    return x, y


def test_woe_sign_follows_risk_direction() -> None:
    x, y = _synthetic()
    binning = fit_binning("util", x["util"], y)
    woes = [row["woe"] for row in binning.table if row["code"] < MISSING_CODE]
    assert woes[0] > 0 > woes[-1]  # baja utilización = menor riesgo = WoE positivo
    assert binning.iv > 0.3


def test_iv_ranks_informative_above_noise() -> None:
    x, y = _synthetic()
    scorecard = CreditScorecard(special_values={"late": (96.0,)}).fit(x, y)
    ivs = {row["feature"]: row["iv"] for row in scorecard.iv_table()}
    assert ivs["util"] > ivs["noise"]
    assert ivs["noise"] < 0.02
    assert information_value_label(ivs["noise"]) == "inútil"


def test_missing_and_special_get_their_own_bins() -> None:
    x, y = _synthetic()
    binning = fit_binning("age", x["age"], y)
    assert any(row["code"] == MISSING_CODE for row in binning.table)
    late = fit_binning("late", x["late"], y, special_values=(96.0,))
    assert any(row["bin"] == "especial = 96" for row in late.table)


def test_score_pd_roundtrip_and_monotonicity() -> None:
    x, y = _synthetic()
    scorecard = CreditScorecard().fit(x, y)
    p = np.array([0.01, 0.05, 0.2, 0.6])
    assert scorecard.score_to_pd(scorecard.pd_to_score(p)) == pytest.approx(p, rel=1e-6)
    assert np.all(np.diff(scorecard.pd_to_score(p)) < 0)


def test_scaling_parameters_hold() -> None:
    scorecard = CreditScorecard(pdo=20, base_score=600, base_odds=50)
    base_pd = 1 / (1 + 50)
    assert scorecard.pd_to_score(base_pd) == pytest.approx(600.0)
    assert scorecard.pd_to_score(base_pd / 2) - 600 == pytest.approx(
        20 * np.log2((1 - base_pd / 2) / (base_pd / 2) / 50), rel=1e-6
    )


def test_holdout_is_only_transformed_and_model_discriminates() -> None:
    x, y = _synthetic(9000)
    scorecard = CreditScorecard(special_values={"late": (96.0,)}).fit(x.iloc[:6000], y[:6000])
    auc = roc_auc_score(y[6000:], scorecard.predict_pd(x.iloc[6000:]))
    assert auc > 0.7


def test_points_table_reconstructs_score() -> None:
    x, y = _synthetic()
    scorecard = CreditScorecard().fit(x, y)
    points = {
        (r["feature"], r["code"]): r["points"] for r in scorecard.points_table()
    }
    row = x.iloc[[0]]
    total = sum(
        points[name, int(scorecard.binnings[name].assign(row[name])[0])]
        for name in scorecard.features
    )
    assert total == pytest.approx(float(scorecard.score(row)[0]), abs=1e-6)


def test_unfitted_scorecard_raises() -> None:
    with pytest.raises(RuntimeError):
        CreditScorecard().predict_proba(pd.DataFrame({"a": [1.0]}))


def test_heavily_tied_counts_keep_their_risk_bins() -> None:
    # 94 % de ceros: los cuantiles coinciden; aun así 0 / 1 / 2+ deben separarse.
    rng = np.random.default_rng(5)
    late = rng.choice([0, 1, 2, 3, 4, 5], 50000, p=[0.94, 0.035, 0.012, 0.007, 0.004, 0.002])
    late = np.r_[late, np.arange(6, 18)].astype(float)  # > 10 valores distintos
    y = rng.binomial(1, np.where(late == 0, 0.04, 0.35))
    binning = fit_binning("late", pd.Series(late), y)
    assert len(binning.edges) >= 2
    assert binning.iv > 0.3
