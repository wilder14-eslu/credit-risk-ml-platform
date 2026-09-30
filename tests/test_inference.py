import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from src.inference.comparison import pairwise_cv_comparison, pairwise_delong_comparison
from src.inference.effect_sizes import interpret_difference, paired_cohens_dz
from src.inference.multiple_testing import holm_adjust


def test_holm_matches_statsmodels() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.2, size=12)
    expected = multipletests(p, method="holm")[1]
    assert holm_adjust(p) == pytest.approx(expected)


def test_holm_preserves_order_and_monotonicity() -> None:
    adjusted = holm_adjust([0.04, 0.001, 0.03])
    assert adjusted == pytest.approx([0.06, 0.003, 0.06])
    assert all(0.0 <= value <= 1.0 for value in adjusted)


def test_holm_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        holm_adjust([0.2, 1.5])
    assert holm_adjust([]) == []


def test_cohens_dz_and_degenerate_cases() -> None:
    assert paired_cohens_dz([1.0, 1.0, 1.0]) == pytest.approx(0.0)
    assert paired_cohens_dz([1.0]) == pytest.approx(0.0)
    assert paired_cohens_dz([1.0, 2.0, 3.0]) == pytest.approx(2.0)


def test_interpretation_separates_statistical_from_practical() -> None:
    assert "irrelevante" in interpret_difference(0.001, 0.001)
    assert "material a favor de A" in interpret_difference(0.02, 0.001)
    assert "equivalentes" in interpret_difference(0.001, 0.5)
    assert "no concluyente" in interpret_difference(0.02, 0.5)


def test_pairwise_cv_comparison_applies_holm_and_has_all_pairs() -> None:
    rng = np.random.default_rng(1)
    base = 0.86 + rng.normal(0, 0.004, 15)
    scores = {
        "a": base,
        "b": base - 0.0005 + rng.normal(0, 0.001, 15),
        "c": base - 0.03 + rng.normal(0, 0.001, 15),
    }
    rows = pairwise_cv_comparison(scores, n_train=96000, n_test=24000)
    assert len(rows) == 3
    for row in rows:
        assert row["p_adjusted_holm"] >= row["p_value"]
    ac = next(r for r in rows if (r["model_a"], r["model_b"]) == ("a", "c"))
    assert ac["p_adjusted_holm"] < 0.05
    assert ac["ci_low"] > 0 and "material" in ac["interpretation"]


def test_pairwise_delong_detects_clearly_better_model() -> None:
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.1, 4000)
    strong = y * 1.5 + rng.normal(0, 1, 4000)
    weak = y * 0.3 + rng.normal(0, 1, 4000)
    rows = pairwise_delong_comparison(y, {"strong": strong, "weak": weak})
    assert rows[0]["delta"] > 0
    assert rows[0]["p_adjusted_holm"] < 0.05
    assert rows[0]["ci_low"] < rows[0]["delta"] < rows[0]["ci_high"]
