import json

import numpy as np

from src.governance.champion_challenger import compare_champion_challenger
from src.governance.provenance import collect_provenance, file_sha256, library_versions


def _scores(seed: int = 0, n: int = 6000):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.1, n)
    strong = y * 1.6 + rng.normal(0, 1, n)
    weak = y * 0.5 + rng.normal(0, 1, n)
    squash = lambda z: 1 / (1 + np.exp(-(z - 2.6)))  # noqa: E731
    return y, squash(strong), squash(weak)


def test_provenance_hashes_dataset_and_is_json_serializable(tmp_path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    info = collect_provenance(csv, {"seed": 42})
    assert info["dataset"]["sha256"] == file_sha256(csv) and len(info["dataset"]["sha256"]) == 64
    assert info["configuration"] == {"seed": 42}
    json.dumps(info)
    assert "numpy" in library_versions()


def test_clear_improvement_is_eligible_but_never_auto_promoted() -> None:
    y, strong, weak = _scores()
    out = compare_champion_challenger(y, weak, strong, "weak", "strong",
                                      max_calibration_slope_error_worsening=10)
    assert out["challenger_eligible_for_review"]
    assert out["auto_promote"] is False and out["requires_human_approval"] is True


def test_worse_challenger_is_not_eligible() -> None:
    y, strong, weak = _scores()
    out = compare_champion_challenger(y, strong, weak)
    assert not out["challenger_eligible_for_review"]
    assert "Mantener el champion" in out["recommendation"]


def test_significant_but_immaterial_gain_is_not_eligible() -> None:
    y, strong, _ = _scores(n=200000)
    rng = np.random.default_rng(9)
    nearly_same = np.clip(strong + rng.normal(0, 0.01, len(strong)), 0, 1)
    out = compare_champion_challenger(y, nearly_same, strong, practical_margin=0.05)
    assert not out["challenger_eligible_for_review"]


def test_shap_helpers_on_a_small_tree_model() -> None:
    import pandas as pd
    from xgboost import XGBClassifier

    from src.ml.explainability import (
        association_direction,
        global_importance,
        interaction_strength,
        shap_sample,
    )

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(800, 3)), columns=["a", "b", "noise"])
    y = (x["a"] + 0.5 * x["b"] * x["a"] + rng.normal(0, 0.5, 800) > 0.5).astype(int)
    model = XGBClassifier(n_estimators=40, max_depth=3, random_state=0).fit(x, y)
    sample, values = shap_sample(model, x, n=300)
    importance = global_importance(values, list(x.columns))
    assert importance[0]["feature"] == "a" and importance[-1]["feature"] == "noise"
    assert association_direction(sample, values)["a"] > 0.5  # más 'a', más riesgo (asociación)
    inter = interaction_strength(model, x, n=200)
    assert inter is not None and inter.shape == (3, 3)
