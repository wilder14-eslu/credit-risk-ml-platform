import json

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import target_name
from credit_risk.models import training as T
from credit_risk.models.candidates import available_candidates, build_estimator, default_params
from credit_risk.models.credit_model import risk_band


def test_splits_are_stratified_and_disjoint(splits, canonical):
    rate = canonical[target_name()].mean()
    for y in (splits.y_train, splits.y_val, splits.y_test):
        assert abs(y.mean() - rate) < 0.01
    assert not set(splits.x_train.index) & set(splits.x_test.index)


def test_benchmark_compares_candidates_on_same_split(splits):
    table, models = T.run_benchmark(splits, ["logistic_regression", "xgboost"])
    assert set(table["algorithm"]) == {"logistic_regression", "xgboost"}
    assert table["test_roc_auc"].is_monotonic_decreasing
    for col in ("test_pr_auc", "test_ks", "test_brier", "latency_ms", "fit_diagnosis"):
        assert col in table
    assert T.select_best(table) in models


def test_models_beat_random_and_pass_gates(xgb_model, splits):
    result = T.evaluate(xgb_model, splits)
    assert result["test_roc_auc"] > 0.80
    passed, reasons = T.check_quality_gates(result)
    assert passed, reasons


def test_quality_gate_rejects_weak_model():
    weak = {"test_roc_auc": 0.6, "auc_gap": 0.2, "test_brier": 0.2, "latency_ms": 50}
    passed, reasons = T.check_quality_gates(weak)
    assert not passed and len(reasons) == 4


def test_champion_vs_challenger_rules():
    assert T.champion_vs_challenger(0.86, None)[0]
    assert T.champion_vs_challenger(0.865, 0.86)[0]
    assert not T.champion_vs_challenger(0.861, 0.86)[0]


def test_optuna_tuning_returns_params(splits):
    params = T.tune("logistic_regression", splits, n_trials=3, timeout=60)
    assert "C" in params


def test_predict_frame_contract(xgb_model, splits):
    frame = xgb_model.predict_frame(splits.x_test.head(20))
    assert list(frame.columns) == ["probability", "decision", "risk_band", "top_factors"]
    assert frame["probability"].between(0, 1).all()
    assert set(frame["decision"]) <= {"APROBAR", "RECHAZAR"}
    factors = json.loads(frame["top_factors"].iloc[0])
    assert len(factors) == 3 and {"feature", "impact", "direction"} <= set(factors[0])


def test_contributions_explain_log_odds_for_logistic(lr_model, splits):
    x = splits.x_test.head(50)
    contrib = lr_model.contributions(x).sum(axis=1).to_numpy()
    intercept = lr_model.estimator.named_steps["clf"].intercept_[0]
    logit = np.log(lr_model.predict_proba(x) / (1 - lr_model.predict_proba(x)))
    np.testing.assert_allclose(contrib + intercept, logit, rtol=1e-6, atol=1e-6)


def test_model_handles_missing_income(xgb_model, splits):
    row = splits.x_test.head(1).copy()
    row["monthly_income"] = np.nan
    row["number_dependents"] = np.nan
    assert 0 <= xgb_model.predict_proba(row)[0] <= 1


def test_training_report_contains_sections(splits, xgb_model):
    table, _ = T.run_benchmark(splits, ["logistic_regression"])
    result = T.evaluate(xgb_model, splits)
    md = T.training_report(table, "xgboost", result, xgb_model.global_importance(splits.x_test.head(200)))
    assert "Benchmark de candidatos" in md and "Train vs test" in md and "SHAP" in md


@pytest.mark.parametrize("p,band", [(0.01, "A"), (0.07, "B"), (0.15, "C"), (0.3, "D"), (0.6, "E")])
def test_risk_band(p, band):
    assert risk_band(p) == band


def test_candidates_factory():
    assert "logistic_regression" in available_candidates(["logistic_regression"])
    est = build_estimator("logistic_regression", default_params("logistic_regression"))
    assert hasattr(est, "fit")
    with pytest.raises(ValueError):
        build_estimator("svm")


def test_pyfunc_roundtrip(tmp_path, lr_model, splits):
    mlflow = pytest.importorskip("mlflow")
    import joblib
    from mlflow.models import infer_signature

    from credit_risk.config import PROJECT_ROOT
    from credit_risk.models.credit_model import CreditRiskPyfunc

    sample = splits.x_test.head(5).reset_index(drop=True)
    signature = infer_signature(sample, lr_model.predict_frame(sample), params={"explain": True})
    model_file = tmp_path / "credit_model.joblib"
    joblib.dump(lr_model, model_file)
    target = tmp_path / "pyfunc"
    mlflow.pyfunc.save_model(
        path=str(target),
        python_model=CreditRiskPyfunc(),
        artifacts={"credit_model": str(model_file)},
        code_paths=[str(PROJECT_ROOT / "src" / "credit_risk")],
        signature=signature,
    )
    loaded = mlflow.pyfunc.load_model(str(target))
    out = loaded.predict(sample, params={"explain": False})
    assert isinstance(out, pd.DataFrame) and len(out) == 5
    assert (out["top_factors"] == "[]").all()
