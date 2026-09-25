import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import base_feature_names, platform_config, target_name
from credit_risk.data.simulation import apply_scenario
from credit_risk.models.metrics import classification_metrics
from credit_risk.monitoring.ab_testing import assign_variant, bootstrap_auc_diff, evaluate_ab_test
from credit_risk.monitoring.concept_drift import concept_drift_report, ddm, page_hinkley, two_proportion_ztest
from credit_risk.monitoring.data_drift import build_reference_profile, feature_drift, psi
from credit_risk.monitoring.decision import retrain_decision

CFG = platform_config()["monitoring"]


def test_psi_zero_for_identical_and_large_for_shifted():
    e = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(e, e) == pytest.approx(0.0)
    assert psi(e, np.array([0.7, 0.1, 0.1, 0.1])) > 0.25


@pytest.fixture(scope="module")
def profile(splits, xgb_model):
    x = splits.x_train[list(base_feature_names())]
    return build_reference_profile(x, xgb_model.predict_proba(splits.x_train))


def test_profile_is_json_serializable(profile):
    restored = json.loads(json.dumps(profile))
    assert "__score__" in restored["features"]


def test_no_drift_on_same_distribution(profile, splits, xgb_model):
    cur = splits.x_test[list(base_feature_names())].copy()
    cur["__score__"] = xgb_model.predict_proba(splits.x_test)
    table = feature_drift(profile, cur)
    assert (table["psi"] < CFG["psi_alert"]).all()


def test_covariate_shift_is_detected(profile, splits, xgb_model, canonical):
    shifted = apply_scenario(canonical.loc[splits.x_test.index], "covariate", intensity=1.5, seed=3)
    cur = shifted[list(base_feature_names())].copy()
    cur["__score__"] = xgb_model.predict_proba(shifted)
    table = feature_drift(profile, cur)
    alerts = table[table["status"] == "alerta"]["feature"].tolist()
    assert {"monthly_income", "debt_ratio"} <= set(alerts)


def test_ddm_detects_error_rate_increase():
    rng = np.random.default_rng(0)
    stream = np.concatenate([rng.random(2000) < 0.05, rng.random(2000) < 0.30]).astype(int)
    result = ddm(stream)
    assert result.state == "drift" and result.index > 2000
    assert ddm((rng.random(3000) < 0.05).astype(int)).state != "drift"


def test_page_hinkley_detects_mean_increase():
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.normal(0.2, 0.05, 1000), rng.normal(1.0, 0.05, 1000)])
    assert page_hinkley(values, lamb=20)["drift"]
    assert not page_hinkley(rng.normal(0.2, 0.05, 2000), lamb=20)["drift"]


def test_two_proportion_ztest():
    z, p = two_proportion_ztest(150, 1000, 70, 1000)
    assert z > 0 and p < 0.001
    assert two_proportion_ztest(70, 1000, 70, 1000)[1] == pytest.approx(1.0)


def test_concept_drift_detected_without_feature_drift(xgb_model, splits, canonical):
    test = canonical.loc[splits.x_test.index]
    reference = classification_metrics(splits.y_test, xgb_model.predict_proba(splits.x_test))
    stable = concept_drift_report(test[target_name()], xgb_model.predict_proba(test), reference, CFG)
    assert not stable["concept_drift"]

    drifted = apply_scenario(test, "concept", intensity=1.5, seed=7)
    report = concept_drift_report(drifted[target_name()], xgb_model.predict_proba(drifted), reference, CFG)
    assert report["concept_drift"]
    assert report["signals"]["auc_degradation"]


def test_retrain_decision_policy():
    drift = pd.DataFrame({"feature": ["a", "b", "c", "prediction"], "psi": [0.3, 0.4, 0.5, 0.05]})
    decision = retrain_decision(drift, None, CFG, n_rows=5000)
    assert decision["retrain"] and decision["severity"] == "alerta"

    calm = pd.DataFrame({"feature": ["a", "prediction"], "psi": [0.01, 0.01]})
    assert not retrain_decision(calm, None, CFG, n_rows=5000)["retrain"]
    assert retrain_decision(drift, None, CFG, n_rows=10)["severity"] == "sin_datos"

    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    cooled = retrain_decision(drift, None, CFG, n_rows=5000, last_retrain_at=recent)
    assert not cooled["retrain"] and "cooldown" in cooled["reasons"][-1]

    concept = {"signals": {"auc_degradation": True}, "concept_drift": True}
    assert retrain_decision(calm, concept, CFG, n_rows=5000)["severity"] == "critico"


def test_ab_assignment_is_sticky_and_respects_traffic():
    ids = [f"APP-{i}" for i in range(20_000)]
    variants = [assign_variant(i, 0.2) for i in ids]
    share = variants.count("challenger") / len(ids)
    assert 0.18 < share < 0.22
    assert all(assign_variant(i, 0.2) == v for i, v in zip(ids[:500], variants[:500]))
    assert assign_variant("x", 0.0) == "champion"


def _arm(rng, n, strength):
    y = (rng.random(n) < 0.07).astype(int)
    p = np.clip(0.05 + strength * y + rng.normal(0, 0.1, n), 0.001, 0.999)
    return {"y": y, "p": p, "approved": p < 0.3}


def test_ab_decisions():
    cfg = platform_config()["ab_testing"]
    rng = np.random.default_rng(0)
    small = evaluate_ab_test(_arm(rng, 100, 0.2), _arm(rng, 100, 0.3), cfg)
    assert small["decision"] == "continuar"

    better = evaluate_ab_test(_arm(rng, 4000, 0.10), _arm(rng, 4000, 0.30), cfg)
    assert better["decision"] == "promover" and better["ci_low"] > 0

    worse = evaluate_ab_test(_arm(rng, 4000, 0.30), _arm(rng, 4000, 0.05), cfg)
    assert worse["decision"] == "detener"


def test_bootstrap_ci_contains_zero_for_equal_models():
    rng = np.random.default_rng(5)
    a, b = _arm(rng, 3000, 0.2), _arm(rng, 3000, 0.2)
    res = bootstrap_auc_diff(a["y"], a["p"], b["y"], b["p"], iterations=200)
    assert res["ci_low"] < 0 < res["ci_high"]
