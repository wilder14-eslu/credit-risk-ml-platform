from types import SimpleNamespace

import pytest

from src.api.monitoring import drift_trigger
from src.ml.validate_model import evaluate_candidate, promote_candidate
from src.monitoring.decision import build_retraining_recommendation


class FakeClient:
    def __init__(self, current_auc: float, has_champion: bool = True) -> None:
        self.current_auc = current_auc
        self.has_champion = has_champion
        self.transitioned = False

    def get_latest_versions(self, model_name: str, stages: list[str]):
        return [SimpleNamespace(run_id="production-run")] if self.has_champion else []

    def get_run(self, run_id: str):
        return SimpleNamespace(data=SimpleNamespace(metrics={"roc_auc": self.current_auc}))

    def search_model_versions(self, query: str):
        return [SimpleNamespace(version="2")]

    def transition_model_version_stage(self, **kwargs):
        self.transitioned = True


def test_gate_rejects_regression_without_touching_registry() -> None:
    client = FakeClient(current_auc=0.8)
    assert not evaluate_candidate(0.79, client=client)["eligible"]
    assert not client.transitioned


def test_evaluating_an_improvement_never_promotes_by_itself() -> None:
    client = FakeClient(current_auc=0.8)
    verdict = evaluate_candidate(0.81, client=client)
    assert verdict["eligible"] and verdict["champion_roc_auc"] == pytest.approx(0.8)
    assert not client.transitioned  # elegible no significa desplegado


def test_first_model_is_eligible_when_no_champion_exists() -> None:
    assert evaluate_candidate(0.7, client=FakeClient(0.0, has_champion=False))["eligible"]


def test_promotion_requires_named_human_approval() -> None:
    client = FakeClient(current_auc=0.8)
    with pytest.raises(PermissionError):
        promote_candidate(approved_by="  ", client=client)
    assert not client.transitioned
    assert promote_candidate(approved_by="model-risk-committee", client=client) == "2"
    assert client.transitioned


def test_drift_never_produces_an_auto_deploy_action() -> None:
    recommendation = build_retraining_recommendation(
        {"feature_drift": True, "performance": True, "rate": False}
    )
    assert recommendation["open_retraining_candidate"]
    assert recommendation["signals_fired"] == ["feature_drift", "performance"]
    assert recommendation["auto_deploy"] is False
    assert recommendation["requires_human_approval"] is True
    quiet = build_retraining_recommendation({"feature_drift": False})
    assert not quiet["open_retraining_candidate"] and quiet["auto_deploy"] is False


def test_drift_trigger() -> None:
    assert drift_trigger([0.8, 0.9, 0.7])
    assert not drift_trigger([0.1, 0.2])
