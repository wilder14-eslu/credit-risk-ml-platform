from types import SimpleNamespace

from src.api.monitoring import drift_trigger
from src.ml.validate_model import evaluate_and_promote


class FakeClient:
    def __init__(self, current_auc: float) -> None:
        self.current_auc = current_auc
        self.transitioned = False

    def get_latest_versions(self, model_name: str, stages: list[str]):
        return [SimpleNamespace(run_id="production-run")]

    def get_run(self, run_id: str):
        return SimpleNamespace(data=SimpleNamespace(metrics={"roc_auc": self.current_auc}))

    def search_model_versions(self, query: str):
        return [SimpleNamespace(version="2")]

    def transition_model_version_stage(self, **kwargs):
        self.transitioned = True


def test_model_gate_rejects_regression() -> None:
    client = FakeClient(current_auc=0.8)
    assert not evaluate_and_promote(0.79, client=client)
    assert not client.transitioned


def test_model_gate_promotes_improvement() -> None:
    client = FakeClient(current_auc=0.8)
    assert evaluate_and_promote(0.81, client=client)
    assert client.transitioned


def test_drift_trigger() -> None:
    assert drift_trigger([0.8, 0.9, 0.7])
    assert not drift_trigger([0.1, 0.2])