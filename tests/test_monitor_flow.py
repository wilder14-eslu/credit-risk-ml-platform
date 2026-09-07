from src.orchestrator.monitor import decide_retrain


def test_decide_retrain_true_when_any_trigger_fires() -> None:
    assert decide_retrain(
        rate_triggered=True, feature_drift_triggered=False, performance_triggered=False
    )
    assert decide_retrain(
        rate_triggered=False, feature_drift_triggered=True, performance_triggered=False
    )
    assert decide_retrain(
        rate_triggered=False, feature_drift_triggered=False, performance_triggered=True
    )


def test_decide_retrain_false_when_nothing_fires() -> None:
    assert not decide_retrain(
        rate_triggered=False, feature_drift_triggered=False, performance_triggered=False
    )
