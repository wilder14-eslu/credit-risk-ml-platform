from src.api.monitoring import (
    compute_live_performance,
    performance_degraded_trigger,
    record_outcome,
    record_prediction,
)


def test_compute_live_performance_reports_insufficient_data(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    outcomes_path = tmp_path / "outcomes.jsonl"

    record_prediction(
        {"age": 40}, 0.8, "rechazar", applicant_id="a1", output_path=predictions_path
    )
    record_outcome("a1", True, output_path=outcomes_path)

    result = compute_live_performance(predictions_path, outcomes_path, min_samples=5)
    assert result["status"] == "insufficient_data"
    assert result["n_matched"] == 1


def test_compute_live_performance_scores_matched_records(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    outcomes_path = tmp_path / "outcomes.jsonl"

    matched = [
        ("a1", 0.9, True),
        ("a2", 0.1, False),
        ("a3", 0.8, True),
        ("a4", 0.2, False),
    ]
    for applicant_id, probability, actual_default in matched:
        decision = "rechazar" if probability >= 0.5 else "aprobar"
        record_prediction(
            {"age": 40},
            probability,
            decision,
            applicant_id=applicant_id,
            output_path=predictions_path,
        )
        record_outcome(applicant_id, actual_default, output_path=outcomes_path)

    result = compute_live_performance(predictions_path, outcomes_path, min_samples=4)
    assert result["status"] == "ok"
    assert result["n_matched"] == 4
    assert result["live_roc_auc"] == 1.0


def test_performance_degraded_trigger() -> None:
    assert performance_degraded_trigger(champion_roc_auc=0.85, live_roc_auc=0.70)
    assert not performance_degraded_trigger(champion_roc_auc=0.85, live_roc_auc=0.83)
