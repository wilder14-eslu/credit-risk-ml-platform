"""Prediction monitoring events for audit, data drift and live performance."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PREDICTION_LOG = Path("data/processed/predictions.jsonl")
DEFAULT_OUTCOME_LOG = Path("data/processed/outcomes.jsonl")


def record_prediction(
    features: dict[str, Any],
    probability: float,
    decision: str,
    applicant_id: str | None = None,
    output_path: Path = DEFAULT_PREDICTION_LOG,
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - keep py3.10 compatible
        "applicant_id": applicant_id,
        "features": features,
        "probability": probability,
        "decision": decision,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as event_file:
        event_file.write(json.dumps(event) + "\n")


def record_outcome(
    applicant_id: str,
    actual_default: bool,
    output_path: Path = DEFAULT_OUTCOME_LOG,
) -> None:
    """Record the real-world result for an applicant scored earlier.

    In credit risk this is the piece that only shows up months later (did
    the applicant actually default?), reported back via
    `POST /api/v1/outcomes`. It is what lets `compute_live_performance`
    measure the model's *actual* performance in production instead of just
    the input-feature drift `drift_trigger`/`compute_feature_drift` catch.
    """
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "applicant_id": applicant_id,
        "actual_default": bool(actual_default),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as event_file:
        event_file.write(json.dumps(event) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_recent_predictions(
    path: Path = DEFAULT_PREDICTION_LOG,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    events = _read_jsonl(path)
    return events[-limit:] if limit else events


def drift_trigger(
    probabilities: list[float],
    threshold: float = 0.5,
) -> bool:
    """Return whether the observed default rate exceeds the retraining threshold.

    This is a cheap, always-available proxy signal (no ground truth needed):
    it does not by itself prove the input *distributions* moved -- for that,
    see `src.monitoring.drift.compute_feature_drift` (PSI), which compares
    actual feature distributions against the training reference.
    """
    if not probabilities:
        return False
    return sum(probability >= threshold for probability in probabilities) / len(
        probabilities
    ) > threshold


def compute_live_performance(
    predictions_path: Path = DEFAULT_PREDICTION_LOG,
    outcomes_path: Path = DEFAULT_OUTCOME_LOG,
    min_samples: int = 30,
) -> dict[str, Any]:
    """Match logged predictions against reported ground truth outcomes.

    This stands in for "Model Performance" in the monitoring diagram: unlike
    data drift (available immediately from the inputs), whether a
    prediction was actually right is only known once the applicant's real
    repayment behaviour is observed and reported via
    `POST /api/v1/outcomes`.
    """
    predictions_by_id = {
        event["applicant_id"]: event
        for event in _read_jsonl(predictions_path)
        if event.get("applicant_id")
    }

    matched_probability: list[float] = []
    matched_actual: list[int] = []
    for outcome in _read_jsonl(outcomes_path):
        prediction = predictions_by_id.get(outcome.get("applicant_id"))
        if prediction is None:
            continue
        matched_probability.append(float(prediction["probability"]))
        matched_actual.append(int(bool(outcome["actual_default"])))

    n_matched = len(matched_actual)
    if n_matched < min_samples or len(set(matched_actual)) < 2:
        return {
            "status": "insufficient_data",
            "n_matched": n_matched,
            "min_samples": min_samples,
            "live_roc_auc": None,
        }

    from sklearn.metrics import roc_auc_score

    live_roc_auc = float(roc_auc_score(matched_actual, matched_probability))
    return {
        "status": "ok",
        "n_matched": n_matched,
        "min_samples": min_samples,
        "live_roc_auc": live_roc_auc,
    }


def performance_degraded_trigger(
    champion_roc_auc: float,
    live_roc_auc: float,
    max_drop: float = 0.05,
) -> bool:
    """Whether live ROC-AUC has degraded enough below the champion's
    validation ROC-AUC to justify a retraining run on its own."""
    return (champion_roc_auc - live_roc_auc) > max_drop
