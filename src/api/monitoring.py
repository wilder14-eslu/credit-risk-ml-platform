"""Prediction monitoring events for audit and drift detection."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PREDICTION_LOG = Path("data/processed/predictions.jsonl")


def record_prediction(
    features: dict[str, Any],
    probability: float,
    decision: str,
    output_path: Path = DEFAULT_PREDICTION_LOG,
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - keep py3.10 compatible
        "features": features,
        "probability": probability,
        "decision": decision,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as event_file:
        event_file.write(json.dumps(event) + "\n")


def drift_trigger(
    probabilities: list[float],
    threshold: float = 0.5,
) -> bool:
    """Return whether the observed default rate exceeds the retraining threshold."""
    if not probabilities:
        return False
    return sum(probability >= threshold for probability in probabilities) / len(
        probabilities
    ) > threshold
