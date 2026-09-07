"""Feature-level data drift detection via Population Stability Index (PSI).

PSI compares the distribution of a feature at training time (the
"reference") against a recent window of production traffic. Both are
bucketed into the same quantile bins and PSI measures how much probability
mass moved between buckets:

    PSI = sum((actual_i - expected_i) * ln(actual_i / expected_i))

As a rule of thumb widely used in credit scoring (e.g. Basel model
validation practice): PSI < 0.1 -> no significant shift, 0.1-0.2 ->
moderate shift worth watching, > 0.2 -> significant shift that should
block trusting the model on that traffic as-is.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BUCKETS = 10
DEFAULT_PSI_THRESHOLD = 0.2
DEFAULT_REFERENCE_PATH = Path("data/processed/reference_distribution.json")
_EPSILON = 1e-4


def _bucket_edges(values: pd.Series, buckets: int) -> list[float]:
    quantiles = np.linspace(0.0, 1.0, buckets + 1)
    edges = sorted(set(values.quantile(quantiles).tolist()))
    if len(edges) < 2:
        # Constant (or near-constant) feature: a single bucket still lets
        # PSI be computed, and it will correctly stay near 0 unless the
        # whole distribution later moves away from that constant.
        edges = [float(values.min()) - 1.0, float(values.max()) + 1.0]
    edges[0] = -math.inf
    edges[-1] = math.inf
    return edges


def _bucket_proportions(values: pd.Series, edges: list[float]) -> list[float]:
    values = values.dropna()
    if values.empty:
        return [0.0] * (len(edges) - 1)
    counts = pd.cut(values, bins=edges, include_lowest=True).value_counts(sort=False)
    total = float(counts.sum()) or 1.0
    return [count / total for count in counts.tolist()]


def save_reference_distribution(
    training_features: pd.DataFrame,
    output_path: Path | str = DEFAULT_REFERENCE_PATH,
    buckets: int = DEFAULT_BUCKETS,
) -> Path:
    """Snapshot the training feature distributions used as the drift baseline.

    Called once per training run (see `src.ml.train.train_model`) so every
    promoted model carries the distribution it was trained on, independent
    of whatever data happens to be flowing through production later.
    """
    reference = {}
    for column in training_features.columns:
        edges = _bucket_edges(training_features[column], buckets)
        reference[column] = {
            "edges": edges,
            "proportions": _bucket_proportions(training_features[column], edges),
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reference), encoding="utf-8")
    return output_path


def load_reference_distribution(path: Path | str = DEFAULT_REFERENCE_PATH) -> dict:
    with Path(path).open(encoding="utf-8") as reference_file:
        return json.load(reference_file)


def population_stability_index(expected: list[float], actual: list[float]) -> float:
    """PSI between two bucketed distributions of the same feature."""
    psi = 0.0
    for expected_share, actual_share in zip(expected, actual):
        expected_share = max(expected_share, _EPSILON)
        actual_share = max(actual_share, _EPSILON)
        psi += (actual_share - expected_share) * math.log(actual_share / expected_share)
    return psi


def compute_feature_drift(
    recent_features: pd.DataFrame,
    reference: dict,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
) -> dict:
    """Compare recent production traffic against the training reference.

    Returns per-feature PSI plus an overall ``drift_detected`` flag (any
    feature above ``psi_threshold``). This is the "Data Drift" box in the
    platform's monitoring diagram; `src.orchestrator.monitor.monitoring_flow`
    uses ``drift_detected`` to decide whether to trigger a retraining run.
    """
    if recent_features.empty:
        return {
            "psi": {},
            "drifted_features": [],
            "drift_detected": False,
            "threshold": psi_threshold,
        }

    psi_by_feature: dict[str, float] = {}
    for feature, distribution in reference.items():
        if feature not in recent_features.columns:
            continue
        actual_props = _bucket_proportions(recent_features[feature], distribution["edges"])
        psi_by_feature[feature] = round(
            population_stability_index(distribution["proportions"], actual_props), 4
        )

    drifted = [feature for feature, psi in psi_by_feature.items() if psi > psi_threshold]
    return {
        "psi": psi_by_feature,
        "drifted_features": drifted,
        "drift_detected": bool(drifted),
        "threshold": psi_threshold,
    }
