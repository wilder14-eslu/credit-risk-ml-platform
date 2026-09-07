"""Automated drift/performance monitoring that closes the platform's
Alert -> Retraining loop.

Runs on a schedule (see the ``if __name__ == "__main__"`` block below and
the ``scheduler`` service in `docker/docker-compose.yml`). Each run inspects
recent production traffic and, with the pure `decide_retrain` rule, decides
whether any of three independent signals justifies kicking off
`src.orchestrator.pipeline.retraining_pipeline`:

- rate drift (`src.api.monitoring.drift_trigger`): the observed default
  rate among recent predictions looks anomalously high.
- feature drift (`src.monitoring.drift.compute_feature_drift`): the
  incoming feature distributions have moved away from what the champion
  model was trained on (PSI).
- performance degradation
  (`src.api.monitoring.performance_degraded_trigger`): once enough ground
  truth has been reported via `POST /api/v1/outcomes`, the live ROC-AUC has
  dropped meaningfully below the champion's validation ROC-AUC.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from prefect import flow, task

from src.api.monitoring import (
    compute_live_performance,
    drift_trigger,
    load_recent_predictions,
    performance_degraded_trigger,
)
from src.monitoring.drift import compute_feature_drift, load_reference_distribution
from src.orchestrator.pipeline import retraining_pipeline

logger = logging.getLogger(__name__)

MODEL_NAME = "bcp_credit_champion"


def decide_retrain(
    rate_triggered: bool,
    feature_drift_triggered: bool,
    performance_triggered: bool,
) -> bool:
    """Pure decision rule, kept apart from I/O so it is trivial to test."""
    return rate_triggered or feature_drift_triggered or performance_triggered


def _champion_roc_auc(model_name: str = MODEL_NAME) -> float | None:
    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return None
        return float(client.get_run(versions[0].run_id).data.metrics.get("roc_auc", 0.0))
    except Exception:  # pragma: no cover - depends on a reachable MLflow server
        logger.exception("No se pudo leer el ROC-AUC del champion desde MLflow.")
        return None


@task
def check_rate_drift() -> tuple[bool, int]:
    predictions = load_recent_predictions()
    probabilities = [event["probability"] for event in predictions]
    return drift_trigger(probabilities), len(predictions)


@task
def check_feature_drift(psi_threshold: float) -> dict:
    predictions = load_recent_predictions()
    empty_result = {
        "psi": {},
        "drifted_features": [],
        "drift_detected": False,
        "threshold": psi_threshold,
    }
    if not predictions:
        return empty_result
    try:
        reference = load_reference_distribution()
    except FileNotFoundError:
        logger.warning("No hay distribución de referencia; entrena un modelo primero.")
        return empty_result
    recent_features = pd.DataFrame([event["features"] for event in predictions])
    return compute_feature_drift(recent_features, reference, psi_threshold=psi_threshold)


@task
def check_performance(max_auc_drop: float) -> tuple[bool, dict]:
    performance = compute_live_performance()
    if performance["status"] != "ok":
        return False, performance
    champion_auc = _champion_roc_auc()
    if champion_auc is None:
        return False, performance
    triggered = performance_degraded_trigger(
        champion_auc, performance["live_roc_auc"], max_auc_drop
    )
    return triggered, {**performance, "champion_roc_auc": champion_auc}


@flow(name="credit-risk-monitoring")
def monitoring_flow() -> dict:
    psi_threshold = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))
    max_auc_drop = float(os.getenv("PERFORMANCE_AUC_DROP_THRESHOLD", "0.05"))

    rate_triggered, n_predictions = check_rate_drift()
    feature_drift = check_feature_drift(psi_threshold)
    performance_triggered, performance = check_performance(max_auc_drop)

    should_retrain = decide_retrain(
        rate_triggered, feature_drift["drift_detected"], performance_triggered
    )

    result = {
        "n_predictions": n_predictions,
        "rate_drift_triggered": rate_triggered,
        "feature_drift": feature_drift,
        "performance": performance,
        "performance_degraded": performance_triggered,
        "retrain_triggered": should_retrain,
    }

    if should_retrain:
        logger.warning("Monitoreo detectó una señal de alerta; disparando reentrenamiento.")
        result["retraining_result"] = retraining_pipeline()

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cron = os.getenv("MONITORING_CRON", "0 * * * *")
    monitoring_flow.serve(name="credit-risk-monitoring", cron=cron)
