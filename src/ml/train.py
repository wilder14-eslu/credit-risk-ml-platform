"""Model training entry point.

Trains the XGBoost default-prediction model and, when MLflow is available,
logs parameters/metrics/artifacts and registers the model (MLOps principles
P3 "Reproducibility", P4 "Versioning" and P7 "ML metadata tracking" from
Kreuzberger et al., 2022 / arXiv:2205.02302). MLflow is best-effort: if it
is not installed or no tracking server is reachable, training still
succeeds and the model is still saved locally, so the core pipeline never
hard-depends on the observability stack.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.data_pipeline.preprocess import prepare_training_data

logger = logging.getLogger(__name__)

MODEL_NAME = "bcp_credit_xgboost"
DEFAULT_MODEL_PATH = "data/processed/model.joblib"
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "auc",
    "random_state": 42,
}


def _log_to_mlflow(
    params: dict[str, Any],
    metrics: dict[str, float],
    model: XGBClassifier,
    local_model_path: Path,
    register: bool,
) -> str | None:
    """Best-effort MLflow logging. Returns the run id, or None if MLflow is
    unavailable / logging failed for any reason (e.g. no tracking server)."""
    try:
        import mlflow
        import mlflow.xgboost
    except ImportError:
        logger.warning("mlflow no está instalado; se omite el registro de experimentos.")
        return None

    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "credit-risk-default"))
        with mlflow.start_run(run_name="xgboost-default-prediction") as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=MODEL_NAME if register else None,
            )
            mlflow.log_artifact(str(local_model_path))
            return run.info.run_id
    except Exception:  # pragma: no cover - depends on external tracking server
        logger.exception("No se pudo registrar la corrida en MLflow; se continúa sin ella.")
        return None


def train_model(
    raw_data: pd.DataFrame,
    params: dict[str, Any] | None = None,
    model_path: Path | str | None = None,
    register: bool = True,
) -> dict[str, float | str | None]:
    """Train the default-prediction model and return its evaluation metrics.

    ``model_path`` defaults to the ``MODEL_PATH`` environment variable so the
    API/Streamlit demo and this function always agree on where the artifact
    lives. Passing it explicitly (as the tests do) keeps runs isolated.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    resolved_model_path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

    x_train, x_test, y_train, y_test = prepare_training_data(raw_data)

    model = XGBClassifier(**params)
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    roc_auc = float(roc_auc_score(y_test, probabilities))
    metrics = {
        "roc_auc": roc_auc,
        "train_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
    }

    resolved_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, resolved_model_path)

    run_id = _log_to_mlflow(params, metrics, model, resolved_model_path, register)

    return {
        "run_id": run_id,
        "roc_auc": roc_auc,
        "model_path": str(resolved_model_path),
    }


if __name__ == "__main__":
    import sys

    from src.data_pipeline.ingest import download_give_me_some_credit
    from src.data_pipeline.validate import clean_out_of_range_rows, validate_input_data

    logging.basicConfig(level=logging.INFO)

    data_dir = download_give_me_some_credit()
    csv_files = sorted(Path(data_dir).glob("cs-training.csv")) or sorted(
        Path(data_dir).glob("*.csv")
    )
    if not csv_files:
        sys.exit(f"No se encontraron archivos CSV en {data_dir}")

    data = pd.read_csv(csv_files[0], index_col=0)
    data = clean_out_of_range_rows(data)
    validate_input_data(data, require_target=True)
    result = train_model(data)
    print(
        f"Modelo entrenado -> roc_auc={result['roc_auc']:.4f} "
        f"model_path={result['model_path']} run_id={result['run_id']}"
    )
