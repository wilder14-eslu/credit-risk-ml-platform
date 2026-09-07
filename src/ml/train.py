"""Model training entry point.

Trains the default-prediction model for one algorithm and, when MLflow is
available, logs parameters/metrics/artifacts and registers the model (MLOps
principles P3 "Reproducibility", P4 "Versioning" and P7 "ML metadata
tracking" from Kreuzberger et al., 2022 / arXiv:2205.02302). MLflow is
best-effort: if it is not installed or no tracking server is reachable,
training still succeeds and the model is still saved locally, so the core
pipeline never hard-depends on the observability stack.

Supports four algorithms (see `ALGORITHMS`): a Logistic Regression baseline
and three gradient-boosting libraries. `src.ml.benchmark` calls this
function once per algorithm on the same train/test split to run the
"MODEL BENCHMARK -> BEST MODEL" comparison described in the project's
architecture; this module by itself always trains exactly one algorithm
(XGBoost by default, matching `src.orchestrator.pipeline`'s previous
behaviour) if you just need a single model.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.data_pipeline.preprocess import prepare_training_data
from src.ml.metrics import compute_classification_metrics, diagnose_fit
from src.monitoring.drift import save_reference_distribution

logger = logging.getLogger(__name__)

MODEL_NAME = "bcp_credit_champion"
DEFAULT_MODEL_PATH = "data/processed/model.joblib"
ALGORITHMS = ("logistic_regression", "xgboost", "lightgbm", "catboost")

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"max_iter": 1000, "random_state": 42},
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "auc",
        "random_state": 42,
    },
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": -1,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
    },
    "catboost": {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "random_state": 42,
    },
}


def _build_model(algorithm: str, params: dict[str, Any]) -> Any:
    if algorithm == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Unlike tree models, linear models need scaled inputs -- the raw
        # features here have wildly different ranges (age in tens,
        # debt_ratio sometimes in the thousands).
        return Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(**params))]
        )

    if algorithm == "xgboost":
        return XGBClassifier(**params)

    if algorithm == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise ImportError(
                "lightgbm no está instalado; instálalo con `pip install lightgbm` "
                "para entrenar con algorithm='lightgbm'."
            ) from error
        return LGBMClassifier(**params)

    if algorithm == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as error:
            raise ImportError(
                "catboost no está instalado; instálalo con `pip install catboost` "
                "para entrenar con algorithm='catboost'."
            ) from error
        return CatBoostClassifier(verbose=False, allow_writing_files=False, **params)

    raise ValueError(f"Algoritmo no soportado: {algorithm!r}. Usa uno de {ALGORITHMS}.")


def _log_to_mlflow(
    algorithm: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    model: Any,
    local_model_path: Path,
    register: bool,
) -> str | None:
    """Best-effort MLflow logging. Returns the run id, or None if MLflow is
    unavailable / logging failed for any reason (e.g. no tracking server)."""
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow no está instalado; se omite el registro de experimentos.")
        return None

    flavor_by_algorithm = {
        "xgboost": "mlflow.xgboost",
        "lightgbm": "mlflow.lightgbm",
        "catboost": "mlflow.catboost",
    }

    try:
        import importlib

        flavor = importlib.import_module(flavor_by_algorithm.get(algorithm, "mlflow.sklearn"))

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "credit-risk-default"))
        with mlflow.start_run(run_name=f"{algorithm}-default-prediction") as run:
            mlflow.log_params({**params, "algorithm": algorithm})
            mlflow.log_metrics(metrics)
            flavor.log_model(
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
    algorithm: str = "xgboost",
) -> dict[str, Any]:
    """Train one candidate model and return its evaluation metrics.

    ``model_path`` defaults to the ``MODEL_PATH`` environment variable so the
    API/Streamlit demo and this function always agree on where the artifact
    lives. Passing it explicitly (as the tests, and `src.ml.benchmark`, do)
    keeps runs isolated. ``algorithm`` picks one of `ALGORITHMS`; the
    returned metrics always include both the training and the test split
    (see `src.ml.metrics`) so callers can tell a well-fit model from an
    over/underfit one, not just read a single accuracy number.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algoritmo no soportado: {algorithm!r}. Usa uno de {ALGORITHMS}.")

    resolved_params = {**DEFAULT_PARAMS[algorithm], **(params or {})}
    resolved_model_path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    threshold = float(os.getenv("DECISION_THRESHOLD", "0.5"))

    x_train, x_test, y_train, y_test = prepare_training_data(raw_data)

    reference_path = save_reference_distribution(
        x_train, resolved_model_path.parent / "reference_distribution.json"
    )

    model = _build_model(algorithm, resolved_params)
    model.fit(x_train, y_train)

    train_probabilities = model.predict_proba(x_train)[:, 1]
    train_metrics = compute_classification_metrics(y_train, train_probabilities, threshold)

    start = time.perf_counter()
    test_probabilities = model.predict_proba(x_test)[:, 1]
    elapsed_seconds = time.perf_counter() - start
    latency_ms_per_row = (elapsed_seconds / max(len(x_test), 1)) * 1000

    test_metrics = compute_classification_metrics(y_test, test_probabilities, threshold)
    diagnosis = diagnose_fit(train_metrics, test_metrics)

    mlflow_metrics = {
        # Backward-compatible top-level key: `evaluate_and_promote` and
        # `src.orchestrator.monitor` read the champion's ROC-AUC from here.
        "roc_auc": test_metrics["roc_auc"],
        "train_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
        "overfit_gap": diagnosis["gap"],
        "latency_ms_per_row": latency_ms_per_row,
        **{f"train_{name}": value for name, value in train_metrics.items()},
        **{f"test_{name}": value for name, value in test_metrics.items()},
    }

    resolved_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, resolved_model_path)

    run_id = _log_to_mlflow(
        algorithm, resolved_params, mlflow_metrics, model, resolved_model_path, register
    )

    return {
        "run_id": run_id,
        "algorithm": algorithm,
        "roc_auc": test_metrics["roc_auc"],
        "model_path": str(resolved_model_path),
        "reference_distribution_path": str(reference_path),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "diagnosis": diagnosis,
        "latency_ms_per_row": latency_ms_per_row,
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
        f"Modelo entrenado ({result['algorithm']}) -> roc_auc={result['roc_auc']:.4f} "
        f"({result['diagnosis']['label']}) model_path={result['model_path']} "
        f"run_id={result['run_id']}"
    )

    try:
        from src.feature_store.features import FEATURE_NAMES
        from src.ml.report import (
            build_report_sections,
            render_markdown,
            render_pdf,
            top_feature_importances,
        )

        trained_model = joblib.load(result["model_path"])
        top_features = top_feature_importances(trained_model, list(FEATURE_NAMES))
        sections = build_report_sections(result, top_features=top_features)

        from datetime import datetime, timezone

        reports_dir = Path("data/processed/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        md_path = reports_dir / f"training_report_{stamp}.md"
        md_path.write_text(render_markdown(sections), encoding="utf-8")
        print(f"Informe (Markdown) -> {md_path}")

        pdf_path = reports_dir / f"training_report_{stamp}.pdf"
        render_pdf(sections, pdf_path)
        print(f"Informe (PDF) -> {pdf_path}")
    except Exception:  # pragma: no cover - report generation must never fail training
        logging.getLogger(__name__).exception("No se pudo generar el informe de entrenamiento.")
