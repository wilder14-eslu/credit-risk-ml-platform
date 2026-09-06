"""Model inference entry point.

Loads the trained model (local joblib artifact by default, or an MLflow
Model Registry URI when `MLFLOW_MODEL_URI` is set, e.g.
``models:/bcp_credit_xgboost/Production``) and scores a single applicant
using the same canonical feature schema used at training time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.feature_store.features import FEATURE_NAMES, build_features

DEFAULT_MODEL_PATH = "data/processed/model.joblib"


class ModelNotAvailableError(RuntimeError):
    """Raised when no trained model artifact can be found."""


def _resolve_model_path(model_path: Path | str | None) -> Path:
    return Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))


def load_model(model_path: Path | str | None = None) -> Any:
    """Load the trained model, preferring the MLflow registry when configured."""
    registry_uri = os.getenv("MLFLOW_MODEL_URI")
    if registry_uri:
        import mlflow.pyfunc

        return mlflow.pyfunc.load_model(registry_uri)

    resolved_path = _resolve_model_path(model_path)
    if not resolved_path.is_file():
        raise ModelNotAvailableError(
            f"No se encontró un modelo entrenado en {resolved_path}. "
            "Ejecuta `python -m src.ml.train` (o `make train`) primero."
        )
    return joblib.load(resolved_path)


def _predict_proba(model: Any, features: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features)[:, 1][0])
    prediction = model.predict(features)  # mlflow.pyfunc models expose predict() only
    value = prediction.iloc[0] if hasattr(prediction, "iloc") else prediction[0]
    return float(value)


def risk_band(probability: float) -> str:
    """Coarse, business-facing risk bucket for a default probability."""
    if probability < 0.10:
        return "bajo"
    if probability < 0.30:
        return "medio"
    return "alto"


def _build_applicant_features(applicant: dict[str, Any]) -> pd.DataFrame:
    raw = pd.DataFrame([applicant])
    features = build_features(raw)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    return features[list(FEATURE_NAMES)]


def predict_default_probability(
    applicant: dict[str, Any],
    threshold: float | None = None,
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    """Score a single applicant using the canonical feature schema.

    ``applicant`` accepts either the raw Kaggle column names (e.g.
    ``RevolvingUtilizationOfUnsecuredLines``) or the canonical ones (e.g.
    ``revolving_utilization_unsecured``), whatever `build_features` finds.
    """
    resolved_threshold = (
        threshold if threshold is not None else float(os.getenv("DECISION_THRESHOLD", "0.5"))
    )
    features = _build_applicant_features(applicant)

    model = load_model(model_path)
    probability = _predict_proba(model, features)
    decision = "rechazar" if probability >= resolved_threshold else "aprobar"

    return {
        "probability": probability,
        "decision": decision,
        "risk_band": risk_band(probability),
        "features": features.iloc[0].to_dict(),
    }


def predict_with_explanation(
    applicant: dict[str, Any],
    top_n: int = 5,
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    """`predict_default_probability` plus a SHAP-based explanation.

    Explanation is best-effort: if SHAP is not installed, or the loaded
    model does not support SHAP's TreeExplainer (e.g. a generic
    ``mlflow.pyfunc`` wrapper), ``top_factors`` is simply empty rather than
    failing the whole prediction.
    """
    result = predict_default_probability(applicant, model_path=model_path)
    try:
        from src.ml.explainability import explain_features

        model = load_model(model_path)
        features_row = pd.DataFrame([result["features"]])[list(FEATURE_NAMES)]
        result["top_factors"] = explain_features(model, features_row, top_n=top_n)
    except Exception:  # noqa: BLE001 - explanation must never break a prediction
        result["top_factors"] = []
    return result
