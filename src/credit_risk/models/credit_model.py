"""Modelo servible: features + imputación + estimador + umbral + explicaciones.

`CreditRiskModel` es un objeto Python puro (se prueba sin MLflow). Se registra
en Unity Catalog envuelto en `CreditRiskPyfunc`, que es lo que ejecuta el
endpoint de Model Serving de Databricks: recibe las 10 variables canónicas
crudas y devuelve probabilidad, decisión, banda de riesgo y los factores que
más empujaron el score (contribuciones SHAP nativas del algoritmo).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.config import feature_definitions
from credit_risk.features.engineering import DERIVED_LABELS, MedianImputer, build_features, feature_names


def risk_band(probability: float) -> str:
    if probability < 0.05:
        return "A"
    if probability < 0.10:
        return "B"
    if probability < 0.20:
        return "C"
    if probability < 0.35:
        return "D"
    return "E"


@dataclass
class CreditRiskModel:
    algorithm: str
    estimator: Any
    imputer: MedianImputer
    threshold: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        return self.imputer.transform(build_features(raw))

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.transform(raw))[:, 1]

    def contributions(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Contribución de cada feature al log-odds (SHAP nativo o coeficiente*x)."""
        x = self.transform(raw)
        names = list(feature_names())
        if self.algorithm == "xgboost":
            import xgboost as xgb

            values = self.estimator.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)
            values = values[:, :-1]
        elif self.algorithm == "lightgbm":
            values = self.estimator.predict(x, pred_contrib=True)[:, :-1]
        elif self.algorithm == "catboost":
            from catboost import Pool

            values = self.estimator.get_feature_importance(Pool(x), type="ShapValues")[:, :-1]
        else:
            scaler = self.estimator.named_steps["scaler"]
            coef = self.estimator.named_steps["clf"].coef_[0]
            values = scaler.transform(x) * coef
        return pd.DataFrame(values, columns=names, index=raw.index)

    def predict_frame(self, raw: pd.DataFrame, explain: bool = True, top_k: int = 3) -> pd.DataFrame:
        proba = self.predict_proba(raw)
        out = pd.DataFrame(
            {
                "probability": proba,
                "decision": np.where(proba >= self.threshold, "RECHAZAR", "APROBAR"),
                "risk_band": [risk_band(p) for p in proba],
            },
            index=raw.index,
        )
        if explain:
            contrib = self.contributions(raw)
            labels = {n: d.get("label", n) for n, d in feature_definitions().items()} | DERIVED_LABELS
            factors = []
            for _, row in contrib.iterrows():
                top = row.reindex(row.abs().sort_values(ascending=False).index)[:top_k]
                factors.append(
                    json.dumps(
                        [
                            {
                                "feature": k,
                                "label": labels.get(k, k),
                                "impact": round(float(v), 4),
                                "direction": "aumenta_riesgo" if v > 0 else "reduce_riesgo",
                            }
                            for k, v in top.items()
                        ],
                        ensure_ascii=False,
                    )
                )
            out["top_factors"] = factors
        return out

    def global_importance(self, sample: pd.DataFrame) -> pd.Series:
        return self.contributions(sample).abs().mean().sort_values(ascending=False)


try:  # MLflow es opcional para tests unitarios del gateway
    import mlflow.pyfunc

    class CreditRiskPyfunc(mlflow.pyfunc.PythonModel):
        """Wrapper pyfunc que corre en Databricks Model Serving."""

        def load_context(self, context) -> None:
            import joblib

            self.model: CreditRiskModel = joblib.load(context.artifacts["credit_model"])

        def predict(self, context, model_input: pd.DataFrame, params: dict | None = None):
            params = params or {}
            explain = bool(params.get("explain", True))
            frame = self.model.predict_frame(model_input, explain=explain)
            if not explain:
                frame["top_factors"] = "[]"
            return frame.reset_index(drop=True)

except ImportError:  # pragma: no cover
    CreditRiskPyfunc = None  # type: ignore[assignment,misc]
