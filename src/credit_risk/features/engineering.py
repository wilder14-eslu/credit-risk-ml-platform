"""Transformaciones de features compartidas por entrenamiento, batch e inferencia online.

La misma función `build_features` se ejecuta al construir la feature table en
Unity Catalog y dentro del modelo pyfunc que sirve el endpoint de Databricks,
así se elimina el training/serving skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import base_feature_names

DERIVED_FEATURES: tuple[str, ...] = (
    "total_past_due",
    "disposable_income",
    "income_per_dependent",
    "income_missing",
    "utilization_capped",
)


DERIVED_LABELS: dict[str, str] = {
    "total_past_due": "Total de atrasos históricos",
    "disposable_income": "Ingreso disponible estimado",
    "income_per_dependent": "Ingreso por dependiente",
    "income_missing": "Ingreso no declarado",
    "utilization_capped": "Utilización de crédito (acotada)",
}


def feature_names() -> tuple[str, ...]:
    return base_feature_names() + DERIVED_FEATURES


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Devuelve la matriz de features canónica (base + derivadas) SIN imputar."""
    missing = [c for c in base_feature_names() if c not in data.columns]
    if missing:
        raise ValueError(f"Faltan features base: {missing}")
    df = data.loc[:, list(base_feature_names())].apply(pd.to_numeric, errors="coerce")
    df = df.astype("float64")

    df["total_past_due"] = (
        df["number_of_time_30_59_days_past_due"].fillna(0)
        + df["number_of_time_60_89_days_past_due"].fillna(0)
        + df["number_of_times_90_days_late"].fillna(0)
    )
    df["disposable_income"] = df["monthly_income"] * (1.0 - df["debt_ratio"].clip(upper=1.0))
    df["income_per_dependent"] = df["monthly_income"] / (df["number_dependents"].fillna(0) + 1.0)
    df["income_missing"] = df["monthly_income"].isna().astype("float64")
    df["utilization_capped"] = df["revolving_utilization_unsecured"].clip(upper=1.5)
    return df[list(feature_names())]


class MedianImputer:
    """Imputador por mediana ajustado en train y serializado junto al modelo."""

    def __init__(self) -> None:
        self.medians_: dict[str, float] = {}

    def fit(self, features: pd.DataFrame) -> MedianImputer:
        self.medians_ = {c: float(np.nan_to_num(features[c].median(), nan=0.0)) for c in features.columns}
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.medians_:
            raise RuntimeError("MedianImputer no está ajustado")
        return features.fillna(value=self.medians_)

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)
