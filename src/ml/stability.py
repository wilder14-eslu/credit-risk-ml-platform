"""Estabilidad de las métricas: semillas, folds, remuestras y segmentos.

Reporta media, desviación estándar, mediana, rango intercuartílico (IQR) e
IC 95 % para cada métrica. Un modelo cuyo AUC cambia mucho con la semilla
del split no es confiable aunque su promedio sea alto.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.ml.evaluation import _fold_metrics
from src.ml.train import DEFAULT_PARAMS, _build_model

logger = logging.getLogger(__name__)

METRICS = ("roc_auc", "pr_auc", "ks", "gini", "brier", "log_loss", "ece")


def summarize_distribution(values: Sequence[float], alpha: float = 0.05) -> dict[str, float]:
    """mean, std, median, IQR, IC t 95 % de la media e IC percentil 95 % de la muestra."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        raise ValueError("Sin valores que resumir.")
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(1 - alpha / 2, n - 1) * std / np.sqrt(n)) if n > 1 else 0.0
    q1, med, q3 = (float(q) for q in np.quantile(v, [0.25, 0.5, 0.75]))
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "median": med,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "ci_mean_low": mean - half,
        "ci_mean_high": mean + half,
        "min": float(v.min()),
        "max": float(v.max()),
    }


def seed_stability(
    x: pd.DataFrame,
    y: pd.Series,
    algorithms: Sequence[str],
    seeds: Sequence[int] = tuple(range(10)),
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Repite split estratificado + entrenamiento con distintas semillas.

    Cada semilla cambia a la vez el split y la semilla del modelo, así que
    mide la sensibilidad total al azar del procedimiento.
    """
    per_seed: dict[str, list[dict[str, float]]] = {a: [] for a in algorithms}
    for seed in seeds:
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=test_size, stratify=y, random_state=seed
        )
        for algorithm in algorithms:
            params = dict(DEFAULT_PARAMS[algorithm])
            for key in ("random_state", "seed"):
                if key in params:
                    params[key] = seed
            model = Pipeline(
                [("imputer", SimpleImputer(strategy="median")),
                 ("model", _build_model(algorithm, params))]
            ).set_output(transform="pandas")
            model.fit(x_tr, y_tr)
            p = model.predict_proba(x_te)[:, 1]
            per_seed[algorithm].append({"seed": seed, **_fold_metrics(y_te.to_numpy(), p)})
            logger.info("seed=%s %s AUC=%.4f", seed, algorithm,
                        per_seed[algorithm][-1]["roc_auc"])
    summary = {
        algorithm: {m: summarize_distribution([r[m] for r in rows]) for m in METRICS}
        for algorithm, rows in per_seed.items()
    }
    return {"seeds": list(seeds), "per_seed": per_seed, "summary": summary}


def summarize_cv_folds(per_fold: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    """Resumen de estabilidad entre folds (misma estructura que por semilla)."""
    return {
        algorithm: {m: summarize_distribution([r[m] for r in rows]) for m in METRICS}
        for algorithm, rows in per_fold.items()
    }
