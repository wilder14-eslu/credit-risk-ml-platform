"""Algoritmos candidatos del benchmark y sus espacios de búsqueda (Optuna)."""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def is_available(name: str) -> bool:
    module = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost"}.get(name)
    return module is None or importlib.util.find_spec(module) is not None


def default_params(name: str) -> dict[str, Any]:
    return {
        "logistic_regression": {"C": 1.0},
        "xgboost": {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
        },
        "lightgbm": {
            "n_estimators": 400,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 50,
        },
        "catboost": {"iterations": 500, "depth": 6, "learning_rate": 0.05},
    }[name]


def suggest_params(trial, name: str) -> dict[str, Any]:
    """Espacio de búsqueda de hiperparámetros por algoritmo."""
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        }
    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 300, 900, step=100),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    raise ValueError(f"Algoritmo desconocido: {name}")


KNOWN = ("logistic_regression", "xgboost", "lightgbm", "catboost")


def build_estimator(name: str, params: dict[str, Any] | None = None, seed: int = 42):
    if name not in KNOWN:
        raise ValueError(f"Algoritmo desconocido: {name}")
    params = {**default_params(name), **(params or {})}
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, **params)),
            ]
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **params,
            eval_metric="auc",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params, subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            **params, random_seed=seed, verbose=False, eval_metric="AUC", allow_writing_files=False
        )
    raise ValueError(f"Algoritmo desconocido: {name}")


def available_candidates(names: list[str]) -> list[str]:
    usable = []
    for name in names:
        if is_available(name):
            usable.append(name)
        else:
            logger.warning("Candidato %s omitido: dependencia no instalada", name)
    return usable
