"""Validación cruzada anidada con optimización de hiperparámetros (Optuna).

* **CV interna** (``inner_splits``): elige hiperparámetros con ROC-AUC medio,
  solo con los datos de entrenamiento del fold externo.
* **CV externa** (``outer_splits``): estima la generalización del
  *procedimiento completo* (incluido el tuning), sin sesgo optimista de
  selección.
* Se compara, sobre los **mismos folds externos**, contra los hiperparámetros
  por defecto para cuantificar si el tuning realmente ayuda.
* El **holdout** permanece aislado: los hiperparámetros finales se eligen con
  el desarrollo y el holdout se usa una sola vez.

Ejecutar: ``python -m src.ml.nested_cv --n-trials 12``
(los resultados parciales se guardan y se reanudan por algoritmo).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data_pipeline.preprocess import split_raw_features
from src.ml import stats as st
from src.ml.evaluation import _fold_metrics
from src.ml.stability import summarize_distribution
from src.ml.train import DEFAULT_PARAMS, _build_model

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEARCH_ALGORITHMS = ("logistic_regression", "xgboost", "lightgbm", "catboost")


def suggest_params(algorithm: str, trial: optuna.Trial) -> dict[str, Any]:
    """Espacio de búsqueda por algoritmo (sobre los parámetros por defecto)."""
    base = dict(DEFAULT_PARAMS[algorithm])
    if algorithm == "logistic_regression":
        base["C"] = trial.suggest_float("C", 1e-3, 10.0, log=True)
    elif algorithm == "xgboost":
        base.update(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 7),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
        )
    elif algorithm == "lightgbm":
        base.update(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            num_leaves=trial.suggest_int("num_leaves", 8, 64),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
        )
    elif algorithm == "catboost":
        base.update(
            iterations=trial.suggest_int("iterations", 100, 500, step=50),
            depth=trial.suggest_int("depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        )
    else:
        raise ValueError(f"Algoritmo sin espacio de búsqueda: {algorithm}")
    return base


def make_pipeline(algorithm: str, params: dict[str, Any]) -> Pipeline:
    """Imputación dentro del estimador: se reajusta en cada fold (sin fuga)."""
    return Pipeline(
        [("imputer", SimpleImputer(strategy="median")),
         ("model", _build_model(algorithm, dict(params)))]
    ).set_output(transform="pandas")


def tune_hyperparameters(
    algorithm: str,
    x: pd.DataFrame,
    y: np.ndarray,
    n_trials: int = 12,
    inner_splits: int = 3,
    seed: int = 42,
) -> tuple[dict[str, Any], float]:
    """Optuna (TPE) maximizando el ROC-AUC medio de una CV estratificada interna."""
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    folds = list(inner.split(x, y))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(algorithm, trial)
        scores = []
        for tr, va in folds:
            model = make_pipeline(algorithm, params)
            model.fit(x.iloc[tr], y[tr])
            scores.append(roc_auc_score(y[va], model.predict_proba(x.iloc[va])[:, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    best = dict(DEFAULT_PARAMS[algorithm])
    best.update(study.best_params)
    return best, float(study.best_value)


def nested_cross_validation(
    algorithm: str,
    x_dev: pd.DataFrame,
    y_dev: pd.Series,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_trials: int = 12,
    seed: int = 42,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """CV anidada de un algoritmo + comparación pareada con sus valores por defecto."""
    y = y_dev.to_numpy()
    outer = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=1, random_state=seed)
    folds = []
    for fold_id, (tr, va) in enumerate(outer.split(x_dev, y)):
        start = time.perf_counter()
        best_params, inner_score = tune_hyperparameters(
            algorithm, x_dev.iloc[tr], y[tr], n_trials, inner_splits, seed + fold_id
        )
        tuned = clone(make_pipeline(algorithm, best_params))
        tuned.fit(x_dev.iloc[tr], y[tr])
        tuned_metrics = _fold_metrics(y[va], tuned.predict_proba(x_dev.iloc[va])[:, 1])

        default = clone(make_pipeline(algorithm, dict(DEFAULT_PARAMS[algorithm])))
        default.fit(x_dev.iloc[tr], y[tr])
        default_metrics = _fold_metrics(y[va], default.predict_proba(x_dev.iloc[va])[:, 1])

        folds.append(
            {
                "fold": fold_id,
                "inner_cv_best_auc": inner_score,
                "outer_tuned": tuned_metrics,
                "outer_default": default_metrics,
                "best_params": best_params,
                "seconds": time.perf_counter() - start,
            }
        )
        if progress:
            progress(
                f"{algorithm} fold {fold_id}: inner={inner_score:.4f} "
                f"outer tuned={tuned_metrics['roc_auc']:.4f} "
                f"default={default_metrics['roc_auc']:.4f}"
            )

    tuned_auc = [f["outer_tuned"]["roc_auc"] for f in folds]
    default_auc = [f["outer_default"]["roc_auc"] for f in folds]
    inner_auc = [f["inner_cv_best_auc"] for f in folds]
    n_val = len(x_dev) // outer_splits
    n_fit = len(x_dev) - n_val
    paired = st.corrected_resampled_ttest(tuned_auc, default_auc, n_train=n_fit, n_test=n_val)
    return {
        "algorithm": algorithm,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "n_trials": n_trials,
        "folds": folds,
        "summary": {
            "outer_tuned_roc_auc": summarize_distribution(tuned_auc),
            "outer_default_roc_auc": summarize_distribution(default_auc),
            "inner_best_roc_auc": summarize_distribution(inner_auc),
            "optimism_inner_minus_outer": float(np.mean(inner_auc) - np.mean(tuned_auc)),
            "tuned_minus_default": paired,
        },
    }


def final_tuning_and_holdout(
    algorithm: str,
    x_dev: pd.DataFrame,
    y_dev: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    n_trials: int = 12,
    inner_splits: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Elige hiperparámetros con TODO el desarrollo y evalúa el holdout una vez."""
    best, cv_auc = tune_hyperparameters(
        algorithm, x_dev, y_dev.to_numpy(), n_trials, inner_splits, seed
    )
    model = make_pipeline(algorithm, best)
    model.fit(x_dev, y_dev.to_numpy())
    p = model.predict_proba(x_test)[:, 1]
    y_arr = y_test.to_numpy()
    default_model = make_pipeline(algorithm, dict(DEFAULT_PARAMS[algorithm]))
    default_model.fit(x_dev, y_dev.to_numpy())
    p_default = default_model.predict_proba(x_test)[:, 1]
    return {
        "algorithm": algorithm,
        "best_params": best,
        "dev_inner_cv_auc": cv_auc,
        "holdout_tuned": _fold_metrics(y_arr, p),
        "holdout_default": _fold_metrics(y_arr, p_default),
        "holdout_auc_delong": st.delong_auc_ci(y_arr, p),
        "delong_tuned_vs_default": st.delong_roc_test(y_arr, p, p_default),
    }


def run(
    raw_data: pd.DataFrame,
    output_path: Path | str = "reports/nested_cv.json",
    algorithms: tuple[str, ...] = SEARCH_ALGORITHMS,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_trials: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    """Ejecuta todo y guarda el avance tras cada algoritmo (reanudable)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"nested": {}, "final": {}}
    if output_path.is_file():
        result = json.loads(output_path.read_text(encoding="utf-8"))

    x_dev, x_test, y_dev, y_test = split_raw_features(raw_data, random_state=seed)
    result["config"] = {
        "outer_splits": outer_splits, "inner_splits": inner_splits,
        "n_trials": n_trials, "seed": seed, "sampler": "Optuna TPE",
        "objective": "roc_auc (media CV interna)", "n_dev": len(x_dev), "n_holdout": len(x_test),
    }
    for algorithm in algorithms:
        if algorithm in result["nested"] and algorithm in result["final"]:
            logger.info("Se omite %s (ya calculado).", algorithm)
            continue
        logger.info("Nested CV: %s", algorithm)
        result["nested"][algorithm] = nested_cross_validation(
            algorithm, x_dev, y_dev, outer_splits, inner_splits, n_trials, seed,
            progress=logger.info,
        )
        result["final"][algorithm] = final_tuning_and_holdout(
            algorithm, x_dev, y_dev, x_test, y_test, n_trials, inner_splits, seed
        )
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    from src.data_pipeline.validate import clean_out_of_range_rows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--algorithms", nargs="+", default=list(SEARCH_ALGORITHMS))
    parser.add_argument("--output", default="reports/nested_cv.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    data = clean_out_of_range_rows(
        pd.read_csv("DATA/GiveMeSomeCredit/cs-training.csv", index_col=0)
    )
    run(data, args.output, tuple(args.algorithms), args.outer_splits, args.inner_splits,
        args.n_trials)
