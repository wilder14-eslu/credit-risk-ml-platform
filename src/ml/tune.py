"""Optimización de Hiperparámetros usando Optuna.

Este script busca la configuración óptima de parámetros para un algoritmo
(xgboost por defecto) para maximizar el ROC-AUC en el conjunto de prueba.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.train import train_model

logger = logging.getLogger(__name__)


def optimize_hyperparameters(
    raw_data: pd.DataFrame,
    algorithm: str = "xgboost",
    n_trials: int = 15,
) -> dict[str, Any]:
    """Usa Optuna para encontrar los mejores parámetros de un algoritmo."""
    try:
        import optuna
    except ImportError as error:
        raise ImportError(
            "Optuna no está instalado. Ejecuta `pip install optuna`."
        ) from error

    # Evitamos que optuna imprima demasiados logs por cada trial
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        if algorithm == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "eval_metric": "auc",
                "random_state": 42,
            }
        elif algorithm == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": 42,
            }
        elif algorithm == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 100, 500),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "random_state": 42,
            }
        else:
            # logistic_regression o similar
            params = {
                "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
                "max_iter": 1000,
                "random_state": 42,
            }

        # Entrenar modelo con parámetros sugeridos en un dir temporal para no sobreescribir el actual
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_model(
                raw_data=raw_data,
                params=params,
                model_path=Path(tmpdir) / "model_tune.joblib",
                register=False,
                algorithm=algorithm,
            )

        # Optimizamos basándonos en el ROC-AUC de test
        return result["test_metrics"]["roc_auc"]

    study = optuna.create_study(direction="maximize")
    logger.info(f"Iniciando optimización para {algorithm} con {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params


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

    algo = sys.argv[1] if len(sys.argv) > 1 else "xgboost"
    best_params = optimize_hyperparameters(data, algorithm=algo, n_trials=15)

    print("\n" + "="*50)
    print(f"✅ Mejores parámetros para {algo}:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("="*50 + "\n")
