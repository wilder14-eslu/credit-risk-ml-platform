"""MODEL BENCHMARK: train every candidate algorithm on the same
train/validation/test split and compare them before picking a champion.

This is the "Comparación de modelos -> BEST MODEL" step of the platform's
architecture: a Logistic Regression baseline plus three gradient-boosting
libraries, scored on ROC-AUC, PR-AUC, F1, recall, precision, calibration
(Brier score/log loss) and inference latency -- not just accuracy, so a
model that is marginally more accurate but far slower or badly calibrated
does not automatically win.

Run with:
    python -m src.ml.benchmark
or from the automated retraining flow (`src.orchestrator.pipeline`), which
runs this on every scheduled retrain and only sends the winner through the
`evaluate_and_promote` champion/challenger gate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.train import ALGORITHMS, train_model

logger = logging.getLogger(__name__)

DEFAULT_SELECTION_METRIC = "roc_auc"


def run_benchmark(
    raw_data: pd.DataFrame,
    algorithms: tuple[str, ...] = ALGORITHMS,
    workdir: Path | str = "data/processed/model_comparison",
) -> dict[str, dict[str, Any]]:
    """Train every algorithm in ``algorithms`` on the same split and return
    each one's full metrics (see `src.ml.train.train_model`).

    Every candidate is trained with ``register=False``: nothing touches the
    MLflow Model Registry here. Algorithms whose optional library is not
    installed (LightGBM/CatBoost) are skipped with a warning instead of
    failing the whole benchmark.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for algorithm in algorithms:
        logger.info("Entrenando candidato: %s", algorithm)
        try:
            results[algorithm] = train_model(
                raw_data,
                model_path=workdir / f"{algorithm}.joblib",
                register=False,
                algorithm=algorithm,
            )
        except ImportError as error:
            logger.warning("Se omite %s: %s", algorithm, error)
    return results


def select_best_model(
    results: dict[str, dict[str, Any]], metric: str = DEFAULT_SELECTION_METRIC
) -> str:
    """Pick the winning algorithm by test-set ``metric`` (default ROC-AUC).

    Ties (or near-ties, in practice) are broken by the caller inspecting
    ``comparison_table``; this always returns a single winner.
    """
    if not results:
        raise ValueError("No hay candidatos entrenados para elegir un modelo.")
    return max(results, key=lambda algorithm: results[algorithm]["test_metrics"][metric])


def comparison_table(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Flat, print/report-friendly view of the benchmark, one row per
    algorithm, sorted best-to-worst by ROC-AUC."""
    rows = [
        {
            "algorithm": algorithm,
            "roc_auc": result["test_metrics"]["roc_auc"],
            "pr_auc": result["test_metrics"]["pr_auc"],
            "f1": result["test_metrics"]["f1"],
            "recall": result["test_metrics"]["recall"],
            "precision": result["test_metrics"]["precision"],
            "brier_score": result["test_metrics"]["brier_score"],
            "latency_ms_per_row": result["latency_ms_per_row"],
            "diagnosis": result["diagnosis"]["label"],
        }
        for algorithm, result in results.items()
    ]
    return sorted(rows, key=lambda row: row["roc_auc"], reverse=True)


if __name__ == "__main__":
    import json
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

    benchmark_results = run_benchmark(data)
    table = comparison_table(benchmark_results)
    print(json.dumps(table, indent=2))

    winner = select_best_model(benchmark_results)
    print(f"\n🏆 Mejor modelo: {winner}")
