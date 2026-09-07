"""Prefect flow for data validation, model benchmarking/training and
promotion.

Mirrors the platform's architecture: ingest -> validate -> **MODEL
BENCHMARK** (Logistic Regression baseline + XGBoost + LightGBM + CatBoost,
see `src.ml.benchmark`) -> pick the best candidate -> train/register it for
real -> `evaluate_and_promote` (the champion/challenger gate against
whatever is currently in `Production`).
"""

from pathlib import Path

import pandas as pd
from prefect import flow, task

from src.data_pipeline.ingest import download_give_me_some_credit
from src.data_pipeline.validate import clean_out_of_range_rows, validate_input_data
from src.ml.benchmark import comparison_table, run_benchmark, select_best_model
from src.ml.train import train_model
from src.ml.validate_model import evaluate_and_promote


@task(retries=2, retry_delay_seconds=30)
def ingest_task() -> Path:
    return download_give_me_some_credit()


@task
def validate_task(data_path: Path) -> pd.DataFrame:
    csv_files = sorted(data_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")
    data = pd.read_csv(csv_files[0])
    data = clean_out_of_range_rows(data)
    validate_input_data(data, require_target=True)
    return data


@task
def benchmark_task(data: pd.DataFrame) -> tuple[str, list[dict]]:
    """Run the model benchmark and return the winning algorithm plus a
    print/log-friendly comparison table of every candidate that ran."""
    results = run_benchmark(data)
    table = comparison_table(results)
    winner = select_best_model(results)
    return winner, table


@task
def train_task(data: pd.DataFrame, algorithm: str) -> dict[str, float | str]:
    """Train (and register) the winning algorithm for real -- this is the
    run that becomes a candidate in the MLflow Model Registry."""
    return train_model(data, algorithm=algorithm)


@flow(name="credit-risk-continuous-training")
def retraining_pipeline() -> dict[str, float | str | bool]:
    data_path = ingest_task()
    data = validate_task(data_path)

    winner, table = benchmark_task(data)
    metrics = train_task(data, winner)

    promoted = evaluate_and_promote(float(metrics["roc_auc"]))
    return {**metrics, "benchmark": table, "promoted": promoted}
