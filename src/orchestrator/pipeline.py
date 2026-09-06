"""Prefect flow for data validation, training and model promotion."""

from pathlib import Path

import pandas as pd
from prefect import flow, task

from src.data_pipeline.ingest import download_give_me_some_credit
from src.data_pipeline.validate import clean_out_of_range_rows, validate_input_data
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
def train_task(data: pd.DataFrame) -> dict[str, float | str]:
    return train_model(data)


@flow(name="credit-risk-continuous-training")
def retraining_pipeline() -> dict[str, float | str | bool]:
    data_path = ingest_task()
    data = validate_task(data_path)
    metrics = train_task(data)
    promoted = evaluate_and_promote(float(metrics["roc_auc"]))
    return {**metrics, "promoted": promoted}
