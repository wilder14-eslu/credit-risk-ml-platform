"""Tests for the benchmark orchestration logic itself (`run_benchmark`,
`select_best_model`, `comparison_table`), using a stubbed `train_model` so
they do not depend on actually having every optional ML library (LightGBM,
CatBoost) installed -- exactly the scenario `run_benchmark` needs to
handle gracefully in the first place.
"""

import pandas as pd

import src.ml.benchmark as benchmark_module
from src.ml.benchmark import comparison_table, run_benchmark, select_best_model


def _fake_result(algorithm: str, roc_auc: float) -> dict:
    return {
        "algorithm": algorithm,
        "roc_auc": roc_auc,
        "test_metrics": {
            "roc_auc": roc_auc,
            "pr_auc": roc_auc - 0.05,
            "f1": 0.5,
            "recall": 0.5,
            "precision": 0.5,
            "brier_score": 0.1,
        },
        "diagnosis": {"label": "buen_ajuste", "gap": 0.01},
        "latency_ms_per_row": 0.2,
    }


def test_run_benchmark_skips_algorithms_whose_library_is_missing(monkeypatch) -> None:
    scores = {"logistic_regression": 0.75, "xgboost": 0.82}

    def fake_train_model(raw_data, model_path=None, register=False, algorithm="xgboost"):
        if algorithm == "catboost":
            raise ImportError("catboost no está instalado")
        return _fake_result(algorithm, scores[algorithm])

    monkeypatch.setattr(benchmark_module, "train_model", fake_train_model)

    results = run_benchmark(
        pd.DataFrame(), algorithms=("logistic_regression", "xgboost", "catboost")
    )
    assert set(results) == {"logistic_regression", "xgboost"}


def test_select_best_model_picks_the_highest_roc_auc() -> None:
    results = {
        "logistic_regression": _fake_result("logistic_regression", 0.75),
        "xgboost": _fake_result("xgboost", 0.82),
    }
    assert select_best_model(results) == "xgboost"


def test_comparison_table_is_sorted_best_first() -> None:
    results = {
        "logistic_regression": _fake_result("logistic_regression", 0.75),
        "xgboost": _fake_result("xgboost", 0.82),
    }
    table = comparison_table(results)
    assert [row["algorithm"] for row in table] == ["xgboost", "logistic_regression"]
