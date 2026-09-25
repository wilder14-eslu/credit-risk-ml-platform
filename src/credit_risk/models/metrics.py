"""Métricas de scoring crediticio y diagnóstico de ajuste."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS de separación entre la distribución de scores de buenos y malos."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)
    both = len(np.unique(y_true)) == 2
    auc = float(roc_auc_score(y_true, y_prob)) if both else float("nan")
    return {
        "roc_auc": auc,
        "gini": 2 * auc - 1 if both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both else float("nan"),
        "ks": ks_statistic(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "ece": expected_calibration_error(y_true, y_prob),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "default_rate_observed": float(y_true.mean()),
        "default_rate_predicted": float(y_prob.mean()),
    }


def optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, cost_fn: float = 5.0, cost_fp: float = 1.0
) -> float:
    """Umbral que minimiza el costo esperado (FN: otorgar a un moroso es más caro)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    grid = np.linspace(0.02, 0.9, 89)
    costs = []
    for t in grid:
        pred = y_prob >= t
        fn = np.sum((~pred) & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        costs.append(cost_fn * fn + cost_fp * fp)
    return float(grid[int(np.argmin(costs))])


def inference_latency_ms(predict_fn, features: pd.DataFrame, repeats: int = 3) -> float:
    """Latencia media por solicitante (ms) midiendo lotes pequeños."""
    sample = features.head(min(len(features), 500))
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        predict_fn(sample)
        best = min(best, time.perf_counter() - start)
    return 1000.0 * best / max(len(sample), 1)


def diagnose_fit(
    train_auc: float,
    test_auc: float,
    overfit_gap_threshold: float = 0.07,
    underfit_auc_threshold: float = 0.65,
) -> dict[str, Any]:
    gap = train_auc - test_auc
    if gap > overfit_gap_threshold:
        label = "sobreajuste"
    elif train_auc < underfit_auc_threshold and test_auc < underfit_auc_threshold:
        label = "subajuste"
    else:
        label = "buen_ajuste"
    return {"fit_diagnosis": label, "auc_gap": float(gap)}
