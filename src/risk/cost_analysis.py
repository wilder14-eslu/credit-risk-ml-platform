"""Umbral de decisión basado en costos para varias relaciones FN:FP.

El umbral de cada escenario se elige **solo con predicciones out-of-fold del
desarrollo**; el holdout únicamente mide el resultado. Los costos son
ilustrativos (unidades relativas), no pérdidas reales: ver
``src.risk.expected_loss`` para el vínculo con PD x LGD x EAD.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.ml import stats as st

DEFAULT_RATIOS = (1, 2, 3, 5, 10, 20)


def _decision_panel(y: np.ndarray, p: np.ndarray, threshold: float, cost_fn: float,
                    cost_fp: float) -> dict[str, float]:
    reject = p >= threshold
    approve = ~reject
    tp = int((reject & (y == 1)).sum())
    fp = int((reject & (y == 0)).sum())
    fn = int((approve & (y == 1)).sum())
    n = len(y)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / y.sum() if y.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "approval_rate": float(approve.mean()),
        "rejection_rate": float(reject.mean()),
        "default_rate_approved": float(y[approve].mean()) if approve.any() else 0.0,
        "expected_cost_per_applicant": float((cost_fn * fn + cost_fp * fp) / n),
        "approve_all_cost_per_applicant": float(cost_fn * y.sum() / n),
    }


def cost_ratio_table(
    y_dev: Any,
    oof_proba: Any,
    y_holdout: Any,
    holdout_proba: Any,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    cost_fp: float = 1.0,
) -> list[dict[str, float]]:
    """Una fila por relación FN:FP con el umbral óptimo (OOF) evaluado en holdout."""
    y_dev = np.asarray(y_dev, dtype=int)
    oof = np.asarray(oof_proba, dtype=float)
    y_hold = np.asarray(y_holdout, dtype=int)
    p_hold = np.asarray(holdout_proba, dtype=float)
    rows = []
    for ratio in ratios:
        cost_fn = float(ratio) * cost_fp
        analysis = st.threshold_analysis(y_dev, oof, cost_fn, cost_fp)
        threshold = analysis["min_cost"]["threshold"]
        panel = _decision_panel(y_hold, p_hold, threshold, cost_fn, cost_fp)
        rows.append(
            {
                "fn_fp_ratio": float(ratio),
                "bayes_threshold": float(analysis["bayes_threshold"]),
                **panel,
            }
        )
    return rows
