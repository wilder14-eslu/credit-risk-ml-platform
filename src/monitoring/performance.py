"""Deriva de desempeño: discriminación y calibración en una ventana con outcomes."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.ml import stats as st
from src.ml.metrics import ks_statistic


def performance_panel(y_true: Any, probabilities: Any) -> dict[str, float]:
    """AUC, PR-AUC, KS, Brier, ECE, pendiente e intercepto de calibración."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.min() == y.max():
        raise ValueError("La ventana necesita ambas clases para medir desempeño.")
    calibration = st.calibration_slope_intercept(y, p)
    return {
        "n": len(y),
        "observed_default_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "ece": st.expected_calibration_error(y, p),
        "calibration_slope": calibration["slope"],
        "calibration_intercept": calibration["intercept"],
    }


def performance_drift(
    reference: dict[str, float],
    current: dict[str, float],
    max_auc_drop: float = 0.05,
    max_ks_drop: float = 0.05,
    max_brier_increase: float = 0.01,
    slope_band: tuple[float, float] = (0.8, 1.2),
) -> dict[str, Any]:
    """Compara un panel actual con el de referencia y marca las degradaciones."""
    deltas = {
        "roc_auc": current["roc_auc"] - reference["roc_auc"],
        "pr_auc": current["pr_auc"] - reference["pr_auc"],
        "ks": current["ks"] - reference["ks"],
        "brier": current["brier"] - reference["brier"],
        "ece": current["ece"] - reference["ece"],
    }
    flags = {
        "auc_degraded": deltas["roc_auc"] < -max_auc_drop,
        "ks_degraded": deltas["ks"] < -max_ks_drop,
        "brier_worse": deltas["brier"] > max_brier_increase,
        "calibration_slope_out_of_band": not (
            slope_band[0] <= current["calibration_slope"] <= slope_band[1]
        ),
    }
    return {"deltas": deltas, "flags": flags, "drift_detected": any(flags.values())}
