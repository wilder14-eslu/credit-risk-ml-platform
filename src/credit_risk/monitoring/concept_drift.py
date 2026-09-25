"""Concept drift: cambios en P(y|x) detectados con etiquetas reales (outcomes).

Tres familias de señales, porque el data drift por sí solo no detecta que la
relación entre variables y default cambió:

1. Degradación de performance en la ventana vs la referencia del entrenamiento
   (AUC, KS, Brier) y descalibración (tasa predicha vs observada).
2. Detectores secuenciales sobre el stream de errores ordenado en el tiempo:
   DDM (Gama et al., 2004) sobre la tasa de error y Page-Hinkley sobre el log-loss.
3. Prior shift / label drift: test de dos proporciones sobre la tasa de default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm

from credit_risk.models.metrics import classification_metrics


@dataclass
class DDMResult:
    state: str  # estable | warning | drift
    index: int | None
    min_p_plus_s: float


def ddm(
    errors: np.ndarray, warning_level: float = 2.0, drift_level: float = 3.0, min_n: int = 100
) -> DDMResult:
    """Drift Detection Method sobre un stream binario de errores (1 = error)."""
    p_min, s_min = math.inf, math.inf
    state, index = "estable", None
    n, p = 0, 0.0
    for i, err in enumerate(np.asarray(errors, dtype=float)):
        n += 1
        p += (err - p) / n
        s = math.sqrt(p * (1 - p) / n)
        if n < min_n or s == 0:  # evita un mínimo degenerado cuando aún no hay errores
            continue
        if p + s < p_min + s_min:
            p_min, s_min = p, s
        if p + s > p_min + drift_level * s_min:
            return DDMResult("drift", i, p_min + s_min)
        if p + s > p_min + warning_level * s_min and state == "estable":
            state, index = "warning", i
    return DDMResult(state, index, p_min + s_min if p_min < math.inf else float("nan"))


def page_hinkley(values: np.ndarray, delta: float = 0.005, lamb: float = 50.0) -> dict[str, Any]:
    """Page-Hinkley para aumentos sostenidos en la media (p.ej. log-loss por fila)."""
    mean, cum, cum_min = 0.0, 0.0, 0.0
    for i, x in enumerate(np.asarray(values, dtype=float), start=1):
        mean += (x - mean) / i
        cum += x - mean - delta
        cum_min = min(cum_min, cum)
        if cum - cum_min > lamb:
            return {"drift": True, "index": i - 1, "statistic": cum - cum_min}
    return {"drift": False, "index": None, "statistic": cum - cum_min}


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Devuelve (z, p-valor bilateral)."""
    if min(n1, n2) == 0:
        return float("nan"), float("nan")
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return float(z), float(2 * (1 - norm.cdf(abs(z))))


def concept_drift_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    reference: dict[str, float],
    cfg: dict[str, Any],
    threshold: float = 0.5,
    reference_n: int = 10_000,
) -> dict[str, Any]:
    """Evalúa la ventana etiquetada (ordenada por tiempo) contra la referencia."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    current = classification_metrics(y_true, y_prob, threshold)

    auc_drop = reference["roc_auc"] - current["roc_auc"]
    ks_drop = reference["ks"] - current["ks"]
    brier_up = current["brier"] - reference["brier"]
    calib_gap = current["default_rate_predicted"] - current["default_rate_observed"]

    errors = ((y_prob >= threshold).astype(int) != y_true).astype(int)
    logloss_rows = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    ddm_res = ddm(errors, cfg["ddm_warning_level"], cfg["ddm_drift_level"])
    ph = page_hinkley(logloss_rows, cfg["page_hinkley_delta"], cfg["page_hinkley_lambda"])

    ref_rate = reference["default_rate_observed"]
    z, p_value = two_proportion_ztest(
        int(y_true.sum()), len(y_true), int(round(ref_rate * reference_n)), reference_n
    )
    prior_shift = abs(current["default_rate_observed"] - ref_rate) >= cfg["default_rate_abs_change"]

    signals = {
        "auc_degradation": bool(auc_drop >= cfg["auc_drop_alert"]),
        "ks_degradation": bool(ks_drop >= cfg["ks_stat_drop_alert"]),
        "calibration_degradation": bool(brier_up >= cfg["brier_increase_alert"]),
        "ddm_drift": ddm_res.state == "drift",
        "page_hinkley_drift": bool(ph["drift"]),
        "prior_shift": bool(prior_shift and p_value < 0.05),
    }
    return {
        "current": current,
        "auc_drop": float(auc_drop),
        "ks_drop": float(ks_drop),
        "brier_increase": float(brier_up),
        "calibration_gap": float(calib_gap),
        "ddm_state": ddm_res.state,
        "ddm_index": ddm_res.index,
        "page_hinkley_statistic": float(ph["statistic"]),
        "prior_shift_pvalue": p_value,
        "prior_shift_z": z,
        "signals": signals,
        "concept_drift": bool(
            signals["auc_degradation"]
            or signals["ks_degradation"]
            or (signals["ddm_drift"] and signals["page_hinkley_drift"])
        ),
    }
