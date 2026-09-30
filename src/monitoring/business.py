"""Deriva de negocio y de predicciones: aprobación, composición de cartera y scores."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.monitoring.drift import population_stability_index


def score_drift(
    reference_scores: Any, current_scores: Any, buckets: int = 10, psi_threshold: float = 0.2
) -> dict[str, Any]:
    """PSI de la distribución de PD (predicción) con bins de cuantiles de referencia."""
    ref = np.asarray(reference_scores, dtype=float)
    cur = np.asarray(current_scores, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, buckets + 1)))
    edges[0], edges[-1] = -math.inf, math.inf
    ref_share = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_share = np.histogram(cur, bins=edges)[0] / len(cur)
    psi = population_stability_index(ref_share.tolist(), cur_share.tolist())
    return {
        "psi": float(psi),
        "mean_pd_reference": float(ref.mean()),
        "mean_pd_current": float(cur.mean()),
        "drift_detected": bool(psi > psi_threshold),
        "reference_share": ref_share.tolist(),
        "current_share": cur_share.tolist(),
        "threshold": psi_threshold,
    }


def business_drift(
    reference_scores: Any,
    current_scores: Any,
    decision_threshold: float,
    current_outcomes: Any = None,
    rejection_tolerance_pp: float = 0.03,
    composition_buckets: int = 10,
) -> dict[str, Any]:
    """Tasa de aprobación/rechazo, composición de cartera y default observado.

    La *composición* es la proporción actual de la población en cada decil
    de riesgo **de referencia** (bajo ausencia de deriva, cada decil ≈ 10 %).
    """
    ref = np.asarray(reference_scores, dtype=float)
    cur = np.asarray(current_scores, dtype=float)
    ref_rej = float((ref >= decision_threshold).mean())
    cur_rej = float((cur >= decision_threshold).mean())
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, composition_buckets + 1)))
    edges[0], edges[-1] = -math.inf, math.inf
    composition = (np.histogram(cur, bins=edges)[0] / len(cur)).tolist()

    report: dict[str, Any] = {
        "rejection_rate_reference": ref_rej,
        "rejection_rate_current": cur_rej,
        "approval_rate_current": 1 - cur_rej,
        "rejection_rate_change_pp": cur_rej - ref_rej,
        "rejection_drift": bool(abs(cur_rej - ref_rej) > rejection_tolerance_pp),
        "portfolio_composition_by_reference_decile": composition,
    }
    if current_outcomes is not None:
        outcomes = np.asarray(current_outcomes, dtype=int)
        report["observed_default_rate"] = float(outcomes.mean())
        report["observed_default_rate_approved"] = (
            float(outcomes[cur < decision_threshold].mean()) if (cur < decision_threshold).any()
            else None
        )
    return report
