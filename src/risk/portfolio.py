"""Análisis de cartera: ganancia acumulada, lift, concentración y segmentos.

Los escenarios de tasa de aprobación son **analíticos**: muestran cómo
cambia el riesgo de la cartera aprobada, no son recomendaciones de crédito.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.ml import stats as st


def gains_curve(y_true: Any, probabilities: Any, n_points: int = 100) -> dict[str, np.ndarray]:
    """Curva de ganancia acumulada y lift al revisar primero los más riesgosos."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    order = np.argsort(-p, kind="stable")
    cum_defaults = np.cumsum(y[order])
    total = y.sum()
    n = len(y)
    k = np.unique(np.linspace(1, n, n_points).astype(int))
    population_share = k / n
    captured = cum_defaults[k - 1] / total if total else np.zeros_like(population_share)
    lift = np.divide(captured, population_share, out=np.zeros_like(captured),
                     where=population_share > 0)
    return {"population_share": population_share, "captured_defaults": captured, "lift": lift}


def risk_concentration(
    y_true: Any, probabilities: Any, shares: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
) -> dict[str, Any]:
    """Qué parte de los defaults se concentra en el X % más riesgoso."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    order = np.argsort(-p, kind="stable")
    cum = np.cumsum(y[order])
    total = y.sum()
    n = len(y)
    top = {}
    for share in shares:
        k = max(round(share * n), 1)
        top[f"top_{int(share * 100)}pct"] = {
            "defaults_captured": float(cum[k - 1] / total) if total else 0.0,
            "default_rate": float(cum[k - 1] / k),
            "lift": float(cum[k - 1] / k / y.mean()) if y.mean() > 0 else 0.0,
        }
    return {
        "top_segments": top,
        "base_default_rate": float(y.mean()),
        # Índice de concentración de defaults = 2 x AUC_gain - 1 (equivale al Gini).
        "gini": float(2 * roc_auc_score(y, p) - 1),
    }


def approval_scenarios(
    y_true: Any, probabilities: Any, approval_rates: tuple[float, ...] = (0.7, 0.8, 0.9, 0.95)
) -> list[dict[str, float]]:
    """Riesgo de la cartera aprobada a distintas tasas de aprobación."""
    rows = st.approval_strategy_curve(y_true, probabilities, approval_rates=approval_rates)
    base = float(np.mean(np.asarray(y_true)))
    return [
        {**row, "default_rate_reduction_vs_approve_all": 1 - row["bad_rate_approved"] / base}
        for row in rows
    ]


def segment_performance(
    y_true: Any,
    probabilities: Any,
    segments: pd.Series,
    n_boot: int = 300,
    min_size: int = 200,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """ROC-AUC (IC bootstrap) y tasa de default por segmento de población."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    seg = pd.Series(np.asarray(segments)).astype(str).to_numpy()
    rows = []
    for name in sorted(set(seg)):
        mask = seg == name
        ys, ps = y[mask], p[mask]
        row: dict[str, Any] = {
            "segment": name,
            "n": int(mask.sum()),
            "defaults": int(ys.sum()),
            "default_rate": float(ys.mean()),
            "mean_predicted_pd": float(ps.mean()),
        }
        if mask.sum() >= min_size and 0 < ys.sum() < len(ys):
            auc: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score
            ci = st.bootstrap_ci(ys, ps, auc, n_boot=n_boot, random_state=random_state)
            row.update(roc_auc=ci["estimate"], roc_auc_ci_low=ci["ci_low"],
                       roc_auc_ci_high=ci["ci_high"])
        else:
            row["note"] = "muestra insuficiente para estimar AUC con incertidumbre"
        rows.append(row)
    return rows
