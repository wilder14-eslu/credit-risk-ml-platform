"""Data drift (covariate shift) y prediction drift: PSI por feature + test KS.

El perfil de referencia (bins por cuantiles del set de entrenamiento) se guarda
como tabla Delta `reference_profile` junto a la versión del modelo, de modo que
el monitoreo siempre compara contra la distribución con la que se entrenó el
champion vigente.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-6


def _edges(values: pd.Series, bins: int) -> list[float]:
    clean = values.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return [-np.inf, np.inf]
    qs = np.unique(np.quantile(clean, np.linspace(0, 1, bins + 1)[1:-1]))
    return [-np.inf, *qs.tolist(), np.inf]


def _hist(values: pd.Series, edges: list[float]) -> np.ndarray:
    clean = values.dropna().to_numpy(dtype=float)
    counts, _ = np.histogram(clean, bins=np.asarray(edges, dtype=float))
    total = max(counts.sum(), 1)
    return counts / total


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def build_reference_profile(
    features: pd.DataFrame, scores: np.ndarray | None = None, bins: int = 10, sample_size: int = 5000
) -> dict[str, Any]:
    """Perfil serializable: bordes, proporciones, nulos y una muestra para KS."""
    profile: dict[str, Any] = {"features": {}}
    rng = np.random.default_rng(42)
    frame = features.copy()
    if scores is not None:
        frame["__score__"] = np.asarray(scores, dtype=float)
    for col in frame.columns:
        edges = _edges(frame[col], bins)
        clean = frame[col].dropna().to_numpy(dtype=float)
        sample = rng.choice(clean, size=min(sample_size, clean.size), replace=False) if clean.size else clean
        profile["features"][col] = {
            "edges": edges,
            "proportions": _hist(frame[col], edges).tolist(),
            "null_rate": float(frame[col].isna().mean()),
            "mean": float(np.nanmean(clean)) if clean.size else float("nan"),
            "sample": sample.tolist(),
        }
    return profile


def feature_drift(
    profile: dict[str, Any],
    current: pd.DataFrame,
    psi_warning: float = 0.10,
    psi_alert: float = 0.25,
    ks_pvalue_alert: float = 0.01,
) -> pd.DataFrame:
    rows = []
    for col, ref in profile["features"].items():
        if col not in current.columns:
            continue
        cur = current[col]
        value = psi(np.asarray(ref["proportions"]), _hist(cur, ref["edges"]))
        clean = cur.dropna().to_numpy(dtype=float)
        if clean.size and len(ref["sample"]):
            ks = ks_2samp(np.asarray(ref["sample"]), clean)
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        else:
            ks_stat, ks_p = float("nan"), float("nan")
        status = "alerta" if value >= psi_alert else "warning" if value >= psi_warning else "estable"
        rows.append(
            {
                "feature": "prediction" if col == "__score__" else col,
                "psi": value,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "ks_drift": bool(ks_p < ks_pvalue_alert) if not np.isnan(ks_p) else False,
                "null_rate_ref": ref["null_rate"],
                "null_rate_cur": float(cur.isna().mean()),
                "mean_ref": ref["mean"],
                "mean_cur": float(np.nanmean(clean)) if clean.size else float("nan"),
                "status": status,
            }
        )
    return pd.DataFrame(rows)
