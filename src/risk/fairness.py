"""Diagnóstico de equidad por grupo (descriptivo, no causal).

En este dataset la única variable potencialmente sensible es ``age`` (en
muchas jurisdicciones está protegida o restringida en crédito). No hay
género, estado civil, etnia ni geografía, así que **no se puede** evaluar
equidad respecto de esos atributos. Las diferencias entre grupos describen
cómo se comporta el modelo; no prueban causalidad ni discriminación, ya que
los grupos difieren también en su riesgo base y en otras variables.

Las tasas llevan intervalos de Wilson 95 % y cada grupo informa su tamaño de
muestra; los grupos pequeños se marcan y no se interpretan.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

AGE_BINS = (18, 30, 40, 50, 60, 70, 121)
AGE_LABELS = ("18-29", "30-39", "40-49", "50-59", "60-69", "70+")


def age_groups(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=list(AGE_BINS), labels=list(AGE_LABELS), right=False)


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo de Wilson para una proporción (válido con n pequeño)."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (float(max(centre - half, 0.0)), float(min(centre + half, 1.0)))


def _rate(successes: int, n: int) -> dict[str, float]:
    low, high = wilson_interval(successes, n)
    return {"value": successes / n if n else float("nan"), "ci_low": low, "ci_high": high,
            "n": n}


def group_fairness_report(
    y_true: Any,
    probabilities: Any,
    groups: Any,
    threshold: float,
    min_group_size: int = 500,
    min_group_defaults: int = 30,
) -> dict[str, Any]:
    """Métricas por grupo a un umbral de rechazo (PD >= umbral) y disparidades
    frente al grupo de referencia (el más numeroso)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    g = pd.Series(np.asarray(groups)).astype(str).to_numpy()
    rejected = p >= threshold

    rows = []
    for name in pd.unique(g):
        if name == "nan":
            continue
        m = g == name
        ys, ps, rj = y[m], p[m], rejected[m]
        n = int(m.sum())
        defaults = int(ys.sum())
        goods = n - defaults
        tp = int((rj & (ys == 1)).sum())
        fp = int((rj & (ys == 0)).sum())
        small = n < min_group_size or defaults < min_group_defaults
        rows.append(
            {
                "group": name,
                "n": n,
                "defaults": defaults,
                "small_sample": bool(small),
                "rejection_rate": _rate(int(rj.sum()), n),
                "tpr_defaults_flagged": _rate(tp, defaults) if defaults else None,
                "fpr_goods_rejected": _rate(fp, goods) if goods else None,
                "precision": _rate(tp, int(rj.sum())) if rj.any() else None,
                "observed_default_rate": _rate(defaults, n),
                "mean_predicted_pd": float(ps.mean()),
                "calibration_gap_obs_minus_pred": float(ys.mean() - ps.mean()),
            }
        )
    rows.sort(key=lambda r: r["group"])

    reference = max(rows, key=lambda r: r["n"])
    disparities = []
    for row in rows:
        if row["group"] == reference["group"]:
            continue
        ref_rej = reference["rejection_rate"]["value"]
        disparities.append(
            {
                "group": row["group"],
                "reference": reference["group"],
                "rejection_rate_ratio": row["rejection_rate"]["value"] / ref_rej
                if ref_rej else float("nan"),
                "tpr_difference": _diff(row["tpr_defaults_flagged"],
                                        reference["tpr_defaults_flagged"]),
                "fpr_difference": _diff(row["fpr_goods_rejected"],
                                        reference["fpr_goods_rejected"]),
                "interpretable": not row["small_sample"],
            }
        )
    return {
        "threshold": threshold,
        "reference_group": reference["group"],
        "groups": rows,
        "disparities_vs_reference": disparities,
        "caveat": (
            "Diferencias descriptivas. No se atribuyen causas: los grupos difieren "
            "en riesgo base y en otras variables. Solo 'age' está disponible como "
            "atributo potencialmente sensible."
        ),
    }


def _diff(a: dict[str, float] | None, b: dict[str, float] | None) -> float:
    if a is None or b is None:
        return float("nan")
    return a["value"] - b["value"]
