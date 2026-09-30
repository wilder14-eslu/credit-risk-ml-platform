"""Comparación pareada de modelos con corrección por comparaciones múltiples.

Dos familias de tests independientes entre sí:

* **Validación cruzada:** t-test remuestreado corregido de Nadeau y Bengio
  (2003) sobre métricas por fold (los folds comparten filas de entrenamiento).
* **Holdout:** test de DeLong (1988) sobre las mismas observaciones.

Con ``k`` modelos hay ``k(k-1)/2`` comparaciones, así que los p-valores se
ajustan con Holm dentro de cada familia.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats

from src.inference.effect_sizes import (
    DEFAULT_PRACTICAL_MARGIN,
    interpret_difference,
    paired_cohens_dz,
)
from src.inference.multiple_testing import holm_adjust
from src.ml import stats as st


def pairwise_cv_comparison(
    fold_scores: Mapping[str, Sequence[float]],
    n_train: int,
    n_test: int,
    practical_margin: float = DEFAULT_PRACTICAL_MARGIN,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Todas las parejas de modelos sobre métricas por fold (mismos folds).

    Devuelve una fila por pareja con delta, IC 95 % (Nadeau-Bengio), p-valor,
    p-valor ajustado por Holm, tamaño de efecto d_z e interpretación.
    """
    names = list(fold_scores)
    rows: list[dict[str, Any]] = []
    for a, b in combinations(names, 2):
        va = np.asarray(fold_scores[a], dtype=float)
        vb = np.asarray(fold_scores[b], dtype=float)
        diff = va - vb
        test = st.corrected_resampled_ttest(va, vb, n_train=n_train, n_test=n_test)
        j = len(diff)
        half = float(
            stats.t.ppf(1 - alpha / 2, j - 1)
            * np.sqrt((1.0 / j + n_test / n_train) * diff.var(ddof=1))
        )
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "delta": test["diff_mean"],
                "ci_low": test["diff_mean"] - half,
                "ci_high": test["diff_mean"] + half,
                "p_value": test["p_value"],
                "effect_size_dz": paired_cohens_dz(diff),
            }
        )
    adjusted = holm_adjust([r["p_value"] for r in rows])
    for row, p_adj in zip(rows, adjusted, strict=True):
        row["p_adjusted_holm"] = p_adj
        row["interpretation"] = interpret_difference(
            row["delta"], p_adj, practical_margin, alpha
        )
    return rows


def pairwise_delong_comparison(
    y_true: Any,
    scores: Mapping[str, Any],
    practical_margin: float = DEFAULT_PRACTICAL_MARGIN,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Todas las parejas en holdout con DeLong; IC 95 % de la diferencia."""
    names = list(scores)
    rows: list[dict[str, Any]] = []
    z_crit = stats.norm.ppf(1 - alpha / 2)
    for a, b in combinations(names, 2):
        test = st.delong_roc_test(y_true, scores[a], scores[b])
        se = abs(test["diff"] / test["z"]) if test["z"] else 0.0
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "delta": test["diff"],
                "ci_low": test["diff"] - z_crit * se,
                "ci_high": test["diff"] + z_crit * se,
                "p_value": test["p_value"],
            }
        )
    adjusted = holm_adjust([r["p_value"] for r in rows])
    for row, p_adj in zip(rows, adjusted, strict=True):
        row["p_adjusted_holm"] = p_adj
        row["interpretation"] = interpret_difference(
            row["delta"], p_adj, practical_margin, alpha
        )
    return rows
