"""Corrección por comparaciones múltiples (Holm-Bonferroni)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """P-valores ajustados por el método step-down de Holm (1979).

    Controla la tasa de error familiar (FWER) y es uniformemente más potente
    que Bonferroni. Con ``m`` hipótesis y p-valores ordenados
    ``p(1) <= ... <= p(m)``, el ajustado es
    ``max_{j <= i} min(1, (m - j + 1) * p(j))``. El resultado conserva el
    orden original de la entrada.
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return []
    if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        raise ValueError("Los p-valores deben estar en [0, 1].")
    m = p.size
    order = np.argsort(p, kind="stable")
    scaled = (m - np.arange(m)) * p[order]
    adjusted_sorted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return [float(value) for value in adjusted]
