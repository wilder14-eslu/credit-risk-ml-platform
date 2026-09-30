"""Tamaños de efecto y significancia práctica.

Significancia estadística responde "¿la diferencia es distinguible del
ruido?"; significancia práctica responde "¿la diferencia importa?". Con
miles de observaciones una diferencia de 0.001 de AUC puede ser
estadísticamente significativa y operativamente irrelevante.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Margen de equivalencia práctica para ROC-AUC. Es una convención de trabajo
# (no un estándar regulatorio): diferencias menores a 0.005 puntos de AUC no
# cambian de forma apreciable el ordenamiento de riesgo. Es un parámetro.
DEFAULT_PRACTICAL_MARGIN = 0.005


def paired_cohens_dz(differences: Sequence[float]) -> float:
    """d_z de Cohen para diferencias pareadas: media / desviación estándar.

    Devuelve 0.0 si no hay variabilidad o hay menos de dos diferencias.
    """
    d = np.asarray(differences, dtype=float)
    if d.size < 2:
        return 0.0
    std = float(d.std(ddof=1))
    if np.isclose(std, 0.0):
        return 0.0
    return float(d.mean() / std)


def interpret_difference(
    delta: float,
    adjusted_p_value: float,
    practical_margin: float = DEFAULT_PRACTICAL_MARGIN,
    alpha: float = 0.05,
) -> str:
    """Lectura conjunta de significancia estadística y práctica.

    ``delta`` = métrica(A) - métrica(B) (positivo favorece a A).
    """
    significant = adjusted_p_value < alpha
    material = abs(delta) >= practical_margin
    if significant and material:
        winner = "A" if delta > 0 else "B"
        return f"Diferencia significativa y material a favor de {winner}"
    if significant and not material:
        return "Estadísticamente significativa pero prácticamente irrelevante"
    if not significant and material:
        return "Diferencia de tamaño relevante pero no concluyente (incertidumbre alta)"
    return "Sin evidencia de diferencia: modelos prácticamente equivalentes"
