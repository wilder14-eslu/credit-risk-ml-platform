"""Marco de pérdida esperada: ``EL = PD x LGD x EAD``.

**Limitación deliberada:** "Give Me Some Credit" no contiene LGD ni EAD (ni
saldos, garantías, recuperaciones). Por eso este módulo *no* trae valores por
defecto y exige que quien lo use aporte LGD y EAD como supuestos explícitos o
estimados con datos propios. Las cifras de ejemplo de los tests son
sintéticas y nunca se reportan como resultados del proyecto.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class LgdEadUnavailableError(ValueError):
    """Se pidió pérdida esperada sin LGD/EAD explícitos."""


def _as_array(value: Any, name: str, n: int, low: float, high: float | None) -> np.ndarray:
    if value is None:
        raise LgdEadUnavailableError(
            f"{name} no está disponible en el dataset: aporta un valor o un vector "
            "explícito (supuesto documentado) antes de calcular la pérdida esperada."
        )
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name} debe ser escalar o tener longitud {n}.")
    if np.any(np.isnan(arr)) or np.any(arr < low) or (high is not None and np.any(arr > high)):
        bound = f"[{low}, {high}]" if high is not None else f">= {low}"
        raise ValueError(f"{name} fuera de rango {bound}.")
    return arr


def expected_loss(pd_values: Any, lgd: Any, ead: Any) -> np.ndarray:
    """Pérdida esperada por operación (misma unidad que ``ead``)."""
    pd_arr = np.asarray(pd_values, dtype=float)
    if pd_arr.ndim != 1:
        raise ValueError("pd_values debe ser un vector.")
    n = len(pd_arr)
    if np.any(np.isnan(pd_arr)) or np.any((pd_arr < 0) | (pd_arr > 1)):
        raise ValueError("PD fuera de [0, 1].")
    lgd_arr = _as_array(lgd, "LGD", n, 0.0, 1.0)
    ead_arr = _as_array(ead, "EAD", n, 0.0, None)
    return pd_arr * lgd_arr * ead_arr


def portfolio_expected_loss(pd_values: Any, lgd: Any, ead: Any) -> dict[str, float]:
    """Total y tasa de pérdida esperada de una cartera."""
    losses = expected_loss(pd_values, lgd, ead)
    total_ead = float(_as_array(ead, "EAD", len(losses), 0.0, None).sum())
    total = float(losses.sum())
    return {
        "expected_loss": total,
        "total_ead": total_ead,
        "expected_loss_rate": total / total_ead if total_ead > 0 else 0.0,
    }
