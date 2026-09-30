"""Validación fuera de tiempo (out-of-time) preparada pero no ejecutable hoy.

"Give Me Some Credit" no tiene ninguna variable temporal (fecha de
originación, mes de observación, etc.). Por lo tanto **no se puede** hacer
validación fuera de tiempo con este dataset y no se inventan fechas.

Este módulo deja la arquitectura lista: cuando exista una columna temporal
real, ``out_of_time_split`` produce desarrollo (pasado) y OOT (futuro) sin
mezcla de períodos.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

OOT_UNAVAILABLE_MESSAGE = (
    "Out-of-time validation cannot be performed with the current dataset."
)

_TEMPORAL_CANDIDATES = (
    "date", "origination_date", "issue_d", "issue_date", "observation_date",
    "application_date", "period", "month", "year",
)


class OutOfTimeValidationUnavailable(RuntimeError):
    """El dataset no contiene una variable temporal real."""


def find_temporal_column(
    data: pd.DataFrame, candidates: Sequence[str] = _TEMPORAL_CANDIDATES
) -> str | None:
    lowered = {str(c).lower(): c for c in data.columns}
    for name in candidates:
        if name in lowered:
            return str(lowered[name])
    for column in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[column]):
            return str(column)
    return None


def out_of_time_split(
    data: pd.DataFrame, date_column: str | None = None, cutoff: str | pd.Timestamp | None = None,
    oot_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parte cronológicamente: todo lo anterior al corte es desarrollo."""
    column = date_column or find_temporal_column(data)
    if column is None or column not in data.columns:
        raise OutOfTimeValidationUnavailable(OOT_UNAVAILABLE_MESSAGE)
    dates = pd.to_datetime(data[column], errors="coerce")
    if dates.isna().all():
        raise OutOfTimeValidationUnavailable(OOT_UNAVAILABLE_MESSAGE)
    if cutoff is None:
        cutoff_ts = dates.quantile(1 - oot_fraction)
    else:
        cutoff_ts = pd.Timestamp(cutoff)
    development = data.loc[dates < cutoff_ts]
    out_of_time = data.loc[dates >= cutoff_ts]
    if development.empty or out_of_time.empty:
        raise ValueError("El corte deja vacío el desarrollo o el período fuera de tiempo.")
    return development, out_of_time
