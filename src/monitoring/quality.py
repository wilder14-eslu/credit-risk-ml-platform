"""Calidad de datos en monitoreo: faltantes, deriva de esquema y valores inválidos."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.feature_store.features import load_feature_schema


def data_quality_report(
    current: pd.DataFrame,
    reference_missing_rates: dict[str, float] | None = None,
    missing_increase_alert: float = 0.05,
) -> dict[str, Any]:
    """Compara una ventana de datos con el esquema y con la tasa de faltantes de referencia.

    * ``schema``: columnas esperadas que faltan e inesperadas que aparecen.
    * ``missingness``: tasa actual, de referencia y alerta si sube más de
      ``missing_increase_alert`` (puntos absolutos).
    * ``invalid_values``: filas fuera de los límites min/max del esquema.
    """
    definitions = load_feature_schema()["features"]
    expected = list(definitions)
    missing_columns = [c for c in expected if c not in current.columns]
    unexpected_columns = [c for c in current.columns if c not in expected]

    reference = reference_missing_rates or {}
    missingness: dict[str, dict[str, Any]] = {}
    invalid: dict[str, dict[str, Any]] = {}
    alerts: list[str] = []
    for name, definition in definitions.items():
        if name not in current.columns:
            continue
        values = pd.to_numeric(current[name], errors="coerce")
        rate = float(values.isna().mean()) if len(values) else 0.0
        ref = float(reference.get(name, 0.0))
        alert = (rate - ref) > missing_increase_alert
        missingness[name] = {"current": rate, "reference": ref, "alert": bool(alert)}
        if alert:
            alerts.append(f"faltantes de {name} suben de {ref:.1%} a {rate:.1%}")

        below = int((values < definition["min"]).sum()) if "min" in definition else 0
        above = int((values > definition["max"]).sum()) if "max" in definition else 0
        invalid[name] = {"below_min": below, "above_max": above}
        if below or above:
            alerts.append(f"{name}: {below + above} valores fuera de rango")

    if missing_columns:
        alerts.append(f"columnas faltantes: {missing_columns}")
    if unexpected_columns:
        alerts.append(f"columnas inesperadas: {unexpected_columns}")
    return {
        "n_rows": len(current),
        "schema": {"missing_columns": missing_columns, "unexpected_columns": unexpected_columns,
                   "schema_drift": bool(missing_columns or unexpected_columns)},
        "missingness": missingness,
        "invalid_values": invalid,
        "alerts": alerts,
        "quality_ok": not alerts,
    }
