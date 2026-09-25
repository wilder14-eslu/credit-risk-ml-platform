"""Simulador de tráfico de producción con escenarios de drift controlados.

Toma solicitantes del `gold_production_pool` (nunca vistos en entrenamiento) y
les aplica un escenario para poder demostrar, de forma reproducible, que el
monitoreo distingue data drift, concept drift y prior shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import target_name

SCENARIOS = ("none", "covariate", "concept", "prior", "mixed")


def apply_scenario(batch: pd.DataFrame, scenario: str, intensity: float = 1.0, seed: int = 0) -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError(f"Escenario desconocido: {scenario}. Usa uno de {SCENARIOS}")
    rng = np.random.default_rng(seed)
    out = batch.copy()
    y = target_name()

    if scenario in ("covariate", "mixed"):
        # Recesión: cae el ingreso, sube la utilización y el endeudamiento.
        out["monthly_income"] = out["monthly_income"] * (1 - 0.40 * intensity)
        out["revolving_utilization_unsecured"] = out["revolving_utilization_unsecured"] * (
            1 + 0.50 * intensity
        )
        out["debt_ratio"] = out["debt_ratio"] * (1 + 0.60 * intensity)
        out["age"] = (out["age"] - rng.integers(0, int(8 * intensity) + 1, len(out))).clip(lower=18)

    if scenario in ("concept", "mixed") and y in out:
        # Cambia P(y|x) sin mover X (p.ej. un shock sectorial): perfiles que el
        # modelo considera de bajo riesgo empiezan a incumplir, y el historial
        # de atrasos deja de ser tan predictivo (refinanciaciones masivas).
        low_risk = (out["revolving_utilization_unsecured"] < 0.3) & (out["age"] >= 40)
        shock = rng.random(len(out)) < min(0.15 * intensity, 1.0)
        out.loc[low_risk & shock, y] = 1
        past_due = (
            out["number_of_times_90_days_late"].fillna(0)
            + out["number_of_time_30_59_days_past_due"].fillna(0)
        ) > 0
        relief = rng.random(len(out)) < min(0.7 * intensity, 1.0)
        out.loc[past_due & relief, y] = 0

    if scenario == "prior" and y in out:
        extra = rng.random(len(out)) < 0.06 * intensity
        out.loc[extra, y] = 1
    return out
