"""Escenarios de stress sobre las variables de entrada.

Es **análisis de escenarios**: responde "¿qué le pasa a la PD estimada y a
las decisiones si la población empeora así?". No es un pronóstico, no
estima probabilidades de ocurrencia de los escenarios y los choques son
supuestos del analista (ver ``config/stress_scenarios.yaml``). La pérdida
esperada solo se calcula si el usuario aporta LGD y EAD (ver
``src.risk.expected_loss``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.risk.expected_loss import portfolio_expected_loss

DEFAULT_SCENARIO_PATH = Path("config/stress_scenarios.yaml")


def load_scenarios(path: Path | str = DEFAULT_SCENARIO_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_scenario(features: pd.DataFrame, scenario: dict[str, Any], seed: int = 42) -> pd.DataFrame:
    """Copia de ``features`` con los choques del escenario aplicados."""
    shocked = features.copy()
    for column, factor in (scenario.get("multiply") or {}).items():
        if column not in shocked.columns:
            raise KeyError(f"Variable desconocida en el escenario: {column}")
        shocked[column] = shocked[column] * float(factor)
    rng = np.random.default_rng(seed)
    for column, fraction in (scenario.get("delinquency_shock") or {}).items():
        if column not in shocked.columns:
            raise KeyError(f"Variable desconocida en el escenario: {column}")
        hit = rng.random(len(shocked)) < float(fraction)
        shocked.loc[hit, column] = shocked.loc[hit, column] + 1
    return shocked


def run_stress_test(
    model: Any,
    features: pd.DataFrame,
    threshold: float,
    scenarios: dict[str, Any] | None = None,
    high_risk_pd: float | None = None,
    lgd: Any = None,
    ead: Any = None,
) -> list[dict[str, Any]]:
    """Evalúa cada escenario y lo compara con ``base``.

    ``high_risk_pd`` define "población de alto riesgo" (por defecto, el
    umbral de decisión). ``lgd``/``ead`` son opcionales: sin ambos no se
    reporta pérdida esperada.
    """
    config = scenarios or load_scenarios()
    seed = int(config.get("seed", 42))
    cutoff = threshold if high_risk_pd is None else high_risk_pd

    rows: list[dict[str, Any]] = []
    for name, scenario in config["scenarios"].items():
        shocked = apply_scenario(features, scenario, seed)
        p = model.predict_proba(shocked)[:, 1]
        row: dict[str, Any] = {
            "scenario": name,
            "description": scenario.get("description", ""),
            "mean_pd": float(p.mean()),
            "predicted_default_rate": float(p.mean()),
            "rejection_rate": float((p >= threshold).mean()),
            "approval_rate": float((p < threshold).mean()),
            "high_risk_share": float((p >= cutoff).mean()),
            "p95_pd": float(np.quantile(p, 0.95)),
        }
        if lgd is not None and ead is not None:
            row.update(portfolio_expected_loss(p, lgd, ead))
        rows.append(row)

    base = next(r for r in rows if r["scenario"] == "base")
    for row in rows:
        row["mean_pd_change_vs_base"] = row["mean_pd"] / base["mean_pd"] - 1
        row["rejection_rate_change_pp"] = row["rejection_rate"] - base["rejection_rate"]
    return rows
