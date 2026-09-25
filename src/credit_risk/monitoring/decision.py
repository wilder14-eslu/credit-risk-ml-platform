"""Política de disparo de Continuous Training (CT) a partir de las señales de monitoreo."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd


def retrain_decision(
    drift_table: pd.DataFrame,
    concept: dict[str, Any] | None,
    cfg: dict[str, Any],
    n_rows: int,
    last_retrain_at: datetime | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    reasons: list[str] = []

    if n_rows < cfg["min_rows"]:
        return {
            "retrain": False,
            "reasons": [f"ventana con {n_rows} filas < {cfg['min_rows']}"],
            "severity": "sin_datos",
        }

    features = drift_table[drift_table["feature"] != "prediction"] if not drift_table.empty else drift_table
    drifted = features[features["psi"] >= cfg["psi_alert"]]["feature"].tolist() if not features.empty else []
    if len(drifted) >= cfg["max_features_drifted"]:
        reasons.append(f"data drift en {len(drifted)} features: {', '.join(drifted)}")

    pred = drift_table[drift_table["feature"] == "prediction"] if not drift_table.empty else drift_table
    if not pred.empty and float(pred["psi"].iloc[0]) >= cfg["prediction_psi_alert"]:
        reasons.append(f"prediction drift PSI={float(pred['psi'].iloc[0]):.3f}")

    if concept:
        for name, fired in concept["signals"].items():
            if fired:
                reasons.append(f"concept drift: {name}")

    severity = "critico" if (concept and concept.get("concept_drift")) else "alerta" if reasons else "estable"
    retrain = bool(reasons)

    if retrain and last_retrain_at is not None:
        if now - last_retrain_at < timedelta(hours=cfg["retrain_cooldown_hours"]):
            reasons.append("cooldown activo: se registra la alerta pero no se reentrena")
            retrain = False

    return {"retrain": retrain, "reasons": reasons, "severity": severity, "drifted_features": drifted}
