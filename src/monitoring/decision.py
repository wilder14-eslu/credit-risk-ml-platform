"""Regla de decisión de reentrenamiento (pura, sin I/O).

Principio de gobierno: **la deriva nunca despliega un modelo**. Las señales
solo habilitan la apertura de un *candidato de reentrenamiento*; entrenarlo,
validarlo y promoverlo a producción requiere revisión y aprobación humana
(ver ``docs/governance/RETRAINING_POLICY.md`` y
``src.ml.validate_model.promote_candidate``).
"""

from __future__ import annotations

from typing import Any


def decide_retrain(
    rate_triggered: bool,
    feature_drift_triggered: bool,
    performance_triggered: bool,
) -> bool:
    """Reentrenar si cualquiera de las tres señales se activa."""
    return rate_triggered or feature_drift_triggered or performance_triggered


def build_retraining_recommendation(signals: dict[str, bool]) -> dict[str, Any]:
    """Convierte señales de monitoreo en una recomendación **no ejecutable sola**.

    ``auto_deploy`` es siempre ``False`` y ``requires_human_approval`` siempre
    ``True``: no existe ruta de código de deriva a producción.
    """
    active = sorted(name for name, fired in signals.items() if fired)
    return {
        "open_retraining_candidate": bool(active),
        "signals_fired": active,
        "auto_deploy": False,
        "requires_human_approval": True,
        "next_step": (
            "Entrenar un challenger, validarlo contra el champion y solicitar aprobación."
            if active else "Sin acción: continuar monitoreo."
        ),
    }
