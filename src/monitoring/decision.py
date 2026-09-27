"""Regla de decisión de reentrenamiento (pura, sin I/O).

En Databricks la ejecuta el job de monitoreo; aquí vive separada para poder
probarla de forma aislada.
"""


def decide_retrain(
    rate_triggered: bool,
    feature_drift_triggered: bool,
    performance_triggered: bool,
) -> bool:
    """Reentrenar si cualquiera de las tres señales se activa."""
    return rate_triggered or feature_drift_triggered or performance_triggered
