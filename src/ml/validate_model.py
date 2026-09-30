"""Gate de calidad del modelo en producción: champion vs challenger.

Dos pasos separados a propósito:

1. ``evaluate_candidate``: decide si un challenger es **elegible** (mejora
   el AUC del champion). No modifica el registry.
2. ``promote_candidate``: transiciona la versión a Production y **exige
   ``approved_by``** (aprobación humana explícita). Nada en el repo llama a
   esta función automáticamente desde el monitoreo.
"""

from __future__ import annotations

from typing import Any

MODEL_NAME = "bcp_credit_champion"


def _client(client: Any = None) -> Any:
    if client is not None:
        return client
    import mlflow

    return mlflow.tracking.MlflowClient()


def evaluate_candidate(
    new_roc_auc: float, model_name: str = MODEL_NAME, client: Any = None
) -> dict[str, Any]:
    """Elegibilidad del challenger frente al champion vigente (sin efectos)."""
    client = _client(client)
    latest_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_versions:
        return {"eligible": True, "reason": "no hay champion en Production",
                "champion_roc_auc": None, "candidate_roc_auc": new_roc_auc}
    current_auc = float(
        client.get_run(latest_versions[0].run_id).data.metrics.get("roc_auc", 0.0)
    )
    eligible = new_roc_auc > current_auc
    return {
        "eligible": eligible,
        "reason": "mejora el AUC del champion" if eligible else "no mejora el AUC del champion",
        "champion_roc_auc": current_auc,
        "candidate_roc_auc": new_roc_auc,
    }


def promote_candidate(
    approved_by: str, model_name: str = MODEL_NAME, client: Any = None
) -> str:
    """Promueve la última versión registrada a Production con aprobación humana."""
    if not approved_by or not approved_by.strip():
        raise PermissionError("La promoción a Production requiere 'approved_by'.")
    client = _client(client)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise LookupError(f"No hay versiones registradas de {model_name}.")
    candidate = max(versions, key=lambda version: int(version.version))
    client.transition_model_version_stage(
        name=model_name,
        version=candidate.version,
        stage="Production",
        archive_existing_versions=True,
    )
    return str(candidate.version)
