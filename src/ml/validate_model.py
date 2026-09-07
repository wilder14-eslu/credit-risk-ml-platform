"""Production model quality gate."""


def evaluate_and_promote(
    new_roc_auc: float,
    model_name: str = "bcp_credit_champion",
    client=None,
) -> bool:
    """Promote a candidate only when it improves over production."""
    if client is None:
        import mlflow

        client = mlflow.tracking.MlflowClient()

    latest_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_versions:
        return True

    current_run_id = latest_versions[0].run_id
    current_auc = float(
        client.get_run(current_run_id).data.metrics.get("roc_auc", 0.0)
    )
    if new_roc_auc <= current_auc:
        return False

    candidate_versions = client.search_model_versions(f"name='{model_name}'")
    if candidate_versions:
        candidate = max(candidate_versions, key=lambda version: int(version.version))
        client.transition_model_version_stage(
            name=model_name,
            version=candidate.version,
            stage="Production",
            archive_existing_versions=True,
        )
    return True
