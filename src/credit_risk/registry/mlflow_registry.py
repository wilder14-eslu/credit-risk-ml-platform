"""MLflow Tracking + Model Registry en Unity Catalog (aliases champion/challenger)."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from credit_risk.config import CHALLENGER_ALIAS, CHAMPION_ALIAS, EXPERIMENT_NAME, PROJECT_ROOT, UCNames
from credit_risk.models.credit_model import CreditRiskModel, CreditRiskPyfunc

logger = logging.getLogger(__name__)

SERVING_REQUIREMENTS = [
    "mlflow>=3.1",
    "pandas>=2.1",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "scipy>=1.11",
    "xgboost>=2.0",
    "lightgbm>=4.3",
    "catboost>=1.2",
    "pyyaml>=6.0",
    "joblib>=1.3",
]


def setup_mlflow(experiment: str = EXPERIMENT_NAME) -> None:
    import mlflow

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment)


def _flat_metrics(result: dict[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in result.items() if isinstance(v, (int, float)) and v == v}


def log_candidate(algorithm: str, result: dict[str, Any], parent_run_id: str | None = None) -> str:
    """Una corrida anidada por candidato del benchmark (P7: metadatos de ML)."""
    import mlflow

    with mlflow.start_run(run_name=f"candidate-{algorithm}", nested=parent_run_id is not None) as run:
        mlflow.set_tag("stage", "benchmark")
        mlflow.set_tag("algorithm", algorithm)
        mlflow.log_params({f"hp_{k}": v for k, v in (result.get("params") or {}).items()})
        mlflow.log_metrics(_flat_metrics(result))
        mlflow.set_tag("fit_diagnosis", result.get("fit_diagnosis", ""))
        return run.info.run_id


def log_and_register(
    model: CreditRiskModel,
    result: dict[str, Any],
    report_md: str,
    benchmark: pd.DataFrame,
    input_example: pd.DataFrame,
    names: UCNames,
    extra_tags: dict[str, str] | None = None,
) -> str:
    """Registra el modelo servible en UC y devuelve la versión creada."""
    import mlflow
    from mlflow.models import infer_signature

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        model_file = tmp_path / "credit_model.joblib"
        joblib.dump(model, model_file)
        (tmp_path / "training_report.md").write_text(report_md, encoding="utf-8")
        benchmark.drop(columns=["params"], errors="ignore").to_csv(tmp_path / "benchmark.csv", index=False)

        output_example = model.predict_frame(input_example).reset_index(drop=True)
        signature = infer_signature(input_example, output_example, params={"explain": True})

        with mlflow.start_run(run_name=f"champion-candidate-{model.algorithm}") as run:
            mlflow.set_tags({"stage": "registration", "algorithm": model.algorithm, **(extra_tags or {})})
            mlflow.log_params({"algorithm": model.algorithm, "threshold": model.threshold})
            mlflow.log_params({f"hp_{k}": v for k, v in model.metadata.get("params", {}).items()})
            mlflow.log_metrics(_flat_metrics(result))
            mlflow.log_artifact(str(tmp_path / "training_report.md"), "reports")
            mlflow.log_artifact(str(tmp_path / "benchmark.csv"), "reports")
            mlflow.log_dict(model.metadata.get("reference_metrics", {}), "reference_metrics.json")
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=CreditRiskPyfunc(),
                artifacts={"credit_model": str(model_file)},
                code_paths=[str(PROJECT_ROOT / "src" / "credit_risk")],
                signature=signature,
                input_example=input_example.head(3),
                pip_requirements=SERVING_REQUIREMENTS,
                registered_model_name=names.model_name,
            )
            version = str(info.registered_model_version)
            logger.info("Registrado %s v%s (run %s)", names.model_name, version, run.info.run_id)

    client = mlflow.MlflowClient()
    for key in ("test_roc_auc", "test_ks", "test_brier", "latency_ms", "threshold"):
        client.set_model_version_tag(names.model_name, version, key, f"{result[key]:.6f}")
    client.set_model_version_tag(names.model_name, version, "algorithm", model.algorithm)
    client.update_model_version(
        names.model_name,
        version,
        description=f"{model.algorithm} | AUC test {result['test_roc_auc']:.4f} | "
        f"umbral {model.threshold:.2f}",
    )
    return version


def get_alias_version(names: UCNames, alias: str) -> str | None:
    import mlflow

    try:
        return str(mlflow.MlflowClient().get_model_version_by_alias(names.model_name, alias).version)
    except Exception:  # alias o modelo inexistente
        return None


def set_alias(names: UCNames, alias: str, version: str) -> None:
    import mlflow

    mlflow.MlflowClient().set_registered_model_alias(names.model_name, alias, version)
    logger.info("Alias @%s -> v%s", alias, version)


def delete_alias(names: UCNames, alias: str) -> None:
    import mlflow

    try:
        mlflow.MlflowClient().delete_registered_model_alias(names.model_name, alias)
    except Exception:
        pass


def load_credit_model(names: UCNames, alias_or_version: str) -> CreditRiskModel:
    """Carga el objeto Python del modelo (para batch scoring, simulación y A/B)."""
    import mlflow

    ref = f"@{alias_or_version}" if not alias_or_version.isdigit() else f"/{alias_or_version}"
    loaded = mlflow.pyfunc.load_model(f"models:/{names.model_name}{ref}")
    return loaded.unwrap_python_model().model


def promote_challenger(names: UCNames) -> dict[str, str | None]:
    """challenger -> champion; el champion anterior queda con alias `previous_champion`."""
    challenger = get_alias_version(names, CHALLENGER_ALIAS)
    if challenger is None:
        raise RuntimeError("No hay challenger para promover")
    previous = get_alias_version(names, CHAMPION_ALIAS)
    if previous:
        set_alias(names, "previous_champion", previous)
    set_alias(names, CHAMPION_ALIAS, challenger)
    delete_alias(names, CHALLENGER_ALIAS)
    return {"champion": challenger, "previous_champion": previous}


def rollback(names: UCNames) -> str:
    previous = get_alias_version(names, "previous_champion")
    if previous is None:
        raise RuntimeError("No hay versión previa para hacer rollback")
    set_alias(names, CHAMPION_ALIAS, previous)
    return previous


def version_tags(names: UCNames, version: str) -> dict[str, str]:
    import mlflow

    return dict(mlflow.MlflowClient().get_model_version(names.model_name, version).tags or {})


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)
