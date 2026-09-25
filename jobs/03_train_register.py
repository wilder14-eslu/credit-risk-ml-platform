"""Job 3 - Benchmark + tuning (Optuna) + registro en Unity Catalog como challenger.

1. Benchmark de 4 algoritmos sobre el mismo split (corridas anidadas en MLflow).
2. Tuning con Optuna del mejor algoritmo (validación, nunca test).
3. Gate absoluto de calidad; si falla, el job termina sin registrar.
4. Registro del modelo pyfunc (features + imputación + estimador) en UC y
   alias `@challenger`; se guarda el perfil de referencia para drift.
"""

# Bootstrap: agrega <raíz del bundle>/src al path (el bundle pasa --project-root).
import os
import sys

_root = next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == "--project-root"), None)
if _root is None:
    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        _root = os.getcwd()
sys.path.insert(0, os.path.join(_root, "src"))
os.environ.setdefault("CREDIT_RISK_CONFIG_DIR", os.path.join(_root, "config"))

import json  # noqa: E402

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk import lakehouse as lh  # noqa: E402
from credit_risk.config import CHALLENGER_ALIAS, base_feature_names, platform_config  # noqa: E402
from credit_risk.features.engineering import build_features  # noqa: E402
from credit_risk.models import training as T  # noqa: E402
from credit_risk.monitoring.data_drift import build_reference_profile  # noqa: E402
from credit_risk.registry import mlflow_registry as R  # noqa: E402


def main() -> None:
    lh.configure_logging()
    args = lh.job_args({"optuna-trials": None, "trigger": "manual", "git-sha": "local"})
    names = lh.names_from(args)
    tcfg = platform_config()["training"]
    trials = int(args.optuna_trials) if args.optuna_trials not in (None, "") else tcfg["optuna_trials"]

    R.setup_mlflow()
    data = lh.read_pandas(names.feature_table)
    splits = T.make_splits(data)

    with mlflow.start_run(run_name=f"retraining-{args.trigger}") as parent:
        mlflow.set_tags({"trigger": args.trigger, "git_sha": args.git_sha, "stage": "pipeline"})
        mlflow.log_param("train_rows", len(splits.x_train))
        table, _ = T.run_benchmark(splits)
        for _, row in table.iterrows():
            R.log_candidate(row["algorithm"], row.to_dict(), parent_run_id=parent.info.run_id)

        best = T.select_best(table)
        params = T.tune(best, splits, trials, tcfg["optuna_timeout_seconds"]) if trials > 0 else {}
        model = T.fit_model(best, params, splits)
        result = T.evaluate(model, splits)
        mlflow.log_metrics({f"best_{k}": v for k, v in result.items() if isinstance(v, float)})

    passed, reasons = T.check_quality_gates(result)
    bench = table.drop(columns=["params"]).copy()
    bench["run_ts"] = lh.utcnow()
    bench["selected"] = bench["algorithm"] == best
    bench["registered_version"] = None
    if not passed:
        lh.write_pandas(bench, names.model_benchmark)
        lh.set_task_value("gate_passed", "false")
        raise RuntimeError(f"Quality gate fallido, no se registra el modelo: {reasons}")

    reference = {k.replace("test_", ""): v for k, v in result.items() if k.startswith("test_")}
    model.metadata["reference_metrics"] = reference
    importance = model.global_importance(splits.x_test.sample(min(2000, len(splits.x_test)), random_state=0))
    report = T.training_report(table, best, result, importance)

    version = R.log_and_register(
        model,
        result,
        report,
        table,
        splits.x_test.head(5).reset_index(drop=True),
        names,
        extra_tags={"trigger": args.trigger, "git_sha": args.git_sha},
    )
    R.set_alias(names, CHALLENGER_ALIAS, version)

    bench["registered_version"] = bench["selected"].map(lambda s: version if s else None)
    lh.write_pandas(bench, names.model_benchmark)

    x_ref = build_features(splits.x_train)[list(base_feature_names())]
    profile = build_reference_profile(x_ref, model.predict_proba(splits.x_train))
    lh.write_pandas(
        pd.DataFrame(
            [
                {
                    "model_version": version,
                    "created_ts": lh.utcnow(),
                    "profile_json": json.dumps(profile),
                    "reference_metrics_json": json.dumps(reference),
                }
            ]
        ),
        names.reference_profile,
    )
    os.makedirs(f"{names.volume_path}/reports", exist_ok=True)
    with open(f"{names.volume_path}/reports/training_report_v{version}.md", "w", encoding="utf-8") as fh:
        fh.write(report)

    lh.set_task_value("gate_passed", "true")
    lh.set_task_value("challenger_version", version)


if __name__ == "__main__":
    main()
