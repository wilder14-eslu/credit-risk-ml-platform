"""Job 4 - Validación offline champion vs challenger + despliegue en Model Serving.

- Sin champion: el challenger se promueve directo (primer despliegue).
- Con champion: ambos se evalúan en el MISMO test set. Si el challenger mejora
  el AUC en `min_auc_improvement`, pasa a prueba online (A/B) con el
  `challenger_traffic` configurado; si no, se descarta.
- Finalmente se actualiza el endpoint de Databricks Model Serving.
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

from credit_risk import lakehouse as lh  # noqa: E402
from credit_risk.config import CHALLENGER_ALIAS, CHAMPION_ALIAS  # noqa: E402
from credit_risk.models import metrics as M  # noqa: E402
from credit_risk.models import training as T  # noqa: E402
from credit_risk.registry import mlflow_registry as R  # noqa: E402
from credit_risk.registry import serving_endpoint as S  # noqa: E402


def main() -> None:
    lh.configure_logging()
    args = lh.job_args({"deploy-serving": "true"})
    names = lh.names_from(args)
    R.setup_mlflow()

    challenger_v = R.get_alias_version(names, CHALLENGER_ALIAS)
    champion_v = R.get_alias_version(names, CHAMPION_ALIAS)
    if challenger_v is None and champion_v is None:
        raise RuntimeError("No hay modelos registrados todavía: corre el job de entrenamiento")

    if challenger_v and champion_v is None:
        R.promote_challenger(names)
        champion_v, challenger_v = challenger_v, None
        lh.logger.info("Primer despliegue: v%s es champion", champion_v)
    elif challenger_v and champion_v:
        splits = T.make_splits(lh.read_pandas(names.feature_table))
        auc = {}
        for alias in (CHAMPION_ALIAS, CHALLENGER_ALIAS):
            model = R.load_credit_model(names, alias)
            auc[alias] = M.classification_metrics(splits.y_test, model.predict_proba(splits.x_test))[
                "roc_auc"
            ]
        ok, reason = T.champion_vs_challenger(auc[CHALLENGER_ALIAS], auc[CHAMPION_ALIAS])
        lh.logger.info("Validación offline: %s (%s)", ok, reason)
        if not ok:
            R.delete_alias(names, CHALLENGER_ALIAS)
            challenger_v = None
        lh.set_task_value("offline_validation", reason)

    if args.deploy_serving.lower() == "true":
        spec = S.deploy(names, champion_v, challenger_v)
        lh.logger.info("Endpoint actualizado: %s", spec)
    lh.set_task_value("champion_version", champion_v)
    lh.set_task_value("challenger_version", challenger_v or "")


if __name__ == "__main__":
    main()
