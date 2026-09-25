"""Job 8 (manual) - Rollback: @champion vuelve a @previous_champion y se redepliega."""

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
from credit_risk.registry import mlflow_registry as R  # noqa: E402
from credit_risk.registry import serving_endpoint as S  # noqa: E402


def main() -> None:
    lh.configure_logging()
    args = lh.job_args({"deploy-serving": "true"})
    names = lh.names_from(args)
    R.setup_mlflow()
    version = R.rollback(names)
    R.delete_alias(names, "challenger")
    if args.deploy_serving.lower() == "true":
        S.deploy(names, version, None)
    lh.logger.info("Rollback completado: champion = v%s", version)


if __name__ == "__main__":
    main()
