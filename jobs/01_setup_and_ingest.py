"""Job 1 - Setup de Unity Catalog + ingesta Bronze.

Crea schema, volumen y tablas operativas (idempotente) y carga el CSV crudo
de Kaggle (versionado en el repo, sincronizado por el bundle) a la capa Bronze.
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

import pandas as pd  # noqa: E402

from credit_risk import lakehouse as lh  # noqa: E402
from credit_risk.data.quality import to_canonical  # noqa: E402


def main() -> None:
    lh.configure_logging()
    args = lh.job_args({"source-file": "DATA/GiveMeSomeCredit/cs-training.csv"})
    names = lh.names_from(args)
    lh.ensure_objects(names)

    path = os.path.join(_root, args.source_file)
    raw = pd.read_csv(path)
    bronze = to_canonical(raw)
    bronze["ingested_ts"] = lh.utcnow()
    bronze["source_file"] = os.path.basename(path)
    lh.write_pandas(bronze, names.bronze_applications, mode="overwrite")
    lh.set_task_value("bronze_rows", int(len(bronze)))


if __name__ == "__main__":
    main()
