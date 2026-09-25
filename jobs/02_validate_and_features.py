"""Job 2 - Validación de calidad (Silver) y Feature Table (Gold) en Unity Catalog.

- Expectativas de calidad registradas en `data_quality_log`; un error bloquea el pipeline.
- Silver: filas limpias en formato canónico.
- Gold: feature table con clave primaria `applicant_id` (feature store offline).
- Production pool: 30% de solicitantes separados por hash, nunca usados para
  entrenar; alimentan la simulación de tráfico y el monitoreo.
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

import hashlib  # noqa: E402

import pandas as pd  # noqa: E402

from credit_risk import lakehouse as lh  # noqa: E402
from credit_risk.config import platform_config, target_name  # noqa: E402
from credit_risk.data.quality import assert_quality, clean, run_expectations  # noqa: E402
from credit_risk.features.engineering import build_features  # noqa: E402


def in_pool(applicant_id: str, fraction: float) -> bool:
    bucket = int(hashlib.md5(applicant_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < fraction


def main() -> None:
    lh.configure_logging()
    args = lh.job_args()
    names = lh.names_from(args)

    bronze = lh.read_pandas(names.bronze_applications)
    results = run_expectations(bronze, require_target=True)
    log = pd.DataFrame([r.to_dict() for r in results]).rename(columns={"column": "column_name"})
    log["run_ts"] = lh.utcnow()
    log["layer"] = "bronze->silver"
    lh.write_pandas(log, names.data_quality_log)
    assert_quality(results)

    silver = clean(bronze.drop(columns=["ingested_ts", "source_file"], errors="ignore"))
    silver[target_name()] = silver[target_name()].astype(int)
    lh.write_pandas(silver, names.silver_applications, mode="overwrite")

    derived = build_features(silver)
    gold = pd.concat([silver[["applicant_id"]], derived, silver[[target_name()]]], axis=1).reset_index(
        drop=True
    )
    gold["feature_ts"] = lh.utcnow()

    fraction = platform_config()["data"]["production_holdout"]
    pool_mask = gold["applicant_id"].map(lambda a: in_pool(a, fraction))
    lh.write_pandas(gold.loc[~pool_mask], names.feature_table, mode="overwrite")
    lh.write_pandas(gold.loc[pool_mask], names.production_pool, mode="overwrite")

    s = lh.spark()
    for table in (names.feature_table, names.production_pool):
        s.sql(f"ALTER TABLE {table} ALTER COLUMN applicant_id SET NOT NULL")
        try:
            s.sql(f"ALTER TABLE {table} ADD CONSTRAINT {table.split('.')[-1]}_pk PRIMARY KEY (applicant_id)")
        except Exception as exc:  # la constraint ya existe en re-ejecuciones
            lh.logger.info("PK ya definida en %s: %s", table, exc)
        s.sql(f"COMMENT ON TABLE {table} IS 'Feature table de riesgo crediticio (PK applicant_id)'")
    lh.set_task_value("train_rows", int((~pool_mask).sum()))


if __name__ == "__main__":
    main()
