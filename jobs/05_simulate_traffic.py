"""Job 5 - Simulación de tráfico de producción + llegada retrasada de etiquetas.

Muestrea solicitantes del production pool, aplica un escenario de drift
(`none | covariate | concept | prior | mixed`), asigna variante A/B con el
mismo hash que usa el gateway, puntúa con champion/challenger y escribe en
`inference_log`. Una fracción de las etiquetas reales llega a `outcomes`.
En un banco real este job no existe: lo reemplazan el gateway y el core bancario.
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

import uuid  # noqa: E402
from datetime import timedelta  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk import lakehouse as lh  # noqa: E402
from credit_risk.config import (  # noqa: E402
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    base_feature_names,
    platform_config,
    target_name,
)
from credit_risk.data.simulation import apply_scenario  # noqa: E402
from credit_risk.monitoring.ab_testing import assign_variant  # noqa: E402
from credit_risk.registry import mlflow_registry as R  # noqa: E402


def main() -> None:
    lh.configure_logging()
    args = lh.job_args(
        {"scenario": "none", "batch-size": "2000", "label-fraction": "0.8", "intensity": "1.0", "seed": None}
    )
    names = lh.names_from(args)
    R.setup_mlflow()
    seed = int(args.seed) if args.seed not in (None, "") else int(lh.utcnow().timestamp())

    pool = lh.read_pandas(names.production_pool)
    batch = pool.sample(n=min(int(args.batch_size), len(pool)), random_state=seed).reset_index(drop=True)
    batch = apply_scenario(batch, args.scenario, float(args.intensity), seed)
    # Cada simulación es una nueva solicitud: id único para no mezclar corridas.
    batch["applicant_id"] = [f"{a}-{seed}" for a in batch["applicant_id"]]

    champion_v = R.get_alias_version(names, CHAMPION_ALIAS)
    challenger_v = R.get_alias_version(names, CHALLENGER_ALIAS)
    if champion_v is None:
        raise RuntimeError("No hay champion desplegado")
    models = {"champion": (champion_v, R.load_credit_model(names, CHAMPION_ALIAS))}
    traffic = 0.0
    if challenger_v:
        models["challenger"] = (challenger_v, R.load_credit_model(names, CHALLENGER_ALIAS))
        traffic = platform_config()["ab_testing"]["challenger_traffic"]

    batch["variant"] = batch["applicant_id"].map(lambda a: assign_variant(a, traffic))
    now = lh.utcnow()
    rng = np.random.default_rng(seed)
    logs = []
    for variant, group in batch.groupby("variant"):
        version, model = models[variant]
        scored = model.predict_frame(group, explain=False)
        frame = group[["applicant_id", *base_feature_names()]].copy()
        frame["request_id"] = [str(uuid.uuid4()) for _ in range(len(frame))]
        frame["event_ts"] = [now - timedelta(minutes=int(m)) for m in rng.integers(0, 24 * 60, len(frame))]
        frame["source"] = "simulator"
        frame["variant"] = variant
        frame["model_version"] = version
        frame["probability"] = scored["probability"].to_numpy()
        frame["decision"] = scored["decision"].to_numpy()
        frame["risk_band"] = scored["risk_band"].to_numpy()
        frame["latency_ms"] = np.nan
        frame["scenario"] = args.scenario
        frame[target_name()] = group[target_name()].to_numpy()
        logs.append(frame)
    log = pd.concat(logs, ignore_index=True)
    lh.write_pandas(log.drop(columns=[target_name()]), names.inference_log)

    labeled = log.sample(frac=float(args.label_fraction), random_state=seed)
    outcomes = pd.DataFrame(
        {
            "applicant_id": labeled["applicant_id"],
            "request_id": labeled["request_id"],
            "actual_default": labeled[target_name()].astype(int),
            "observed_ts": now,
            "source": "simulator",
        }
    )
    lh.write_pandas(outcomes, names.outcomes)
    lh.set_task_value("simulated_rows", int(len(log)))


if __name__ == "__main__":
    main()
