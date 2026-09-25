"""Job 6 - Monitoreo de producción: data drift, prediction drift y concept drift.

Compara la ventana reciente del champion contra su perfil de referencia y,
con las etiquetas disponibles, contra sus métricas de entrenamiento. Publica
`retrain=true|false` como task value; la condition_task del job decide si
dispara el pipeline de Continuous Training (run_job_task).
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
import uuid  # noqa: E402

import pandas as pd  # noqa: E402

from credit_risk import lakehouse as lh  # noqa: E402
from credit_risk.config import CHAMPION_ALIAS, base_feature_names, platform_config  # noqa: E402
from credit_risk.monitoring.concept_drift import concept_drift_report  # noqa: E402
from credit_risk.monitoring.data_drift import feature_drift  # noqa: E402
from credit_risk.monitoring.decision import retrain_decision  # noqa: E402
from credit_risk.registry import mlflow_registry as R  # noqa: E402


def main() -> None:
    lh.configure_logging()
    args = lh.job_args()
    names = lh.names_from(args)
    cfg = platform_config()["monitoring"]
    R.setup_mlflow()

    version = R.get_alias_version(names, CHAMPION_ALIAS)
    if version is None:
        lh.set_task_value("retrain", "true")
        lh.logger.warning("No hay champion: se solicita entrenamiento inicial")
        return

    ref_rows = lh.read_pandas(
        f"SELECT * FROM {names.reference_profile} WHERE model_version = '{version}' "
        "ORDER BY created_ts DESC LIMIT 1"
    )
    if ref_rows.empty:
        raise RuntimeError(f"No hay perfil de referencia para v{version}")
    profile = json.loads(ref_rows.iloc[0]["profile_json"])
    reference = json.loads(ref_rows.iloc[0]["reference_metrics_json"])

    window = lh.read_pandas(f"""
        SELECT l.*, o.actual_default
        FROM {names.inference_log} l
        LEFT JOIN {names.outcomes} o ON l.request_id = o.request_id
        WHERE l.variant = 'champion' AND l.model_version = '{version}'
          AND l.event_ts >= current_timestamp() - INTERVAL {int(cfg["window_days"])} DAYS
        ORDER BY l.event_ts
    """)

    current = window[list(base_feature_names())].copy()
    current["__score__"] = window["probability"]
    drift = feature_drift(profile, current, cfg["psi_warning"], cfg["psi_alert"], cfg["ks_pvalue_alert"])

    labeled = window.dropna(subset=["actual_default"])
    concept = None
    if len(labeled) >= cfg["min_rows"] and labeled["actual_default"].nunique() == 2:
        tags = R.version_tags(names, version)
        concept = concept_drift_report(
            labeled["actual_default"].to_numpy(),
            labeled["probability"].to_numpy(),
            reference,
            cfg,
            threshold=float(tags.get("threshold", 0.5)),
        )

    last = lh.read_pandas(f"SELECT max(event_ts) AS ts FROM {names.retrain_events}")
    last_ts = last["ts"].iloc[0] if not last.empty else None
    last_ts = None if pd.isna(last_ts) else pd.Timestamp(last_ts).tz_localize("UTC").to_pydatetime()
    decision = retrain_decision(drift, concept, cfg, len(window), last_ts)

    run_id, now = str(uuid.uuid4()), lh.utcnow()
    if not drift.empty:
        drift_out = drift.assign(run_id=run_id, run_ts=now, model_version=version)
        lh.write_pandas(drift_out, names.drift_features)
    pred_psi = drift.loc[drift["feature"] == "prediction", "psi"]
    cur = concept["current"] if concept else {}
    metrics_row = {
        "run_id": run_id,
        "run_ts": now,
        "model_version": version,
        "window_days": int(cfg["window_days"]),
        "n_rows": int(len(window)),
        "n_labeled": int(len(labeled)),
        "features_drifted": len(decision.get("drifted_features", [])),
        "prediction_psi": float(pred_psi.iloc[0]) if len(pred_psi) else None,
        "roc_auc": cur.get("roc_auc"),
        "ks": cur.get("ks"),
        "brier": cur.get("brier"),
        "auc_drop": concept["auc_drop"] if concept else None,
        "default_rate_observed": cur.get("default_rate_observed"),
        "default_rate_predicted": cur.get("default_rate_predicted"),
        "ddm_state": concept["ddm_state"] if concept else None,
        "page_hinkley_statistic": concept["page_hinkley_statistic"] if concept else None,
        "concept_drift": bool(concept["concept_drift"]) if concept else False,
        "severity": decision["severity"],
        "retrain": decision["retrain"],
        "reasons": json.dumps(decision["reasons"], ensure_ascii=False),
    }
    lh.write_pandas(pd.DataFrame([metrics_row]), names.monitoring_metrics)

    if decision["retrain"]:
        lh.write_pandas(
            pd.DataFrame(
                [
                    {
                        "event_ts": now,
                        "trigger": "monitoring",
                        "reasons": json.dumps(decision["reasons"], ensure_ascii=False),
                        "model_version": version,
                    }
                ]
            ),
            names.retrain_events,
        )
    lh.logger.info("Decisión de monitoreo: %s", decision)
    lh.set_task_value("retrain", "true" if decision["retrain"] else "false")
    lh.set_task_value("severity", decision["severity"])


if __name__ == "__main__":
    main()
