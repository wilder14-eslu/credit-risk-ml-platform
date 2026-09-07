"""Panel de monitoreo: drift de datos, performance en vivo y estado del
reentrenamiento automático.

Lee directamente los mismos artefactos que usan la API y el flow de
monitoreo (`src.orchestrator.monitor`): el log de predicciones/resultados
(`data/processed/*.jsonl`) y la distribución de referencia que guarda cada
entrenamiento (`data/processed/reference_distribution.json`). No llama a la
API por HTTP, igual que `app.py`, para que la demo funcione con un solo
proceso.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.api.monitoring import (
    compute_live_performance,
    drift_trigger,
    load_recent_predictions,
    record_outcome,
)
from src.monitoring.drift import compute_feature_drift, load_reference_distribution
from src.orchestrator.monitor import decide_retrain

st.set_page_config(
    page_title="Monitoreo · Credit Risk ML Platform", page_icon="📊", layout="centered"
)
st.title("📊 Monitoreo del modelo")
st.caption(
    "Estado de drift de datos, performance en producción y la señal de "
    "reentrenamiento que consume `src.orchestrator.monitor.monitoring_flow` "
    "(el mismo chequeo que corre solo, cada hora por defecto, en el "
    "servicio `scheduler` del docker-compose)."
)

predictions = load_recent_predictions()

col1, col2, col3 = st.columns(3)
col1.metric("Predicciones registradas", len(predictions))

probabilities = [event["probability"] for event in predictions]
rate_triggered = drift_trigger(probabilities) if probabilities else False
col2.metric("Tasa de rechazo anómala", "Sí" if rate_triggered else "No")

st.subheader("Drift de datos (PSI por feature)")
try:
    reference = load_reference_distribution()
except FileNotFoundError:
    st.info(
        "Aún no hay una distribución de referencia guardada: entrena un "
        "modelo primero con `python -m src.ml.train` (o `make train`)."
    )
    feature_drift = {
        "psi": {},
        "drift_detected": False,
        "drifted_features": [],
        "threshold": 0.2,
    }
else:
    recent_features = (
        pd.DataFrame([event["features"] for event in predictions])
        if predictions
        else pd.DataFrame()
    )
    feature_drift = compute_feature_drift(recent_features, reference)
    if feature_drift["psi"]:
        psi_df = pd.DataFrame(
            {
                "feature": list(feature_drift["psi"].keys()),
                "psi": list(feature_drift["psi"].values()),
            }
        ).set_index("feature")
        st.bar_chart(psi_df)
        drifted = ", ".join(feature_drift["drifted_features"]) or "ninguna"
        st.caption(
            f"Umbral de alerta: PSI > {feature_drift['threshold']}. "
            f"Features con drift: {drifted}."
        )
    else:
        st.caption("Todavía no hay suficientes predicciones para calcular PSI.")

col3.metric("Drift de features", "Sí" if feature_drift["drift_detected"] else "No")

st.subheader("Performance en vivo (requiere resultados reales)")
performance = compute_live_performance()
if performance["status"] == "ok":
    st.metric(
        "ROC-AUC en producción",
        f"{performance['live_roc_auc']:.3f}",
        help=f"Calculado sobre {performance['n_matched']} resultados confirmados.",
    )
else:
    st.info(
        f"Aún no hay suficientes resultados confirmados "
        f"({performance['n_matched']}/{performance['min_samples']}) para calcular "
        "performance en vivo. Reporta resultados reales abajo."
    )

should_retrain = decide_retrain(rate_triggered, feature_drift["drift_detected"], False)
if should_retrain:
    st.error(
        "⚠️ Al menos una señal de alerta está activa: el reentrenamiento "
        "automático se disparará en el próximo chequeo programado."
    )
else:
    st.success("✅ Sin señales de alerta activas.")

st.divider()
st.subheader("Reportar un resultado real")
st.caption(
    "Simula lo que en un banco haría el sistema de cobranzas meses después: "
    "confirma si un solicitante ya evaluado terminó incumpliendo o no, para "
    "alimentar el cálculo de performance en vivo."
)
with st.form("report_outcome"):
    applicant_id = st.text_input("Identificador del solicitante (el mismo usado en /predict)")
    actual_default = st.selectbox("¿Incumplió?", ["No", "Sí"]) == "Sí"
    submitted = st.form_submit_button("Registrar resultado")

if submitted:
    if not applicant_id:
        st.warning("Ingresa el identificador del solicitante.")
    else:
        record_outcome(applicant_id, actual_default)
        st.success(f"Resultado registrado para «{applicant_id}».")
