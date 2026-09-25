"""Demo y panel de monitoreo (Streamlit en Render).

La app no carga modelos: habla solo con el gateway (API_URL), que a su vez usa
Databricks Model Serving y lee las tablas Delta de monitoreo.
"""

from __future__ import annotations

import json
import os

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
if not API_URL.startswith("http"):  # Render inyecta solo el host del gateway
    API_URL = f"https://{API_URL}"
API_KEY = os.getenv("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# Paleta validada (dataviz): categorías fijas por entidad y colores de estado reservados.
SERIES = {"champion": "#2a78d6", "challenger": "#eb6834"}
STATUS = {"estable": "#0ca30c", "warning": "#fab219", "alerta": "#d03b3b"}
STATUS_ICON = {"estable": "✅", "warning": "⚠️", "alerta": "🛑"}

st.set_page_config(page_title="Credit Risk ML Platform", page_icon="💳", layout="wide")


@st.cache_data(ttl=300)
def get(path: str) -> dict:
    response = httpx.get(f"{API_URL}{path}", headers=HEADERS, timeout=90)
    response.raise_for_status()
    return response.json()


def post(path: str, payload: dict) -> dict:
    response = httpx.post(f"{API_URL}{path}", json=payload, headers=HEADERS, timeout=90)
    response.raise_for_status()
    return response.json()


def page_scoring() -> None:
    st.header("Evaluación de riesgo crediticio")
    st.caption(
        "El score lo calcula el modelo servido en Databricks Model Serving (champion o challenger según A/B)."
    )
    try:
        schema = get("/api/v1/features")["features"]
    except Exception as exc:
        st.error(f"No se pudo contactar al gateway en {API_URL}: {exc}")
        return

    defaults = {
        "revolving_utilization_unsecured": 0.35,
        "age": 42.0,
        "number_of_time_30_59_days_past_due": 0.0,
        "debt_ratio": 0.30,
        "monthly_income": 5200.0,
        "number_open_credit_lines": 8.0,
        "number_of_times_90_days_late": 0.0,
        "number_real_estate_loans": 1.0,
        "number_of_time_60_89_days_past_due": 0.0,
        "number_dependents": 1.0,
    }
    with st.form("scoring"):
        applicant_id = st.text_input(
            "Id del solicitante (define la variante A/B de forma estable)", "APP-DEMO-001"
        )
        cols = st.columns(2)
        values = {}
        for i, (name, meta) in enumerate(schema.items()):
            with cols[i % 2]:
                values[name] = st.number_input(
                    meta.get("label", name),
                    min_value=float(meta.get("min", 0.0)),
                    value=float(defaults.get(name, 0.0)),
                    help=meta.get("description"),
                )
        submitted = st.form_submit_button("Evaluar solicitud", type="primary")
    if not submitted:
        return
    try:
        result = post("/api/v1/predict", {"applicant_id": applicant_id, "features": values})
    except Exception as exc:
        st.error(f"Error del gateway: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probabilidad de default", f"{result['probability']:.1%}")
    c2.metric("Decisión", result["decision"])
    c3.metric("Banda de riesgo", result["risk_band"])
    c4.metric("Variante / versión", f"{result['variant']} · v{result['model_version']}")
    st.caption(f"request_id `{result['request_id']}` · latencia {result['latency_ms']:.0f} ms")

    factors = pd.DataFrame(result["top_factors"])
    if not factors.empty:
        factors["color"] = factors["direction"].map(
            {"aumenta_riesgo": STATUS["alerta"], "reduce_riesgo": STATUS["estable"]}
        )
        fig = go.Figure(
            go.Bar(
                x=factors["impact"],
                y=factors["label"],
                orientation="h",
                marker_color=factors["color"],
                hovertemplate="%{y}: %{x:.3f} log-odds<extra></extra>",
            )
        )
        fig.update_layout(
            title="Factores que más influyeron (SHAP)",
            height=260,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="impacto en log-odds",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(factors[["label", "impact", "direction"]], hide_index=True)

    with st.expander("Registrar el resultado real (feedback loop)"):
        default = st.radio("¿El cliente incumplió?", ["No", "Sí"], horizontal=True)
        if st.button("Enviar outcome"):
            post(
                "/api/v1/outcomes",
                {
                    "request_id": result["request_id"],
                    "applicant_id": applicant_id,
                    "actual_default": default == "Sí",
                },
            )
            st.success("Outcome registrado en Delta Lake")


def page_monitoring() -> None:
    st.header("Monitoreo de producción")
    try:
        summary = get("/api/v1/monitoring/summary")
        models = get("/api/v1/models")
    except Exception as exc:
        st.error(f"No se pudo leer el monitoreo: {exc}")
        return
    if not summary.get("available"):
        st.info(summary.get("detail", "Monitoreo no disponible"))
        return

    served = models.get("served", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Champion", f"v{served.get('champion', '-')}")
    c2.metric("Challenger", f"v{served['challenger']}" if "challenger" in served else "sin A/B activo")
    c3.metric("Tráfico al challenger", f"{models.get('challenger_traffic', 0):.0%}")

    mon = pd.DataFrame(summary.get("monitoring", []))
    if not mon.empty:
        latest = mon.iloc[0]
        sev = str(latest.get("severity"))
        st.subheader(f"{STATUS_ICON.get('alerta' if sev == 'critico' else sev, 'ℹ️')} Último chequeo: {sev}")
        reasons = json.loads(latest.get("reasons") or "[]")
        for r in reasons:
            st.write(f"- {r}")
        mon["run_ts"] = pd.to_datetime(mon["run_ts"])
        for col in ("roc_auc", "prediction_psi", "default_rate_observed", "default_rate_predicted"):
            mon[col] = pd.to_numeric(mon[col], errors="coerce")
        fig = px.line(
            mon.sort_values("run_ts"),
            x="run_ts",
            y="roc_auc",
            markers=True,
            title="AUC en producción (etiquetas reales)",
        )
        fig.update_traces(line=dict(width=2, color=SERIES["champion"]), marker=dict(size=8))
        fig.update_layout(hovermode="x unified", yaxis_title="ROC-AUC", xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    drift = pd.DataFrame(summary.get("drift", []))
    if not drift.empty:
        drift["psi"] = pd.to_numeric(drift["psi"], errors="coerce")
        drift = drift.sort_values("psi")
        fig = go.Figure(
            go.Bar(
                x=drift["psi"],
                y=drift["feature"],
                orientation="h",
                marker_color=drift["status"].map(STATUS).fillna(STATUS["estable"]),
                customdata=drift["status"],
                hovertemplate="%{y}: PSI %{x:.3f} (%{customdata})<extra></extra>",
            )
        )
        fig.add_vline(x=0.10, line_dash="dot", annotation_text="warning 0.10")
        fig.add_vline(x=0.25, line_dash="dot", annotation_text="alerta 0.25")
        fig.update_layout(title="Data drift por feature (PSI vs entrenamiento)", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(drift[["feature", "psi", "ks_pvalue", "status"]], hide_index=True)

    st.subheader("A/B testing champion vs challenger")
    ab = pd.DataFrame(summary.get("ab_tests", []))
    st.dataframe(ab, hide_index=True) if not ab.empty else st.caption("Sin evaluaciones A/B todavía.")

    st.subheader("Último benchmark de modelos")
    bench = pd.DataFrame(summary.get("benchmark", []))
    st.dataframe(bench, hide_index=True) if not bench.empty else st.caption("Sin benchmark registrado.")

    st.subheader("Reentrenamientos disparados")
    ret = pd.DataFrame(summary.get("retrains", []))
    st.dataframe(ret, hide_index=True) if not ret.empty else st.caption("Ninguno todavía.")


def page_architecture() -> None:
    st.header("Arquitectura")
    st.markdown(
        """
- **Databricks Free Edition**: Delta Lake (Bronze/Silver/Gold), feature table en Unity Catalog,
  MLflow (tracking + registry con aliases `@champion`/`@challenger`), jobs serverless de
  Continuous Training, monitoreo y A/B, y **Model Serving** (API REST del modelo).
- **Render**: este panel (Streamlit) y el gateway FastAPI (auth, A/B sticky, logging a Delta).
- **GitHub Actions**: CI (lint, tests, validación del bundle) y CD (`databricks bundle deploy`
  a dev y prod + deploy hooks de Render).
        """
    )


PAGES = {"Scoring": page_scoring, "Monitoreo": page_monitoring, "Arquitectura": page_architecture}
choice = st.sidebar.radio("Navegación", list(PAGES))
st.sidebar.caption(f"Gateway: {API_URL}")
PAGES[choice]()
