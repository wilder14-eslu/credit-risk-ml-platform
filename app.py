"""Streamlit demo for the Credit Risk ML Platform.

Run with:
    streamlit run app.py

It calls the same prediction/explanation logic the FastAPI service exposes
under /api/v1/predict (`src.ml.predict.predict_with_explanation`), so the
numbers shown here match the API's output. Each input field is labeled and
annotated with the exact description from `config/data_schema.yaml`, so it
is always explicit what information is needed to get an evaluation.

On a fresh deployment (e.g. Streamlit Community Cloud) there is no
pre-trained model artifact committed to the repo (it is gitignored on
purpose: models do not belong in git history). `_ensure_model()` below
trains one automatically, once, from the dataset that *is* committed
(`DATA/GiveMeSomeCredit/`), and caches it for the life of the container so
this file works as a true "clone and run" demo.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.feature_store.features import load_feature_schema
from src.ml.predict import (
    DEFAULT_MODEL_PATH,
    ModelNotAvailableError,
    predict_with_explanation,
)

# Configuración inicial de la página con mejor layout
st.set_page_config(
    page_title="Credit Risk ML Platform", 
    page_icon="💳", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Inyección de CSS personalizado (Fondo, colores y animaciones)
st.markdown(
    """
    <style>
    /* Fondo con gradiente moderno para la aplicación principal */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
    }
    
    /* Estilo del panel lateral (sidebar) */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
    
    /* Animación para el menú de navegación (app y monitoreo) */
    [data-testid="stSidebarNav"] a {
        transition: all 0.3s ease-in-out !important;
        border-radius: 8px !important;
        margin: 0px 8px !important;
    }
    [data-testid="stSidebarNav"] a:hover {
        transform: translateX(8px);
        background-color: #e2e8f0 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    [data-testid="stSidebarNav"] a:active {
        transform: scale(0.95);
    }
    
    /* Estilo tipo 'tarjeta' para el formulario */
    [data-testid="stForm"] {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    /* Colores para los títulos */
    h1, h2, h3, h4 {
        color: #102a43;
    }
    
    /* Animación suave para el botón principal al pasar el mouse */
    [data-testid="baseButton-primary"] {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    [data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)



@st.cache_resource(show_spinner=False)
def _ensure_model() -> str:
    """Train the model on first run if no artifact exists yet, and return its path."""
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.is_file():
        # Animación moderna de carga con checklist (Status)
        with st.status("🛠️ **Configurando la plataforma (Primera Ejecución)**", expanded=True) as status:
            import time
            from src.data_pipeline.ingest import download_give_me_some_credit
            from src.data_pipeline.validate import (
                clean_out_of_range_rows,
                validate_input_data,
            )
            from src.ml.train import train_model

            st.write("📥 Descargando y preparando el dataset...")
            time.sleep(0.5) # Pequeña pausa visual
            data_dir = download_give_me_some_credit()
            csv_files = sorted(Path(data_dir).glob("cs-training.csv")) or sorted(
                Path(data_dir).glob("*.csv")
            )
            
            st.write("🧹 Limpiando y validando datos crudos...")
            time.sleep(0.5)
            raw_data = pd.read_csv(csv_files[0], index_col=0)
            raw_data = clean_out_of_range_rows(raw_data)
            validate_input_data(raw_data, require_target=True)
            
            st.write("🧠 Entrenando el modelo de Machine Learning (puede tomar unos segundos)...")
            train_model(raw_data, model_path=model_path, register=False)
            
            st.write("💾 Guardando el modelo para uso futuro...")
            time.sleep(0.5)
            
            status.update(label="¡Modelo entrenado exitosamente! ✅", state="complete", expanded=False)
            
    return str(model_path)


MODEL_PATH = _ensure_model()
schema = load_feature_schema()
fields = schema["features"]

# Inicializar historial de sesión si no existe
if "history" not in st.session_state:
    st.session_state.history = []

# --- PANEL LATERAL (SIDEBAR) ---
with st.sidebar:
    st.title("ℹ️ Acerca de la App")
    st.info(
        "Esta aplicación evalúa el riesgo crediticio de un solicitante utilizando un modelo de "
        "Machine Learning entrenado con el dataset **Give Me Some Credit**."
    )
    
    st.markdown("---")
    st.subheader("📖 Diccionario de Datos")
    st.caption("¿Qué significa cada campo del formulario?")
    
    with st.expander("Ver descripciones de los campos", expanded=False):
        for name, definition in fields.items():
            st.markdown(f"- **{definition.get('label', name)}**: {definition.get('description', '')}")
            
    st.markdown("---")
    st.link_button("📄 Ver Documentación API (Swagger)", "http://127.0.0.1:8000/docs", use_container_width=True)

    st.markdown("---")
    st.subheader("🕒 Historial de la Sesión")
    if not st.session_state.history:
        st.caption("No hay predicciones recientes.")
    else:
        for item in st.session_state.history[:5]:
            icon_hist = "✅" if item["band"] == "bajo" else "⚠️" if item["band"] == "medio" else "🚨"
            st.markdown(f"**{item['id']}**<br>{icon_hist} {item['probability']:.1%} - {item['decision'].upper()}", unsafe_allow_html=True)
        if st.button("Limpiar historial"):
            st.session_state.history = []
            st.rerun()


# --- INTERFAZ PRINCIPAL ---
st.title("💳 Evaluación de Riesgo Crediticio")
st.markdown(
    "Completa los datos del solicitante a continuación. El modelo evaluará la "
    "**probabilidad de impago (default)** y sugerirá si el crédito debe ser aprobado o rechazado."
)
st.divider()

with st.form("credit_application"):
    st.subheader("📝 Formulario de Solicitud")
    
    applicant_id = st.text_input(
        "Identificador del solicitante (opcional)", 
        placeholder="Ej. Nombre, DNI, Pasaporte..."
    )
    
    inputs: dict[str, float] = {}
    
    # Valores por defecto para que la demo sea más rápida
    defaults = {
        "age": 40.0,
        "monthly_income": 5000.0,
        "number_open_credit_lines": 4.0,
        "debt_ratio": 0.3,
        "revolving_utilization_unsecured": 0.2,
        "number_dependents": 1.0,
    }
    
    def get_input(field_name: str, col):
        """Helper para renderizar inputs dinámicamente y guardarlos en el dict `inputs`"""
        definition = fields[field_name]
        min_value = float(definition.get("min", 0.0))
        max_value = definition.get("max")
        default_value = max(defaults.get(field_name, min_value), min_value)
        with col:
            inputs[field_name] = st.number_input(
                definition.get("label", field_name),
                min_value=min_value,
                max_value=float(max_value) if max_value is not None else None,
                value=float(default_value),
                help=definition.get("description", ""),
            )

    # 1. Datos Personales
    st.markdown("#### 👤 Datos Personales")
    col1, col2 = st.columns(2)
    get_input("age", col1)
    get_input("number_dependents", col2)
    
    # 2. Salud Financiera
    st.markdown("#### 💰 Salud Financiera")
    col3, col4 = st.columns(2)
    get_input("monthly_income", col3)
    get_input("debt_ratio", col4)
    
    col5, col6, col7 = st.columns(3)
    get_input("number_open_credit_lines", col5)
    get_input("number_real_estate_loans", col6)
    get_input("revolving_utilization_unsecured", col7)
    
    # 3. Historial de Pagos
    st.markdown("#### ⚠️ Historial de Atrasos (Últimos 2 años)")
    col8, col9, col10 = st.columns(3)
    get_input("number_of_time_30_59_days_past_due", col8)
    get_input("number_of_time_60_89_days_past_due", col9)
    get_input("number_of_times_90_days_late", col10)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Botón principal resaltado
    submitted = st.form_submit_button("🚀 Evaluar Perfil de Riesgo", type="primary", use_container_width=True)


# --- RESULTADOS Y EXPLICACIÓN ---
if submitted:
    with st.spinner("🧠 Analizando el perfil con el modelo..."):
        try:
            result = predict_with_explanation(inputs, model_path=MODEL_PATH)
        except ModelNotAvailableError as error:
            st.error(str(error))
            st.info(
                "Entrena un modelo primero con `python -m src.ml.train` "
                "(o `make train`) y vuelve a intentarlo."
            )
        else:
            st.divider()
            st.header("📊 Resultados del Análisis")
            
            probability = result["probability"]
            decision = result["decision"]
            band = result["risk_band"]
            
            # Semáforo de UX según la banda de riesgo
            if band == "bajo":
                alert_box = st.success
                icon = "✅"
            elif band == "medio":
                alert_box = st.warning
                icon = "⚠️"
            else:
                alert_box = st.error
                icon = "🚨"
                
            alert_box(f"**Decisión sugerida:** `{decision.upper()}` | **Nivel de riesgo:** {band.upper()} {icon}")
            
            # Guardar en el historial de sesión
            hist_id = applicant_id if applicant_id else f"Anónimo {len(st.session_state.history) + 1}"
            st.session_state.history.insert(0, {
                "id": hist_id,
                "probability": probability,
                "decision": decision,
                "band": band
            })
            
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("Probabilidad de Impago (Default)", f"{probability:.1%}")
            with col_b:
                st.markdown("**Nivel de Alerta Visual:**")
                st.progress(min(probability, 1.0)) # Asegura que el valor máximo sea 1.0
                
            top_factors = result.get("top_factors", [])
            if top_factors:
                st.subheader("🔍 Principales Factores de Riesgo")
                st.markdown("Las siguientes variables fueron las que más impacto (SHAP) tuvieron en esta decisión:")
                
                factors_df = pd.DataFrame(top_factors)
                factors_df = factors_df.sort_values(by="impact", ascending=True)
                
                col_chart, col_table = st.columns([2, 1])
                
                with col_chart:
                    fig = go.Figure()
                    colors = ['#ff4b4b' if x > 0 else '#21c354' for x in factors_df['impact']]
                    fig.add_trace(go.Bar(
                        x=factors_df['impact'],
                        y=factors_df['feature'],
                        orientation='h',
                        marker_color=colors
                    ))
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis_title="Impacto en la probabilidad (SHAP)",
                        yaxis_title="",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with col_table:
                    st.dataframe(
                        factors_df.sort_values(by="impact", ascending=False)[["feature", "value", "impact"]],
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.caption(
                    "La explicación SHAP no está disponible (instala la librería `shap` para "
                    "habilitarla)."
                )
