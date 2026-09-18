import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from src.ml.predict import load_model
from src.feature_store.features import build_features, FEATURE_NAMES

st.set_page_config(page_title="Explicabilidad Global", page_icon="💡", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%); }
    [data-testid="stSidebar"] { background-color: #ffffff !important; box-shadow: 2px 0 10px rgba(0,0,0,0.05); }
    [data-testid="stSidebarNav"] a { transition: all 0.3s ease-in-out !important; border-radius: 8px !important; margin: 0px 8px !important; }
    [data-testid="stSidebarNav"] a:hover { transform: translateX(8px); background-color: #e2e8f0 !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1, h2, h3, h4 { color: #102a43; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💡 Explicabilidad Global (SHAP)")
st.markdown(
    "Mientras que el evaluador principal muestra por qué se aprobó o rechazó a **una persona específica** (SHAP Local), "
    "esta pantalla revela **qué variables son las más importantes para el modelo en general** al tomar decisiones."
)

@st.cache_data(show_spinner=False)
def load_sample_data():
    from src.data_pipeline.ingest import download_give_me_some_credit
    data_dir = download_give_me_some_credit()
    csv_files = sorted(Path(data_dir).glob("cs-training.csv")) or sorted(Path(data_dir).glob("*.csv"))
    # Tomamos una muestra aleatoria de 500 para calcular SHAP de forma rápida en la demo
    df = pd.read_csv(csv_files[0], index_col=0).sample(500, random_state=42)
    features = build_features(df)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    return features[list(FEATURE_NAMES)]

with st.spinner("🧠 Calculando impactos globales..."):
    try:
        model = load_model()
        X_sample = load_sample_data()
        
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Para el impacto global, usamos el promedio del valor absoluto de SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            "feature": FEATURE_NAMES,
            "importance": mean_abs_shap
        }).sort_values("importance", ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=importance_df["importance"],
            y=importance_df["feature"],
            orientation='h',
            marker_color='#3b82f6'
        ))
        fig.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Impacto Medio Absoluto en la Probabilidad",
            yaxis_title="",
            height=max(400, len(FEATURE_NAMES) * 30)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("⬆️ **Interpretabilidad:** Las variables en la parte superior son las que 'mueven la aguja' más drásticamente. Notarás si las nuevas variables creadas (como `total_past_due` o `disposable_income`) ganaron protagonismo.")
        
    except Exception as e:
        st.error(f"⚠️ No se pudo generar la explicabilidad. Error: {e}")
