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

from src.feature_store.features import load_feature_schema
from src.ml.predict import (
    DEFAULT_MODEL_PATH,
    ModelNotAvailableError,
    predict_with_explanation,
)

st.set_page_config(page_title="Credit Risk ML Platform", page_icon="💳", layout="centered")


@st.cache_resource(show_spinner="Entrenando el modelo (primera vez en este servidor)...")
def _ensure_model() -> str:
    """Train the model on first run if no artifact exists yet, and return its path."""
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.is_file():
        from src.data_pipeline.ingest import download_give_me_some_credit
        from src.data_pipeline.validate import (
            clean_out_of_range_rows,
            validate_input_data,
        )
        from src.ml.train import train_model

        data_dir = download_give_me_some_credit()
        csv_files = sorted(Path(data_dir).glob("cs-training.csv")) or sorted(
            Path(data_dir).glob("*.csv")
        )
        raw_data = pd.read_csv(csv_files[0], index_col=0)
        raw_data = clean_out_of_range_rows(raw_data)
        validate_input_data(raw_data, require_target=True)
        train_model(raw_data, model_path=model_path, register=False)
    return str(model_path)


MODEL_PATH = _ensure_model()

st.title("💳 Evaluación de riesgo crediticio")
st.caption(
    "Demo de *default prediction* sobre el dataset Kaggle "
    "\"Give Me Some Credit\". Completa los datos del solicitante: cada "
    "campo indica, en su descripción, exactamente qué información "
    "necesitas reunir para poder evaluar el riesgo."
)

schema = load_feature_schema()
fields = schema["features"]

with st.form("credit_application"):
    st.subheader("Datos del solicitante")
    applicant_id = st.text_input("Identificador del solicitante (opcional)")

    inputs: dict[str, float] = {}
    field_items = list(fields.items())
    columns = st.columns(2)

    defaults = {
        "age": 40.0,
        "monthly_income": 5000.0,
        "number_open_credit_lines": 4.0,
        "debt_ratio": 0.3,
        "revolving_utilization_unsecured": 0.2,
        "number_dependents": 1.0,
    }

    for index, (name, definition) in enumerate(field_items):
        target_column = columns[index % 2]
        min_value = float(definition.get("min", 0.0))
        max_value = definition.get("max")
        default_value = max(defaults.get(name, min_value), min_value)
        with target_column:
            inputs[name] = st.number_input(
                definition.get("label", name),
                min_value=min_value,
                max_value=float(max_value) if max_value is not None else None,
                value=float(default_value),
                help=definition.get("description", ""),
            )

    submitted = st.form_submit_button("Evaluar riesgo")

if submitted:
    try:
        result = predict_with_explanation(inputs, model_path=MODEL_PATH)
    except ModelNotAvailableError as error:
        st.error(str(error))
        st.info(
            "Entrena un modelo primero con `python -m src.ml.train` "
            "(o `make train`) y vuelve a intentarlo."
        )
    else:
        probability = result["probability"]
        decision = result["decision"]
        band = result["risk_band"]
        band_color = {"bajo": "green", "medio": "orange", "alto": "red"}[band]

        st.subheader("Resultado")
        col1, col2 = st.columns(2)
        col1.metric("Probabilidad de incumplimiento (default)", f"{probability:.1%}")
        col2.markdown(
            f"**Decisión sugerida:** `{decision.upper()}`\n\n"
            f"**Nivel de riesgo:** :{band_color}[{band.upper()}]"
        )

        top_factors = result.get("top_factors", [])
        if top_factors:
            st.subheader("Factores que más influyeron (SHAP)")
            factors_df = pd.DataFrame(top_factors).set_index("feature")
            st.bar_chart(factors_df["impact"])
            st.dataframe(
                factors_df.reset_index()[["feature", "value", "impact"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption(
                "La explicación SHAP no está disponible (instala `shap` para "
                "habilitarla)."
            )

st.divider()
with st.expander("¿Qué información necesito para usar esta demo?"):
    for name, definition in fields.items():
        st.markdown(f"- **{definition.get('label', name)}**: {definition.get('description', '')}")
    st.markdown(
        "Estos son exactamente los mismos campos que expone "
        "`GET /api/v1/features` en la API."
    )
