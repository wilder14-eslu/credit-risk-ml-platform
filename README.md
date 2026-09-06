# Credit Risk ML Platform

Plataforma de evaluación de riesgo crediticio (*default prediction*) de
extremo a extremo (desde el dato crudo hasta una API y una demo web),
construida como ejercicio de **MLOps de nivel bancario**: pipeline
reproducible, feature store, registro de modelos, explicabilidad, API de
inferencia, monitoreo y CI/CD.

- **Dataset:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle), ~150,000 solicitantes.
- **Modelo:** XGBoost (clasificación binaria de incumplimiento a 2 años) con explicabilidad SHAP.
- **Tracking/registro de modelos:** MLflow.
- **API:** FastAPI.
- **Demo interactiva:** Streamlit.
- **Persistencia:** PostgreSQL (esquema en `src/database/init_db.sql`).
- **Orquestación:** Prefect (`src/orchestrator/pipeline.py`).
- **CI/CD:** GitHub Actions (`.github/workflows/`).

## Por qué está construido así

El diseño sigue dos referencias:

1. **[MLOps: continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)**
   (Google Cloud Architecture Center): define los tres niveles de madurez de
   MLOps que se usan como mapa de ruta de este repositorio.
2. **[Kreuzberger, Kühl & Hirschl (2023), "Machine Learning Operations
   (MLOps): Overview, Definition, and Architecture", arXiv:2205.02302](https://arxiv.org/abs/2205.02302)**
   de donde se toman los 9 principios de MLOps (P1-P9) que se listan más
   abajo, mapeados explícitamente a los módulos del repo.

### Elementos de un sistema de ML real

Solo una fracción de un sistema de ML de producción es "código de ML": todo
lo demás (configuración, validación de datos, monitoreo, gestión de
infraestructura...) es igual o más importante.

![Componentes de un sistema de ML](docs/images/01-componentes-sistema-ml.png)

### Los tres niveles de madurez MLOps (Google Cloud)

**Nivel 0: proceso manual.** Cuadernos y scripts ejecutados a mano; el
paso de un experimento a producción es una entrega manual del modelo
entrenado.

![MLOps Nivel 0](docs/images/02-mlops-nivel-0-manual.png)

**Nivel 1: pipeline de entrenamiento automatizado.** El pipeline completo
(validación de datos → preparación → entrenamiento → validación del
modelo) se ejecuta automáticamente y de forma repetible, con un *feature
store* y un almacén de metadatos como piezas centrales.

![MLOps Nivel 1](docs/images/03-mlops-nivel-1-pipeline-automatizado.png)

**Nivel 2: automatización de CI/CD.** Además del pipeline automatizado, el
propio pipeline se construye, prueba y despliega mediante CI/CD, cerrando
el ciclo con monitoreo de producción que dispara nuevas iteraciones.

![MLOps Nivel 2, fases](docs/images/04-mlops-nivel-2-cicd-fases.png)
![MLOps Nivel 2, flujo](docs/images/05-mlops-nivel-2-flujo-cicd.png)

**Dónde está este repo hoy:** el pipeline de entrenamiento (`src/ml/train.py`,
`src/data_pipeline/`, `src/orchestrator/pipeline.py`) y el registro de
modelos ya cubren el **Nivel 1**. El **Nivel 2** está parcialmente cubierto:
`ci-pipeline.yaml` ya prueba, lintea y construye la imagen Docker en cada
push/PR, pero el despliegue a producción (`cd-deployment.yaml`) es manual
(`workflow_dispatch`) y su paso final de publicación es un placeholder que
debes completar con tu registro/infraestructura real.

### Principios de MLOps aplicados (Kreuzberger et al., 2022)

| # | Principio | Dónde vive en este repo |
|---|-----------|--------------------------|
| P1 | CI/CD | `.github/workflows/ci-pipeline.yaml`, `ci-cd.yml`, `cd-deployment.yaml` |
| P2 | Orquestación de flujos (DAG) | `src/orchestrator/pipeline.py` (flujo Prefect) |
| P3 | Reproducibilidad | Esquema fijo (`config/data_schema.yaml`), semillas fijas en `src/ml/train.py`, MLflow |
| P4 | Versionado (datos, modelo, código) | Git + MLflow Model Registry (`bcp_credit_xgboost`) |
| P5 | Colaboración | Feature store centralizado (`src/feature_store/features.py`) usado por entrenamiento, API y demo |
| P6 | Entrenamiento y evaluación continuos | `src/ml/train.py` + `src/ml/validate_model.py` (gate de promoción) |
| P7 | Registro de metadatos de ML | `mlflow.log_params` / `log_metrics` en cada corrida |
| P8 | Monitoreo continuo | `src/data_pipeline/validate.py` (calidad de datos) + `src/api/monitoring.py` (predicciones y *drift*) |
| P9 | Ciclos de retroalimentación | `drift_trigger()` en `src/api/monitoring.py` alimenta al disparador de reentrenamiento del pipeline |

## Estructura del proyecto

```text
credit-risk-ml-platform/
├── .github/workflows/       # CI (lint+test+build) y CD (despliegue manual)
├── config/
│   └── data_schema.yaml     # Esquema canónico: columnas crudas -> features, límites, labels
├── DATA/GiveMeSomeCredit/    # Dataset crudo (Kaggle)
├── docs/images/              # Diagramas de arquitectura (este README)
├── docker/
│   ├── Dockerfile.api        # Imagen de la API (FastAPI)
│   ├── Dockerfile.app        # Imagen de la demo (Streamlit)
│   └── docker-compose.yml    # postgres + mlflow + api + app
├── src/
│   ├── feature_store/        # Transformaciones compartidas (entrenamiento == inferencia)
│   ├── data_pipeline/        # Ingesta, limpieza y validación de datos
│   ├── ml/                   # Entrenamiento, predicción y explicabilidad (SHAP)
│   ├── api/                  # API REST (FastAPI) + monitoreo de predicciones
│   ├── database/             # Modelos SQLAlchemy y esquema PostgreSQL
│   └── orchestrator/         # Flujo de reentrenamiento continuo (Prefect)
├── tests/                    # PyTest: calidad de datos, API, gates de MLOps, roundtrip train→predict
├── app.py                    # Demo interactiva (Streamlit)
├── Makefile
└── requirements.txt
```

## Inicio rápido (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Entrena el modelo (lee DATA/GiveMeSomeCredit, guarda data/processed/model.joblib)
python -m src.ml.train

# 2a. Levanta la API
uvicorn src.api.main:app --reload

# 2b. O levanta la demo interactiva
streamlit run app.py
```

En Linux/macOS lo mismo aplica cambiando la activación del entorno por
`source .venv/bin/activate`, o usando los atajos del `Makefile`:
`make install`, `make train`, `make run`, `make demo`, `make test`, `make lint`.

> **Nota:** la API necesita un modelo entrenado en `MODEL_PATH` (por
> defecto `data/processed/model.joblib`); si no existe, `/api/v1/predict`
> responde `503` explicando cómo entrenarlo. La demo de Streamlit (`app.py`)
> es distinta: si no encuentra un modelo lo entrena sola la primera vez
> (usando el dataset ya incluido en `DATA/GiveMeSomeCredit/`) y lo deja
> en caché, así que funciona tal cual al desplegarla en Streamlit Community
> Cloud sin pasos manuales.

## Todo el stack con Docker Compose

```bash
cp .env.example .env
docker compose -f docker/docker-compose.yml up --build
```

Esto levanta:

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| `postgres` | 5432 | Esquema de persistencia listo para conectar (`src/database/init_db.sql`), ver nota abajo |
| `mlflow` | 5000 | Servidor de tracking + Model Registry |
| `api` | 8000 | API de inferencia (FastAPI) |
| `app` | 8501 | Demo interactiva (Streamlit) |

`api` y `app` comparten el volumen `../data`, así que un modelo entrenado con
`make train` (o dentro del contenedor `api`) es visible para ambos.

## La demo (Streamlit)

`app.py` expone un formulario con los mismos campos que la API, cada uno
con su descripción (qué dato exacto se necesita, de dónde sale y sus
límites válidos) tomada directamente de `config/data_schema.yaml`: la
misma fuente de verdad que usa la API. Al enviarlo, muestra:

- La probabilidad de incumplimiento y la decisión sugerida (aprobar/rechazar).
- El nivel de riesgo (bajo/medio/alto).
- Los factores que más influyeron en la predicción (SHAP), para que la
  respuesta sea auditable y no una caja negra.
- Un panel "¿Qué información necesito?" con la lista completa de campos
  requeridos, para reutilizar como guion de recolección de datos.

## API

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/v1/health` | GET | Liveness check |
| `/api/v1/features` | GET | Qué campos necesita `/predict` (nombre, etiqueta, descripción, mínimos/máximos) |
| `/api/v1/predict` | POST | Probabilidad de incumplimiento, decisión, nivel de riesgo y factores SHAP |

Ejemplo de solicitud a `/api/v1/predict`:

```json
{
  "applicant_id": "demo-1",
  "revolving_utilization_unsecured": 0.9,
  "age": 29,
  "number_of_time_30_59_days_past_due": 3,
  "debt_ratio": 1.2,
  "monthly_income": 1800,
  "number_open_credit_lines": 2,
  "number_of_times_90_days_late": 2,
  "number_real_estate_loans": 0,
  "number_of_time_60_89_days_past_due": 1,
  "number_dependents": 3
}
```

Respuesta:

```json
{
  "applicant_id": "demo-1",
  "probability": 0.72,
  "decision": "rechazar",
  "risk_band": "alto",
  "top_factors": [
    {"feature": "number_of_times_90_days_late", "value": 2.0, "impact": 1.45},
    {"feature": "number_of_time_30_59_days_past_due", "value": 3.0, "impact": 0.73}
  ],
  "model_version": "local"
}
```

## Estado del monitoreo y la base de datos

Hoy cada predicción se registra en un archivo JSONL local (`data/processed/predictions.jsonl`, vía `src.api.monitoring.record_prediction`) para auditoría y para alimentar `drift_trigger()`. El esquema de PostgreSQL (`src/database/init_db.sql`, tabla `predictions`) y la conexión async (`src/database/connection.py`) ya están listos, pero escribir en la base de datos desde `record_prediction` queda como el siguiente paso natural para un despliegue productivo (sustituir o complementar el JSONL por un `INSERT` async).

## Variables de entorno (`.env`)

Ver `.env.example`. Las relevantes para el modelo:

- `MODEL_PATH`: ruta del artefacto local (`joblib`) que usan la API y la demo.
- `MLFLOW_TRACKING_URI`: por defecto `file:./mlruns` (sin servidor); en
  Docker Compose apunta al servicio `mlflow`.
- `MLFLOW_MODEL_URI`: si se define (p. ej. `models:/bcp_credit_xgboost/Production`),
  la API sirve el modelo desde el Model Registry en vez del artefacto local.
- `DECISION_THRESHOLD`: umbral de probabilidad para pasar de "aprobar" a "rechazar" (por defecto `0.5`).

## Pruebas y calidad

```bash
make test   # pytest: validación de datos, API, gates de promoción de modelo, roundtrip train->predict
make lint   # ruff check .
```

`tests/test_train_predict.py` entrena un modelo real sobre una muestra del
dataset y lo usa para predecir, para detectar cualquier desalineación entre
`src.ml.train`, `src.ml.predict` y el esquema de features; no son mocks.

## Reentrenamiento continuo

`src/orchestrator/pipeline.py` define un flujo de Prefect
(`credit-risk-continuous-training`) que encadena ingesta → limpieza/validación
→ entrenamiento → `evaluate_and_promote` (el modelo nuevo solo se promueve a
`Production` en el Model Registry si su ROC-AUC supera al vigente). Puede
dispararse manualmente, en una agenda, o desde `drift_trigger()` cuando el
monitoreo detecta una tasa de incumplimiento anómala.

## Licencia de los datos

El dataset "Give Me Some Credit" se distribuye bajo los términos de la
competencia de Kaggle; revisa su licencia antes de redistribuirlo.
