FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config ./config
# Dataset crudo (17MB, commiteado a proposito): permite entrenar el
# modelo en el primer arranque cuando no hay artefacto (ver
# src/ml/bootstrap.py), sin necesitar credenciales de Kaggle.
COPY DATA ./DATA
COPY app.py ./app.py

ENV PYTHONPATH=/app
# Entrena el modelo AQUI (build time), no en cada arranque del contenedor:
# el plan free de Render no tiene disco persistente, asi que sin esto el
# modelo se reentrenaria en cada "cold start" (lento en una instancia con
# CPU/RAM compartida, y ademas repetia una llamada a MLflow que las
# versiones nuevas del filestore rechazan por defecto). Con el modelo ya
# horneado en la imagen, ensure_model_trained() en runtime (src/api/main.py,
# app.py) solo verifica que el archivo existe y no hace nada mas.
ENV MLFLOW_ALLOW_FILE_STORE=true
RUN python -c "from src.ml.bootstrap import ensure_model_trained; ensure_model_trained()"
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
