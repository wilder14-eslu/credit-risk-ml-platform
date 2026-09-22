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
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
