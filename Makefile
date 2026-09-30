.PHONY: setup install test lint evaluate evaluate-quick nested-cv report report-quick train benchmark api run demo all

# 1. Instalar dependencias
setup install:
	pip install -r requirements.txt

# 2. Calidad de código
lint:
	ruff check .

test:
	pytest

# 3. Evaluación base: CV repetida 5x3, holdout aislado, figuras 01-10 (~7 min)
evaluate:
	python -m src.ml.evaluation

evaluate-quick:
	python -m src.ml.evaluation --quick

# 4. CV anidada con Optuna (lenta; reanudable). Escribe reports/nested_cv.json
nested-cv:
	python -m src.ml.nested_cv --n-trials 12

# 5. Reporte completo: evaluación + scorecard + inferencia + costos + cartera +
#    estabilidad + equidad + stress + monitoreo + SHAP + Model/Data Card
report:
	python -m src.pipelines.full_report

report-quick:
	python -m src.pipelines.full_report --quick --output reports_quick

# 6. Entrenar y registrar el modelo que usa la API (MLflow best-effort)
train:
	python -m src.ml.train

benchmark:
	python -m src.ml.benchmark

# 7. Servir
api run:
	uvicorn src.api.main:app --reload

demo:
	streamlit run app.py

all: setup lint test nested-cv report
