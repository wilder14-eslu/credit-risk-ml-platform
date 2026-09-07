.PHONY: install train test lint run demo monitor monitor-once

install:
	pip install -r requirements.txt

train:
	python -m src.ml.train

run:
	uvicorn src.api.main:app --reload

demo:
	streamlit run app.py

test:
	pytest

lint:
	ruff check .

monitor:
	python -m src.orchestrator.monitor

monitor-once:
	python -c "from src.orchestrator.monitor import monitoring_flow; import json; print(json.dumps(monitoring_flow(), indent=2, default=str))"
