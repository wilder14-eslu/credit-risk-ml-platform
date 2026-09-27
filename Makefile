.PHONY: install train test lint run demo

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
