.PHONY: install lint format test cov gateway dashboard validate deploy-dev deploy-prod train-dev monitor-dev drift-demo

install:
	pip install -r requirements-dev.txt

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check --fix .

test:
	pytest

cov:
	pytest --cov --cov-report=term-missing --cov-report=xml

gateway:
	uvicorn gateway.main:app --reload

dashboard:
	streamlit run app/streamlit_app.py

validate:
	databricks bundle validate -t dev

deploy-dev:
	databricks bundle deploy -t dev

deploy-prod:
	databricks bundle deploy -t prod

train-dev:
	databricks bundle run -t dev ct_training_pipeline

monitor-dev:
	databricks bundle run -t dev production_monitoring

drift-demo:
	databricks bundle run -t dev production_monitoring --params scenario=mixed,batch_size=3000
