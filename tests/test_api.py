from fastapi.testclient import TestClient

from src.api.main import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_features_schema_lists_required_inputs() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/features")
    assert response.status_code == 200
    body = response.json()
    field_names = {field["name"] for field in body["fields"]}
    assert "age" in field_names
    assert "monthly_income" in field_names
    assert all(field["label"] and field["description"] for field in body["fields"])


def test_predict_returns_503_without_a_trained_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing-model.joblib"))
    client = TestClient(app)
    payload = {
        "revolving_utilization_unsecured": 0.2,
        "age": 40,
        "debt_ratio": 0.3,
        "monthly_income": 5000,
        "number_open_credit_lines": 4,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 503
