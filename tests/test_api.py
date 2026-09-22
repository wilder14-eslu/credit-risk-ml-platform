from fastapi.testclient import TestClient

from src.api.main import app
from src.config import settings

AUTH_HEADERS = {"X-API-Key": settings.api_key}


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


def test_predict_requires_api_key() -> None:
    """Fase 0, Capa 1: sin X-API-Key, /predict responde 401, nunca corre la predicción."""
    client = TestClient(app)
    payload = {
        "revolving_utilization_unsecured": 0.2,
        "age": 40,
        "debt_ratio": 0.3,
        "monthly_income": 5000,
        "number_open_credit_lines": 4,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 401


def test_predict_rejects_wrong_api_key() -> None:
    client = TestClient(app)
    payload = {
        "revolving_utilization_unsecured": 0.2,
        "age": 40,
        "debt_ratio": 0.3,
        "monthly_income": 5000,
        "number_open_credit_lines": 4,
    }
    response = client.post(
        "/api/v1/predict", json=payload, headers={"X-API-Key": "clave-incorrecta"}
    )
    assert response.status_code == 401


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
    response = client.post("/api/v1/predict", json=payload, headers=AUTH_HEADERS)
    assert response.status_code == 503


def test_docs_require_api_key_when_enabled() -> None:
    """/docs y /openapi.json quedan detrás de la misma API key (Fase 0)."""
    if not settings.docs_enabled:
        return
    client = TestClient(app)
    assert client.get("/docs").status_code == 401
    assert client.get("/openapi.json").status_code == 401
    assert client.get("/docs", headers=AUTH_HEADERS).status_code == 200


def test_docs_also_accept_api_key_as_query_param() -> None:
    """Un navegador normal no manda cabeceras al navegar; /docs debe aceptar
    también `?api_key=...` (y el openapi_url embebido en la página debe
    llevar la misma key, para que el fetch del propio Swagger UI funcione).
    """
    if not settings.docs_enabled:
        return
    client = TestClient(app)
    response = client.get(f"/docs?api_key={settings.api_key}")
    assert response.status_code == 200
    assert f"api_key={settings.api_key}" in response.text
    assert client.get(f"/openapi.json?api_key={settings.api_key}").status_code == 200
    assert client.get("/docs?api_key=clave-incorrecta").status_code == 401
