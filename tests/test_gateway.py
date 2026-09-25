import json

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.backends import DatabricksServingBackend, LocalBackend
from gateway.databricks_client import DatabricksError, DatabricksREST
from gateway.main import create_app
from gateway.settings import GatewaySettings
from gateway.sinks import DatabricksDeltaSink, MemorySink, SQLiteSink

PAYLOAD = {
    "applicant_id": "APP-TEST-1",
    "features": {
        "revolving_utilization_unsecured": 0.9,
        "age": 29,
        "number_of_time_30_59_days_past_due": 2,
        "debt_ratio": 0.6,
        "monthly_income": 2500,
        "number_open_credit_lines": 4,
        "number_of_times_90_days_late": 1,
        "number_real_estate_loans": 0,
        "number_of_time_60_89_days_past_due": 1,
        "number_dependents": 2,
    },
}


@pytest.fixture()
def client_and_sink(lr_model, xgb_model):
    sink = MemorySink()
    settings = GatewaySettings(api_keys="secret", challenger_traffic=0.5, log_sink="memory")
    app = create_app(settings, backend=LocalBackend(lr_model, xgb_model), sink=sink)
    with TestClient(app) as client:
        yield client, sink


def test_health_is_public(client_and_sink):
    client, _ = client_and_sink
    body = client.get("/health").json()
    assert body["status"] == "ok" and "champion" in body["versions"]


def test_auth_is_required(client_and_sink):
    client, _ = client_and_sink
    assert client.post("/api/v1/predict", json=PAYLOAD).status_code == 401
    assert client.post("/api/v1/predict", json=PAYLOAD, headers={"X-API-Key": "bad"}).status_code == 401


def test_predict_logs_to_sink(client_and_sink):
    client, sink = client_and_sink
    res = client.post("/api/v1/predict", json=PAYLOAD, headers={"X-API-Key": "secret"})
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["probability"] <= 1
    assert body["variant"] in {"champion", "challenger"}
    assert len(body["top_factors"]) == 3
    logged = sink.predictions[-1]
    assert logged["request_id"] == body["request_id"] and logged["age"] == 29


def test_ab_assignment_is_sticky_through_api(client_and_sink):
    client, _ = client_and_sink
    h = {"X-API-Key": "secret"}
    first = client.post("/api/v1/predict", json=PAYLOAD, headers=h).json()["variant"]
    for _ in range(3):
        assert client.post("/api/v1/predict", json=PAYLOAD, headers=h).json()["variant"] == first
    variants = {
        client.post("/api/v1/predict", json={**PAYLOAD, "applicant_id": f"A{i}"}, headers=h).json()["variant"]
        for i in range(40)
    }
    assert variants == {"champion", "challenger"}


def test_validation_rejects_invalid_age(client_and_sink):
    client, _ = client_and_sink
    bad = json.loads(json.dumps(PAYLOAD))
    bad["features"]["age"] = 5
    assert client.post("/api/v1/predict", json=bad, headers={"X-API-Key": "secret"}).status_code == 422


def test_outcomes_endpoint(client_and_sink):
    client, sink = client_and_sink
    res = client.post(
        "/api/v1/outcomes",
        json={"applicant_id": "APP-TEST-1", "request_id": "r1", "actual_default": True},
        headers={"X-API-Key": "secret"},
    )
    assert res.status_code == 200 and sink.outcomes[-1]["actual_default"] == 1


def test_models_and_monitoring_without_databricks(client_and_sink):
    client, _ = client_and_sink
    h = {"X-API-Key": "secret"}
    assert client.get("/api/v1/models", headers=h).json()["ab_test_active"] is True
    assert client.get("/api/v1/monitoring/summary", headers=h).json()["available"] is False


def test_sqlite_sink(tmp_path):
    sink = SQLiteSink(str(tmp_path / "log.sqlite"))
    sink.log_prediction({"request_id": "r1", "probability": 0.2})
    sink.log_outcome({"applicant_id": "a", "actual_default": 0})
    assert sink.conn.execute("select count(*) from inference_log").fetchone()[0] == 1


def _mock_databricks(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request.content))
        path = request.url.path
        if path.endswith("/invocations"):
            return httpx.Response(
                200,
                json={
                    "predictions": [
                        {
                            "probability": 0.12,
                            "decision": "APROBAR",
                            "risk_band": "C",
                            "top_factors": json.dumps(
                                [
                                    {
                                        "feature": "age",
                                        "label": "Edad",
                                        "impact": 0.3,
                                        "direction": "aumenta_riesgo",
                                    }
                                ]
                            ),
                        }
                    ]
                },
            )
        if path == "/api/2.0/serving-endpoints/credit-risk-endpoint":
            return httpx.Response(
                200,
                json={
                    "config": {
                        "served_entities": [
                            {"name": "champion", "entity_version": "3"},
                            {"name": "challenger", "entity_version": "4"},
                        ]
                    }
                },
            )
        if path == "/api/2.0/sql/statements":
            return httpx.Response(
                200,
                json={
                    "status": {"state": "SUCCEEDED"},
                    "manifest": {"schema": {"columns": [{"name": "x"}]}},
                    "result": {"data_array": [["1"]]},
                },
            )
        return httpx.Response(404, text="not found")

    return httpx.MockTransport(handler)


def test_databricks_backend_queries_served_model():
    calls = []
    rest = DatabricksREST("my.cloud.databricks.com", "tok", transport=_mock_databricks(calls))
    backend = DatabricksServingBackend(rest, "credit-risk-endpoint")
    assert backend.versions() == {"champion": "3", "challenger": "4"}
    out = backend.predict("challenger", [PAYLOAD["features"]], explain=True)
    assert out[0]["top_factors"][0]["feature"] == "age"
    assert any(p.endswith("/served-models/challenger/invocations") for _, p, _ in calls)


def test_delta_sink_batches_inserts():
    calls = []
    rest = DatabricksREST("https://h", "tok", transport=_mock_databricks(calls))
    sink = DatabricksDeltaSink(rest, "wh1", "workspace.credit_risk", flush_every_n=3, flush_every_seconds=999)
    for i in range(2):
        sink.log_prediction(
            {
                "request_id": f"r{i}",
                "probability": 0.1,
                "event_ts": "2026-01-01 00:00:00",
                "decision": "O'Neil",
            }
        )
    assert calls == []
    sink.log_outcome({"applicant_id": "a", "actual_default": 1, "observed_ts": "2026-01-01 00:00:00"})
    statements = [json.loads(c[2])["statement"] for c in calls]
    assert any("INSERT INTO workspace.credit_risk.inference_log" in s for s in statements)
    assert any("O\\'Neil" in s for s in statements)


def test_rest_client_requires_credentials():
    with pytest.raises(DatabricksError):
        DatabricksREST("", "")
