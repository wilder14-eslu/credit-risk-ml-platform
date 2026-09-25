"""Gateway de inferencia (FastAPI, desplegado en Render).

Responsabilidades que NO tiene el endpoint de Databricks por sí solo:
- Autenticación por API key y validación del contrato de entrada.
- Asignación A/B determinista (sticky por `applicant_id`) champion/challenger.
- Registro de cada predicción y de los outcomes reales en Delta Lake
  (tablas `inference_log` y `outcomes`), que cierran el ciclo de monitoreo.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, status

from credit_risk import __version__
from credit_risk.config import feature_definitions, platform_config
from credit_risk.monitoring.ab_testing import assign_variant
from gateway.backends import DatabricksServingBackend, LocalBackend, ServingBackend
from gateway.databricks_client import DatabricksREST
from gateway.schemas import (
    OutcomeRequest,
    OutcomeResponse,
    PredictionRequest,
    PredictionResponse,
)
from gateway.settings import GatewaySettings, get_settings
from gateway.sinks import DatabricksDeltaSink, MemorySink, Sink, SQLiteSink

logger = logging.getLogger("gateway")
logging.basicConfig(level=logging.INFO)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def build_components(settings: GatewaySettings) -> tuple[ServingBackend, Sink, DatabricksREST | None]:
    client = None
    if settings.databricks_host and settings.databricks_token:
        client = DatabricksREST(
            settings.databricks_host, settings.databricks_token, settings.request_timeout_seconds
        )
    if settings.serving_backend == "databricks":
        if client is None:
            raise RuntimeError("SERVING_BACKEND=databricks requiere DATABRICKS_HOST y DATABRICKS_TOKEN")
        backend: ServingBackend = DatabricksServingBackend(client, settings.serving_endpoint)
    else:
        backend = LocalBackend.from_path_or_train(settings.local_model_path)

    if settings.log_sink == "databricks" and client and settings.databricks_warehouse_id:
        sink: Sink = DatabricksDeltaSink(
            client,
            settings.databricks_warehouse_id,
            settings.fq,
            settings.flush_every_n,
            settings.flush_every_seconds,
        )
    elif settings.log_sink == "memory":
        sink = MemorySink()
    else:
        sink = SQLiteSink(settings.sqlite_path)
    return backend, sink, client


def create_app(
    settings: GatewaySettings | None = None,
    backend: ServingBackend | None = None,
    sink: Sink | None = None,
    client: DatabricksREST | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.backend is None:
            app.state.backend, app.state.sink, app.state.client = build_components(settings)
        yield
        app.state.sink.flush()

    app = FastAPI(
        title="Credit Risk Gateway",
        version=__version__,
        description="Gateway de scoring crediticio: A/B testing y feedback loop sobre Databricks.",
        lifespan=lifespan,
    )
    app.state.backend, app.state.sink, app.state.client = backend, sink, client

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        keys = settings.api_key_set
        if keys and x_api_key not in keys:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API key inválida o ausente")

    def traffic() -> float:
        if settings.challenger_traffic is not None:
            return settings.challenger_traffic
        return float(platform_config()["ab_testing"]["challenger_traffic"])

    @app.get("/health")
    def health() -> dict:
        versions = app.state.backend.versions() if app.state.backend else {}
        return {"status": "ok", "backend": getattr(app.state.backend, "name", None), "versions": versions}

    @app.get("/api/v1/features")
    def features() -> dict:
        return {"features": feature_definitions()}

    @app.get("/api/v1/models", dependencies=[Depends(require_key)])
    def models() -> dict:
        versions = app.state.backend.versions()
        return {
            "served": versions,
            "ab_test_active": "challenger" in versions,
            "challenger_traffic": traffic() if "challenger" in versions else 0.0,
        }

    @app.post("/api/v1/predict", response_model=PredictionResponse, dependencies=[Depends(require_key)])
    def predict(request: PredictionRequest) -> PredictionResponse:
        backend = app.state.backend
        versions = backend.versions()
        applicant_id = request.applicant_id or f"anon-{uuid.uuid4().hex[:12]}"
        share = traffic() if "challenger" in versions else 0.0
        variant = assign_variant(applicant_id, share)
        record = request.features.model_dump()

        start = time.perf_counter()
        try:
            result = backend.predict(variant, [record], request.explain)[0]
        except Exception as exc:
            if variant == "challenger":  # degradación elegante al champion
                logger.warning("Challenger falló (%s); se usa champion", exc)
                variant = "champion"
                result = backend.predict(variant, [record], request.explain)[0]
            else:
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE, f"Serving no disponible: {exc}"
                ) from exc
        latency = (time.perf_counter() - start) * 1000

        response = PredictionResponse(
            request_id=str(uuid.uuid4()),
            applicant_id=applicant_id,
            probability=float(result["probability"]),
            decision=str(result["decision"]),
            risk_band=str(result["risk_band"]),
            top_factors=result.get("top_factors") or [],
            variant=variant,
            model_version=str(versions.get(variant, "unknown")),
            latency_ms=round(latency, 2),
        )
        app.state.sink.log_prediction(
            {
                **record,
                "request_id": response.request_id,
                "applicant_id": applicant_id,
                "event_ts": _now(),
                "source": "gateway",
                "variant": variant,
                "model_version": response.model_version,
                "probability": response.probability,
                "decision": response.decision,
                "risk_band": response.risk_band,
                "latency_ms": response.latency_ms,
                "scenario": "live",
            }
        )
        return response

    @app.post("/api/v1/outcomes", response_model=OutcomeResponse, dependencies=[Depends(require_key)])
    def outcomes(request: OutcomeRequest) -> OutcomeResponse:
        app.state.sink.log_outcome(
            {
                "applicant_id": request.applicant_id,
                "request_id": request.request_id,
                "actual_default": int(request.actual_default),
                "observed_ts": _now(),
                "source": "gateway",
            }
        )
        return OutcomeResponse(status="registrado", applicant_id=request.applicant_id)

    @app.get("/api/v1/monitoring/summary", dependencies=[Depends(require_key)])
    def monitoring_summary() -> dict:
        client = app.state.client
        if client is None or not settings.databricks_warehouse_id:
            return {"available": False, "detail": "Configura DATABRICKS_WAREHOUSE_ID para leer Delta"}
        fq, wh = settings.fq, settings.databricks_warehouse_id
        queries = {
            "monitoring": f"SELECT * FROM {fq}.monitoring_metrics ORDER BY run_ts DESC LIMIT 30",
            "drift": f"""SELECT * FROM {fq}.drift_by_feature
                         WHERE run_id = (SELECT run_id FROM {fq}.monitoring_metrics
                                         ORDER BY run_ts DESC LIMIT 1)""",
            "ab_tests": f"SELECT * FROM {fq}.ab_test_results ORDER BY run_ts DESC LIMIT 10",
            "benchmark": f"""SELECT * FROM {fq}.model_benchmark
                             WHERE run_ts = (SELECT max(run_ts) FROM {fq}.model_benchmark)""",
            "retrains": f"SELECT * FROM {fq}.retrain_events ORDER BY event_ts DESC LIMIT 10",
        }
        out: dict = {"available": True}
        for key, sql in queries.items():
            try:
                out[key] = client.sql(wh, sql)
            except Exception as exc:
                out[key] = []
                out.setdefault("errors", {})[key] = str(exc)
        return out

    return app


app = create_app()
