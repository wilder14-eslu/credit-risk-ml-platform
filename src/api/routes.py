import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.monitoring import (
    compute_live_performance,
    drift_trigger,
    load_recent_predictions,
    record_outcome,
    record_prediction,
)
from src.api.schemas import (
    FeatureDrift,
    FeatureField,
    FeatureSchemaResponse,
    LivePerformance,
    MonitoringStatusResponse,
    OutcomeRequest,
    OutcomeResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.database.connection import session_factory
from src.database.repository import persist_outcome, persist_prediction
from src.feature_store.features import load_feature_schema
from src.ml.predict import ModelNotAvailableError, predict_with_explanation
from src.monitoring.drift import compute_feature_drift, load_reference_distribution

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/features", response_model=FeatureSchemaResponse)
async def features_schema() -> FeatureSchemaResponse:
    """Describe exactly which inputs the model needs.

    Both the Streamlit demo and any external client can call this first to
    know which fields to collect from a user before calling /predict.
    """
    schema = load_feature_schema()
    fields = [
        FeatureField(
            name=name,
            label=definition.get("label", name),
            description=definition.get("description", ""),
            min=definition.get("min"),
            max=definition.get("max"),
        )
        for name, definition in schema["features"].items()
    ]
    return FeatureSchemaResponse(target=schema["target"], fields=fields)


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    payload = request.model_dump(exclude={"applicant_id"})
    try:
        result = predict_with_explanation(payload)
    except ModelNotAvailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    try:
        record_prediction(
            payload,
            result["probability"],
            result["decision"],
            applicant_id=request.applicant_id,
        )
    except Exception:  # pragma: no cover - logging must never break a prediction
        logger.exception("No se pudo registrar la predicción para monitoreo.")

    await persist_prediction(
        session_factory,
        request.applicant_id,
        result["probability"],
        result["decision"],
        result.get("top_factors"),
    )

    return PredictionResponse(
        applicant_id=request.applicant_id,
        probability=result["probability"],
        decision=result["decision"],
        risk_band=result["risk_band"],
        top_factors=result.get("top_factors", []),
    )


@router.post("/outcomes", response_model=OutcomeResponse)
async def submit_outcome(request: OutcomeRequest) -> OutcomeResponse:
    """Report the real-world result for an applicant scored earlier.

    This is what feeds `compute_live_performance`: without it the platform
    can only see input drift, never whether the model is actually still
    accurate.
    """
    try:
        record_outcome(request.applicant_id, request.actual_default)
    except Exception:  # pragma: no cover - logging must never break the request
        logger.exception("No se pudo registrar el resultado real para monitoreo.")

    await persist_outcome(session_factory, request.applicant_id, request.actual_default)
    return OutcomeResponse()


@router.get("/monitoring/status", response_model=MonitoringStatusResponse)
async def monitoring_status() -> MonitoringStatusResponse:
    """Current drift/performance signals and whether they'd trigger a retrain.

    Same checks `src.orchestrator.monitor.monitoring_flow` runs on a
    schedule, exposed here so any external dashboard (or a human) can see
    the platform's monitoring state without reading the JSONL logs directly.
    """
    predictions = load_recent_predictions()
    probabilities = [event["probability"] for event in predictions]
    rate_triggered = drift_trigger(probabilities)

    feature_drift_payload: FeatureDrift | None = None
    try:
        reference = load_reference_distribution()
    except FileNotFoundError:
        pass
    else:
        recent_features = pd.DataFrame([event["features"] for event in predictions])
        feature_drift_payload = FeatureDrift(**compute_feature_drift(recent_features, reference))

    performance = compute_live_performance()

    retrain_recommended = rate_triggered or bool(
        feature_drift_payload and feature_drift_payload.drift_detected
    )

    return MonitoringStatusResponse(
        n_predictions=len(predictions),
        rate_drift_triggered=rate_triggered,
        feature_drift=feature_drift_payload,
        performance=LivePerformance(**performance),
        retrain_recommended=retrain_recommended,
    )
