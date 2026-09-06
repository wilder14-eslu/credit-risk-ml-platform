import logging

from fastapi import APIRouter, HTTPException

from src.api.monitoring import record_prediction
from src.api.schemas import (
    FeatureField,
    FeatureSchemaResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.feature_store.features import load_feature_schema
from src.ml.predict import ModelNotAvailableError, predict_with_explanation

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
        record_prediction(payload, result["probability"], result["decision"])
    except Exception:  # pragma: no cover - logging must never break a prediction
        logger.exception("No se pudo registrar la predicción para monitoreo.")

    return PredictionResponse(
        applicant_id=request.applicant_id,
        probability=result["probability"],
        decision=result["decision"],
        risk_band=result["risk_band"],
        top_factors=result.get("top_factors", []),
    )
