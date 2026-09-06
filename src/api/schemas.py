from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input contract for /api/v1/predict.

    Field descriptions mirror `config/data_schema.yaml` so API consumers
    know exactly what information they need to provide; the same schema
    powers the `/api/v1/features` endpoint and the Streamlit demo form.
    """

    applicant_id: str | None = Field(
        default=None, description="Identificador opcional del solicitante."
    )
    revolving_utilization_unsecured: float = Field(
        ...,
        ge=0,
        description="Utilización de líneas de crédito no garantizadas (saldo / límite).",
    )
    age: float = Field(..., ge=18, le=120, description="Edad del solicitante, en años.")
    number_of_time_30_59_days_past_due: float = Field(
        0, ge=0, description="Veces con atraso de 30-59 días en los últimos 2 años."
    )
    debt_ratio: float = Field(
        ..., ge=0, description="Deuda mensual / ingreso mensual bruto."
    )
    monthly_income: float = Field(..., ge=0, description="Ingreso mensual bruto.")
    number_open_credit_lines: float = Field(
        ..., ge=0, description="Número de líneas de crédito y préstamos abiertos."
    )
    number_of_times_90_days_late: float = Field(
        0, ge=0, description="Veces con atraso de 90 días o más."
    )
    number_real_estate_loans: float = Field(
        0, ge=0, description="Número de préstamos hipotecarios o de bienes raíces."
    )
    number_of_time_60_89_days_past_due: float = Field(
        0, ge=0, description="Veces con atraso de 60-89 días en los últimos 2 años."
    )
    number_dependents: float = Field(
        0, ge=0, description="Número de dependientes económicos."
    )


class TopFactor(BaseModel):
    feature: str
    value: float
    impact: float


class PredictionResponse(BaseModel):
    applicant_id: str | None = None
    probability: float
    decision: str
    risk_band: str
    top_factors: list[TopFactor] = []
    model_version: str = "local"


class FeatureField(BaseModel):
    name: str
    label: str
    description: str
    min: float | None = None
    max: float | None = None


class FeatureSchemaResponse(BaseModel):
    target: str
    fields: list[FeatureField]
