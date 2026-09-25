"""Contratos de la API (Pydantic). Límites tomados de config/data_schema.yaml."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ApplicantFeatures(BaseModel):
    revolving_utilization_unsecured: float = Field(..., ge=0, examples=[0.35])
    age: float = Field(..., ge=18, le=120, examples=[42])
    number_of_time_30_59_days_past_due: float = Field(0, ge=0, examples=[0])
    debt_ratio: float = Field(..., ge=0, examples=[0.30])
    monthly_income: float | None = Field(None, ge=0, examples=[5200])
    number_open_credit_lines: float = Field(..., ge=0, examples=[8])
    number_of_times_90_days_late: float = Field(0, ge=0, examples=[0])
    number_real_estate_loans: float = Field(0, ge=0, examples=[1])
    number_of_time_60_89_days_past_due: float = Field(0, ge=0, examples=[0])
    number_dependents: float | None = Field(None, ge=0, examples=[2])


class PredictionRequest(BaseModel):
    applicant_id: str | None = Field(None, description="Id estable del solicitante (sticky A/B)")
    features: ApplicantFeatures
    explain: bool = True


class Factor(BaseModel):
    feature: str
    label: str
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    request_id: str
    applicant_id: str
    probability: float
    decision: str
    risk_band: str
    top_factors: list[Factor]
    variant: str
    model_version: str
    latency_ms: float


class OutcomeRequest(BaseModel):
    request_id: str | None = None
    applicant_id: str
    actual_default: bool


class OutcomeResponse(BaseModel):
    status: str
    applicant_id: str
