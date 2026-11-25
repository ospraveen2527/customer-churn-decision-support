"""Explicit request/response schemas for the optional FastAPI service."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerRecord(BaseModel):
    """One customer's service/billing attributes (no customer ID is required or logged)."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0, le=500)
    TotalCharges: Optional[float] = Field(default=None, ge=0)


class PolicyMetadata(BaseModel):
    """What was applied, for traceability."""

    threshold: float
    risk_band: str
    rules_version: str
    note: str = "Rule-based suggestions for human review; not predictions of intervention success."


class PredictionResponse(BaseModel):
    churn_probability: float
    threshold_decision: bool
    model_version: str
    policy: PolicyMetadata
    suggested_actions: list[str]


class BatchRequest(BaseModel):
    records: list[CustomerRecord] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
