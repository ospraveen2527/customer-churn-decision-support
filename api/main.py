"""Optional FastAPI service: ``uvicorn api.main:app --reload``.

Logging policy: only counts, latency and model version are logged. Request
bodies (customer attributes) are never logged.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402

from api.schemas import (  # noqa: E402
    BatchRequest, BatchResponse, CustomerRecord, HealthResponse, PolicyMetadata, PredictionResponse,
)
from src.app_utils import (  # noqa: E402
    InputValidationError, ModelBundle, load_bundle, validate_dataframe,
)
from src.policy import RULES_VERSION, recommend_actions  # noqa: E402

import pandas as pd  # noqa: E402

logger = logging.getLogger("churn-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(title="Churn Decision-Support API", version="0.1.0")
_bundle: ModelBundle | None = None


def get_bundle() -> ModelBundle:
    """Lazy-load the model once."""
    global _bundle
    if _bundle is None:
        try:
            _bundle = load_bundle()
        except FileNotFoundError as exc:
            logger.error("Model artifacts missing")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _bundle


def _predict(bundle: ModelBundle, records: list[CustomerRecord]) -> list[PredictionResponse]:
    frame = pd.DataFrame([r.model_dump() for r in records])
    try:
        validated = validate_dataframe(frame, bundle.feature_columns)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors) from exc
    X = validated[bundle.feature_columns]
    proba = bundle.model.predict_proba(X)[:, 1]
    out: list[PredictionResponse] = []
    for (_, row), p in zip(X.iterrows(), proba):
        rec = recommend_actions(row.to_dict(), float(p), bundle.assumptions,
                                bundle.cfg["retention_rules"], bundle.threshold)
        out.append(PredictionResponse(
            churn_probability=round(float(p), 4),
            threshold_decision=bool(p >= bundle.threshold),
            model_version=bundle.version,
            policy=PolicyMetadata(threshold=bundle.threshold, risk_band=rec.risk_band,
                                  rules_version=RULES_VERSION),
            suggested_actions=[a["action"] for a in rec.actions],
        ))
    return out


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check; reports whether model artifacts can be loaded."""
    try:
        b = get_bundle()
        return HealthResponse(status="ok", model_loaded=True, model_version=b.version)
    except HTTPException:
        return HealthResponse(status="degraded", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse)
def predict(record: CustomerRecord) -> PredictionResponse:
    """Score one customer."""
    start = time.perf_counter()
    result = _predict(get_bundle(), [record])[0]
    logger.info("predict ok latency_ms=%.1f model=%s", (time.perf_counter() - start) * 1000,
                result.model_version)
    return result


@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    """Score up to 1000 customers."""
    start = time.perf_counter()
    preds = _predict(get_bundle(), request.records)
    logger.info("predict-batch ok n=%d latency_ms=%.1f", len(preds), (time.perf_counter() - start) * 1000)
    return BatchResponse(predictions=preds)
