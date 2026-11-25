# Optional Deployment Extension (Stage 11)

A **local demo** FastAPI service. It is not a production deployment.

## Endpoints
| Method | Path | Notes |
|---|---|---|
| GET | `/health` | `status`, `model_loaded`, `model_version` |
| POST | `/predict` | One `CustomerRecord`; returns probability, threshold decision, model version, policy metadata, suggested actions |
| POST | `/predict-batch` | `{"records": [...]}` up to 1,000 records |

Schemas: `api/schemas.py` (explicit Pydantic models with allowed values and ranges; invalid input -> HTTP 422).
Logging: counts, latency and model version only; **request bodies are never logged**.

## Run locally
```bash
pip install -r requirements.txt "fastapi>=0.110" "uvicorn[standard]>=0.29" "pydantic>=2.6" httpx
uvicorn api.main:app --reload            # http://127.0.0.1:8000/docs
pytest tests/test_api.py -q
```

## Docker
Train first (artifacts are git-ignored but must exist at build time), then:
```bash
docker compose up --build                # or: docker build -t churn-api . && docker run -p 8000:8000 churn-api
curl http://localhost:8000/health
```
The image runs as a non-root user and copies only `src/`, `api/`, `configs/` and `artifacts/`.

## Not implemented
Authentication, TLS, rate limiting, model registry, CI/CD, live monitoring dashboards, retraining, privacy/security review.
