# Model Card: Telco Churn Risk Model

> Status: **no results exist yet.** Fields marked `TBD` must be filled from your own run. This card must not be
> published with numbers that were not produced by `python -m src.evaluate`.

## Model details
* Task: binary classification of `Churn = Yes`. Output: calibrated (if validation says so) churn probability.
* Pipeline: `ColumnTransformer` (median-impute + scale numeric; mode-impute + one-hot categorical) + selected model
  (TBD: logistic regression / random forest / optional XGBoost, chosen on validation data).
* Version: TBD (`artifacts/frozen_config.json -> model_version`). Seed 42. Package versions in `artifacts/*_metadata.json`.

## Intended use
* Educational / portfolio demonstration of a leakage-safe churn workflow.
* Prioritising customers for **human review** in a hypothetical retention team, together with explicit, editable cost assumptions.

## Out-of-scope use
* Automatic decisions about customers (pricing, service denial, contract terms) without human review.
* Claiming that a recommended action will reduce churn or save revenue.
* Use on a different company, country or period without re-validation.
* Using sensitive attributes to decide who gets expensive treatment.

## Data limitations
Public single-snapshot dataset, 7,043 rows, no dates; possible optimism because churners' tenure/TotalCharges cover their
whole lifetime; unknown collection process; one company. See `reports/assumptions_and_limitations.md`.

## Fairness considerations
`gender`, `SeniorCitizen`, `Partner`, `Dependents` are model inputs by default (per project spec) and may act directly or
as proxies. They are never used in retention rules or campaign selection. A group-wise audit (error rates, calibration by
group) and an exclusion experiment (`features.exclude_columns`) have **not** been run: TBD.

## Evaluation results (test set, evaluated once)
| Metric | Value (95% CI) |
|---|---|
| Average precision | TBD |
| ROC-AUC | TBD |
| Precision / Recall / F1 at frozen threshold | TBD |
| Brier score / Log loss | TBD |
| Precision@budget / Recall@budget | TBD |

Campaign simulation results are **simulated from assumptions** and are not model performance.

## Ethical risks
* Misreading probabilities as certainty, or simulated ROI as measured savings.
* Targeting offers in ways that disadvantage groups; manipulative retention tactics.
* Storing or exposing customer data in dashboards/APIs (the app stores none; logs contain no customer attributes).
* Automation bias: reviewers rubber-stamping suggestions. Human review is required for expensive actions.

## Deployment limitations
Local Streamlit demo and optional local FastAPI/Docker service only. No authentication, rate limiting, live monitoring,
retraining pipeline or privacy/security review. **Not production-ready.**

## Monitoring plan (proposed, not implemented live)
Track input schema failures, missingness, score distribution, latency, error rate, delayed-label performance and data
drift (`reports/monitoring.md`). Drift does not prove failure; retraining requires investigation and validation.
