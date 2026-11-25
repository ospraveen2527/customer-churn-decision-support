# Monitoring Plan (proposal; NOT implemented live)

No live monitoring exists in this repository. This is what a realistic deployment would track.

| Signal | How | Example alert (placeholder) | First response |
|---|---|---|---|
| Input schema failures | Count of 422/validation errors per day | > 1% of requests | Check upstream data contract |
| Missingness | Share of missing values per feature vs training | TotalCharges missing >> training rate | Check pipeline / source system |
| Score distribution | Mean and quantiles of predicted probability; PSI vs validation scores | Large shift | Investigate inputs before blaming the model |
| Prediction latency | p50/p95/p99 per request | p95 above agreed limit | Profile / scale |
| Error rate | 5xx per day | > 0.5% | Logs, model artifact check |
| Delayed-label performance | Join predictions to outcomes after the churn window; AP, calibration | Material drop vs validation | Diagnose; compare to baseline |
| Data drift | Per-feature PSI / KS / chi-square vs training | Threshold per feature | Investigate |

## Principles
* **Drift is a signal, not a verdict.** Drift can be harmless (seasonal mix) or a data bug; it does not prove the model has failed.
  Conversely, no drift does not prove the model is good.
* **Retraining requires investigation and validation:** identify the cause, rebuild with the same leakage-safe pipeline on new
  data, compare to the current model on a fresh validation set, then freeze a new threshold and re-run a locked evaluation.
* Logs must not contain customer attributes. Use aggregates and pseudonymous references.
* Retention outcomes should be measured with a randomised holdout, otherwise improvement cannot be attributed to the campaign.
