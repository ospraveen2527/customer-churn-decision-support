# Portfolio, Resume and Interview Materials (Stage 12)

> Replace every `{TBD: ...}` with a number from YOUR run (`artifacts/metrics.csv`, `reports/final_evaluation.md`).
> Do not publish a claim you cannot trace to a file in this repo.

## 1. GitHub repository description
Leakage-safe churn prediction on the Telco dataset: calibrated probabilities, validation-selected threshold, permutation/occlusion
explanations, rule-based retention suggestions, a budget-aware campaign **simulator** (assumption-based), Streamlit dashboard, tests.

## 2. LinkedIn post (draft)
I built an end-to-end churn decision-support project, aiming for correctness over flashy numbers. It separates measured model
performance (test AP {TBD: AP}, ROC-AUC {TBD: AUC}, evaluated once on a locked test set) from simulated campaign outcomes, which depend
entirely on stated assumptions and are not measured savings. Highlights: stratified train/validation/test split with all preprocessing
inside sklearn Pipelines, calibration inside CV, a threshold frozen before testing, explanations that show associations not causes,
and a Streamlit app. Biggest lessons: why accuracy misleads on imbalanced data, why leakage hides in preprocessing, and why predicting
churn is not the same as knowing an offer works. Code: {TBD: link}

## 3. Resume bullets (fill in metrics)
* Built a leakage-safe churn pipeline (sklearn Pipeline/ColumnTransformer, stratified 60/20/20 split, 5-fold CV on train only) achieving test ROC-AUC {TBD} / average precision {TBD} on a locked, once-evaluated test set.
* Calibrated probabilities (sigmoid inside CV; Brier {TBD} vs {TBD}) and froze a validation-selected threshold; wrote 60+ pytest tests covering data validation, leakage, policy logic and dashboard scoring.
* Developed a budget-aware retention-campaign simulator and Streamlit dashboard with clearly labelled assumption-based scenarios, rule-based human-review recommendations and audit trail.

## 4. Two-minute demo script
0:00 "This predicts churn and helps prioritise a limited retention budget, while keeping measurement, explanation and simulation separate."
0:20 Overview page: the four evidence levels. 0:40 Single-customer page: enter a month-to-month, short-tenure customer; show probability, band, top factors
("associations, not causes") and suggested actions for human review. 1:10 Batch upload, download CSV. 1:25 Campaign simulator: change save rate and contact cost, show that
the best policy can change; point at the SIMULATED banner. 1:50 Evaluation page and limitations: test set used once, no causal claim, not production-ready.

## 5. Ten likely interview questions
1. **Why not report accuracy?** About a quarter churn, so predicting "no churn" is ~73% accurate and useless. I use average precision, ROC-AUC, precision/recall, Brier and log loss.
2. **How did you prevent leakage?** Split first; imputation, scaling and encoding inside a Pipeline fitted on train; calibration wraps the pipeline; tuning on train CV; selection on validation; test locked and used once.
3. **Why calibrate?** Campaign value uses probability x save rate x value, so probabilities must be meaningful. I compared sigmoid against uncalibrated using Brier/log loss and a reliability diagram on validation.
4. **How was the threshold chosen?** A rule declared in config (max recall subject to minimum precision) applied on validation, then frozen before testing.
5. **Why not SMOTE?** It must happen after splitting and inside CV, and it distorts probabilities; threshold and metric choice handle imbalance here.
6. **Does your model prove an offer will retain customers?** No. That needs randomised intervention data or uplift modelling; my simulation is assumption-based and labelled so.
7. **What do your explanations mean?** Permutation importance and occlusion show associations in the model, not causes; I never alter demographic attributes to create fake scenarios.
8. **How do you handle fairness?** Sensitive attributes are never used for retention treatment; I documented proxy risks and listed a group-wise audit and exclusion experiment as future work.
9. **What is a weakness of the dataset?** Single snapshot, no dates; churners' tenure and TotalCharges cover their whole lifetime, so performance may be optimistic.
10. **How would you deploy and monitor it?** Optional FastAPI/Docker demo exists; the monitoring plan (schema, missingness, score drift, latency, delayed labels) is a plan only, and drift is a signal not proof of failure.

## 6. Limitations to mention honestly
Public snapshot data; small sample; no causal evidence; simulation assumptions are placeholders; fairness not fully audited; not production-ready; dashboard/API are local demos.

## 7. Improvements that would strengthen the project
Randomised pilot / uplift model; group-wise fairness audit; time-based validation; XGBoost + SHAP comparison with the simpler-model rule; cost-curve threshold; optimisation-based targeting with validation; MLflow tracking; CI with GitHub Actions; real monitoring.
