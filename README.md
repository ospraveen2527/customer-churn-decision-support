# An Explainable, Calibrated Customer Churn Decision-Support System with Budget-Aware Retention Simulation

A portfolio project that estimates customer churn risk on the public Telco dataset, explains the model's
associations, suggests rule-based retention actions for human review, and **simulates** campaign outcomes under
explicit assumptions.

> **Status: no results yet.** All result tables below are `TBD` until you run the pipeline on the real dataset.
> Nothing in this repository was measured on the real data, and no unsupported claims are made.

## 1. Problem statement
A subscription business has a limited retention budget. Whom should it contact? This project separates four things:
1. **Measured model performance** (held-out metrics).
2. **Explainability** (what the model associates with its scores; not causes).
3. **Simulated campaign outcomes** (calculated from user assumptions; not measured).
4. **Causal retention impact** (not claimed; needs randomised intervention data).

## 2. Dataset
[Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn):
`WA_Fn-UseC_-Telco-Customer-Churn.csv`, 7,043 rows, 21 columns, target `Churn` (Yes/No). `customerID` is an
identifier and is never a feature. Blank `TotalCharges` values are converted to NaN, logged, kept, and imputed inside
the pipeline using training data only. See `reports/data_dictionary.md`.

## 3. Architecture
```
raw CSV -> src/data.py (validate, clean, stratified 60/20/20 split, IDs saved)
        -> src/features.py (Pipeline + ColumnTransformer: impute, scale, one-hot)
        -> src/train.py (baselines, 5-fold CV on train, model comparison on validation)
        -> src/calibrate.py (sigmoid calibration inside CV, threshold on validation, FREEZE)
        -> src/explain.py (permutation importance + occlusion on the deployed predict_proba)
        -> src/policy.py (campaign simulator + rule-based recommendations + audit log)
        -> app/streamlit_app.py / api/main.py (use src/app_utils.py)
        -> src/evaluate.py (locked, one-time test evaluation)
```
Leakage controls: split before any learned preprocessing; everything learned is inside a Pipeline; calibration wraps the
whole pipeline; tuning uses train CV only; model/threshold selection uses validation only; the test split cannot be
loaded except by the final evaluation, which also refuses to run twice; no resampling/SMOTE.

## 4. Setup
Python 3.11+.
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put WA_Fn-UseC_-Telco-Customer-Churn.csv in data/raw/
pytest -q                          # tests use synthetic data, no download needed
```

## 5. How to run training (in order)
```bash
python -m src.data               # validate, clean, split, data-quality report
python -m src.train baseline     # Stage 3: dummy + logistic regression  -> artifacts/metrics.csv
python -m src.train compare      # Stage 4: tune + compare, select model -> reports/model_comparison.md
python -m src.calibrate          # Stage 5: calibration, threshold, FREEZE -> artifacts/frozen_config.json
python -m src.explain            # Stage 6: explanations on validation data
python -m src.policy             # Stage 7: campaign simulation on validation data
```
(`make data baseline compare calibrate explain simulate` does the same.) Each command overwrites the matching
`TBD` report in `reports/` with numbers computed from your run.

**Final test evaluation (run once, only after you are happy with everything above):**
```bash
python -m src.evaluate           # writes reports/final_evaluation.md, artifacts/test_metrics.json
```
Do not change the model, threshold, calibration, policy or assumptions after seeing test results.

## 6. How to run Streamlit
```bash
streamlit run app/streamlit_app.py
```
Pages: overview, single-customer prediction, batch CSV prediction (schema-validated, downloadable results),
explanation, campaign simulator (clearly labelled SIMULATED), model evaluation, limitations. The app loads the
saved pipeline, stores no uploaded data, and sends no communications. Optional API: see `reports/deployment.md`.

## 7. Results
| Item | Value |
|---|---|
| Validation average precision / ROC-AUC (selected model) | TBD |
| Calibration (Brier, log loss, before vs after) | TBD |
| Frozen threshold | TBD |
| Test average precision, ROC-AUC, precision, recall, F1 (95% CI) | TBD |
| Recall@budget / Precision@budget | TBD |
Fill these from `artifacts/metrics.csv`, `reports/calibration.md` and `reports/final_evaluation.md` only after running.

## 8. Explainability
Permutation importance (global) and one-feature-at-a-time occlusion (local), both computed through the deployed
`predict_proba`. They show model associations, not proven causes. Demographic attributes are never altered to create
"lower-risk scenarios", and no explanation implies that changing a feature will prevent churn. SHAP is an optional
extension. See `reports/explainability.md`.

## 9. Campaign simulation assumptions
Defaults in `configs/default.yaml` (margin 0.30, 12-month horizon, save rate 0.20, contact cost 5, budget 2,500) are
**illustrative placeholders, not estimates**. Value proxy = monthly charges x margin x horizon; expected net value =
p(churn) x save rate x value proxy - contact cost. Random, highest-risk and expected-net-value targeting are compared;
sensitivity sweeps vary save rate, contact cost, margin and horizon. Outputs are simulations: they are not revenue
saved or churn reduced, and the policy is not claimed to be optimal. See `reports/campaign_simulation.md`.

Rule-based recommendations (`reports/retention_actions.md`) are transparent suggestions for human review with a
per-rule audit trail; sensitive attributes are never used.

## 10. Limitations
Snapshot data with possible optimism from tenure/TotalCharges for churners; small dataset; no causal evidence;
fairness not fully audited; simulation is assumption-based; not production-ready. Details:
`reports/assumptions_and_limitations.md` and `reports/model_card.md`.

## 11. Future improvements
* Randomised retention pilot or uplift modelling to estimate real treatment effects.
* Group-wise fairness audit and a sensitive-attribute exclusion experiment.
* Time-based validation on data with dates; drift monitoring on real traffic.
* Optional XGBoost / SHAP / MLflow (see `requirements-optional.txt`); cost-curve-based threshold; optimisation-based targeting.

## Repository layout
See `reports/project_plan.md` for the file-by-file breakdown. Tests: `pytest -q` (data, features, training,
evaluation, policy, dashboard logic; API tests run if FastAPI is installed).
