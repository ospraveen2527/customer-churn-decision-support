# Stage 0: Project Plan and Repository Blueprint

> **Status: no results exist yet.** All metrics are `TBD` until you run the code.

## 1. Project plan (short)
Build a decision-support system for a subscription business with limited retention budget. It will (1) estimate each customer's churn probability, (2) explain what the model associates with that estimate, (3) suggest transparent rule-based actions for human review, and (4) simulate the business impact of campaigns under explicit, editable assumptions.

## 2. The ML problem
* **Task:** binary classification. **Target:** `Churn = Yes` -> 1, `No` -> 0.
* **Identifier:** `customerID` is never a feature; it is only used to save and join splits.
* **Data:** public Telco Customer Churn dataset (7,043 rows, 21 columns).

## 3. Three different things (do not confuse them)
| Concept | Question answered | Evidence |
|---|---|---|
| **Churn prediction** | Who is likely to leave? | Held-out metrics (AP, ROC-AUC, calibration) |
| **Explanation** | What does the model associate with that score? | Permutation importance, occlusion - associations only |
| **Retention intervention** | Will an offer keep them? | Needs randomised experiment; **cannot be answered from this data** |

## 4. Why accuracy alone is insufficient
About 26-27% of customers churn. A model that predicts "no churn" for everyone is ~73% accurate yet finds zero churners and is useless for retention. Accuracy ignores how rankings, probabilities and error types matter. We report average precision (PR-AUC), ROC-AUC, precision, recall, F1, the confusion matrix, Brier score and log loss, and compare against a Dummy baseline.

## 5. Evaluation strategy (leakage-safe)
1. Validate and clean (row-level only; no learning from data).
2. **Stratified split 60/20/20, seed 42**, saved as ID files.
3. All learned preprocessing lives inside `Pipeline`/`ColumnTransformer`, fitted on train only; calibration wraps the whole pipeline so preprocessing is refit inside each calibration fold.
4. Tune with 5-fold stratified CV on **train only**; choose models, calibration and threshold on **validation**.
5. Freeze model, calibration, threshold, policy and assumptions (`artifacts/frozen_config.json`, with a hash of the assumptions).
6. Evaluate on **test once**; a lock file blocks reruns; `load_split("test")` is blocked elsewhere.
7. No resampling/SMOTE anywhere. (Class imbalance is handled by metric choice and threshold selection.)

## 6. Campaign-simulation strategy
Inputs: churn probability, monthly charges, contribution margin, retention horizon, assumed save rate, contact cost, budget.
* value proxy = monthly_charges x margin x horizon
* expected net value = churn_probability x save_rate x value proxy - contact_cost
* Policies: random (many seeds), highest risk, expected net value (descending until budget is used; never negative; deterministic ties).
* Output: contacted, historical churners targeted, simulated retained, simulated net value, simulated ROI, plus sensitivity sweeps.
* Everything is labelled **SIMULATED**. No claim of optimality or real savings.

## 7. Stage roadmap
| Stage | What | Main files | Done when |
|---|---|---|---|
| 0 | Plan, structure, config | this file, README, configs | you approve |
| 1 | Load, validate, clean, split | `src/config.py`, `src/data.py`, `tests/test_data.py` | splits + quality report saved, tests pass |
| 2 | EDA + feature audit | `notebooks/01_...ipynb`, `reports/data_audit.md` | notebook runs on real data |
| 3 | Pipeline + baselines | `src/features.py`, `src/train.py` | `artifacts/metrics.csv` |
| 4 | Compare + tune | `src/train.py`, `src/evaluate.py` | `reports/model_comparison.md` |
| 5 | Calibration + threshold | `src/calibrate.py` | frozen config |
| 6 | Explainability | `src/explain.py` | `reports/explainability.md` |
| 7 | Campaign simulator | `src/policy.py` | `reports/campaign_simulation.md` |
| 8 | Rule-based recommendations | `src/policy.py` | `reports/retention_actions.md` |
| 9 | Streamlit dashboard | `app/streamlit_app.py`, `src/app_utils.py` | runs locally |
| 10 | Final test, model card, README | `src/evaluate.py`, `reports/*` | `final_evaluation.md` |
| 11 | Optional API/Docker | `api/`, `Dockerfile` | runs locally |
| 12 | Portfolio materials | `reports/portfolio_materials.md` | metrics pasted in |

## 8. File-by-file purpose
| File | Purpose |
|---|---|
| `configs/default.yaml` | Every tunable choice (seed, split, grids, threshold rule, simulation assumptions) |
| `src/config.py` | Paths, column lists, allowed category values, config loader, hashing |
| `src/data.py` | Load, validate, clean, quality report, stratified split, protected split loader |
| `src/features.py` | Feature spec, `ColumnTransformer`, `Pipeline` builders, reference values |
| `src/train.py` | Baselines, CV, tuning, model comparison and selection, saving models + metadata |
| `src/evaluate.py` | Metrics, curves, bootstrap CIs, precision/recall@budget, locked final evaluation |
| `src/calibrate.py` | Sigmoid calibration (inside CV), threshold analysis/selection, freezing |
| `src/explain.py` | Permutation importance, occlusion local explanations, plain-language text |
| `src/policy.py` | Campaign simulator, sensitivity analysis, rule-based recommendations, audit log |
| `src/app_utils.py` | Input validation, scoring, CSV export (shared by dashboard and API) |
| `src/plotting.py` | Small matplotlib helpers (extra file, not in the original list) |
| `app/streamlit_app.py` | 7-page dashboard |
| `tests/` | One test module per source module (+ `test_evaluate.py`, optional `test_api.py`) |
| `api/`, `Dockerfile`, `docker-compose.yml` | Optional FastAPI extension |
| `reports/` | Plans, audits, generated results, model card |
| `data/raw|processed|splits` | Source CSV, cleaned CSV + quality report, split ID files |
| `artifacts/` | Saved models, metrics, metadata, frozen config |

## 9. Folder structure
```
customer-churn/
  README.md  requirements.txt  requirements-optional.txt  .gitignore  Makefile  pytest.ini
  Dockerfile  docker-compose.yml
  configs/default.yaml
  data/{raw,processed,splits}/
  notebooks/01_exploratory_data_analysis.ipynb
  src/{__init__,config,data,features,train,evaluate,calibrate,explain,policy,app_utils,plotting}.py
  app/streamlit_app.py
  api/{main,schemas}.py
  tests/{conftest,test_data,test_features,test_train,test_evaluate,test_policy,test_predictions,test_api}.py
  artifacts/   reports/ (figures/)
```

## 10. Assumptions and limitations
See `reports/assumptions_and_limitations.md`.
