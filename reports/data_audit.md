# Data Audit and Feature Audit

> Status: the **feature-availability and fairness review below is analysis of the schema** and does not depend on results.
> Numeric findings are `TBD` until you run `notebooks/01_exploratory_data_analysis.ipynb` on the real data
> (it uses the TRAINING split only). Observed patterns are **associations, not causes**.

## 1. Quality checks (run `python -m src.data` -> `reports/data_quality_report.md`)
Target distribution: TBD. Blank `TotalCharges`: TBD rows (expected to be tenure 0 customers; verified by the report).
Duplicates, negative charges, invalid tenure, unexpected category values: TBD.

## 2. Observed churn patterns (fill from the notebook; do not claim causation)
| Topic | Observation | Figure |
|---|---|---|
| Target distribution | TBD | `fig_target_distribution.png` |
| Tenure vs churn | TBD | `fig_tenure_by_churn.png`, `fig_churn_by_tenure_bin.png` |
| MonthlyCharges / TotalCharges | TBD | `fig_charges_by_churn.png` |
| Contract | TBD | `fig_churn_by_contract.png` |
| PaymentMethod | TBD | `fig_churn_by_paymentmethod.png` |
| InternetService | TBD | `fig_churn_by_internetservice.png` |
| Add-on services | TBD | `fig_churn_by_addons.png` |
| Numeric correlations | TBD | `fig_numeric_correlation.png` |

Association is not causation: customers on month-to-month contracts may churn more for many reasons (price sensitivity,
life events, competitors) that the data does not contain; changing someone's contract is not shown to change their behaviour.

## 3. Would the feature be available before a retention decision?
| Feature | Available before decision? | Comment |
|---|---|---|
| customerID | n/a | Identifier; excluded |
| gender, SeniorCitizen, Partner, Dependents | Yes | **Sensitive**; see section 4 |
| tenure | Yes (current value) | **Caution:** in this snapshot churners' tenure ends at churn, so it may be partly outcome-influenced |
| PhoneService, MultipleLines, InternetService | Yes | Current subscription |
| OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | Yes | Current add-ons |
| Contract, PaperlessBilling, PaymentMethod | Yes | Current terms |
| MonthlyCharges | Yes | Current plan price |
| TotalCharges | Yes, with caution | Cumulative; for churners it includes all charges to churn (tenure x charges); same optimism risk as tenure |
| Churn | **No (target)** | Outcome; never a feature |

No cancellation reason, churn date, or other post-churn field exists in the dataset, and none may be created.

## 4. Sensitive attributes and fairness
Potentially sensitive: `gender`, `SeniorCitizen` (age proxy), `Partner`, `Dependents`. Other fields (payment method,
paperless billing) may proxy for socioeconomic factors. Handling:
* Kept as model inputs by default (project spec) but removable via `features.exclude_columns`.
* Never used in retention rules, campaign selection or in "what-if" scenarios.
* Group-wise churn rates, error rates and calibration should be compared (TBD, future work).

## 5. Features entering the model
Numeric (3): tenure, MonthlyCharges, TotalCharges. Categorical (16): all other columns except customerID and Churn.
Total 19 inputs. Nothing derived from the target or the future is created.
