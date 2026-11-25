# Data Dictionary (Telco Customer Churn, Kaggle)

7,043 rows x 21 columns. One row = one customer snapshot. Values below are the documented values of the public dataset.

| Column | Type | Allowed values | Role | Notes |
|---|---|---|---|---|
| customerID | string | unique ID | **Identifier - never a feature** | Used only to save/join splits and label rows |
| gender | categorical | Female, Male | Feature (sensitive) | See fairness notes |
| SeniorCitizen | categorical (0/1) | 0, 1 | Feature (sensitive) | Stored as integer; treated as categorical |
| Partner | categorical | Yes, No | Feature (sensitive) | |
| Dependents | categorical | Yes, No | Feature (sensitive) | |
| tenure | numeric (months) | 0-72 in the data; validated 0-120 | Feature | 0 for brand-new customers |
| PhoneService | categorical | Yes, No | Feature | |
| MultipleLines | categorical | Yes, No, No phone service | Feature | |
| InternetService | categorical | DSL, Fiber optic, No | Feature | |
| OnlineSecurity | categorical | Yes, No, No internet service | Feature | add-on |
| OnlineBackup | categorical | Yes, No, No internet service | Feature | add-on |
| DeviceProtection | categorical | Yes, No, No internet service | Feature | add-on |
| TechSupport | categorical | Yes, No, No internet service | Feature | add-on |
| StreamingTV | categorical | Yes, No, No internet service | Feature | add-on |
| StreamingMovies | categorical | Yes, No, No internet service | Feature | add-on |
| Contract | categorical | Month-to-month, One year, Two year | Feature | |
| PaperlessBilling | categorical | Yes, No | Feature | |
| PaymentMethod | categorical | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) | Feature | |
| MonthlyCharges | numeric | >= 0 | Feature | Used in the campaign value proxy |
| TotalCharges | numeric | >= 0, **blank for new customers** | Feature | Blank -> NaN -> imputed (median) inside the pipeline using training data only |
| Churn | binary | Yes, No | **Target** | Cleaned to 1 = Yes (churned), 0 = No |

## Modelling roles
* **Numeric features (3):** tenure, MonthlyCharges, TotalCharges -> median impute + standard scale.
* **Categorical features (16):** everything else except customerID and Churn -> most-frequent impute + one-hot (`handle_unknown="ignore"`).
* **Excluded always:** customerID (identifier), Churn (target).
* **Optional exclusion:** `features.exclude_columns` in `configs/default.yaml` (for example the four sensitive attributes).
* The dataset has no cancellation-reason, churn-date or other post-churn fields. None may be added.
