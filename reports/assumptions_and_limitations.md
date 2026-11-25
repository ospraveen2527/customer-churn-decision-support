# Assumptions and Limitations

**No results exist yet.** Every metric in this repository is `TBD` until you run the code on the real dataset.

## Assumptions
1. The public Telco CSV has the documented schema (21 columns, `Churn` in {Yes, No}, unique `customerID`).
2. Blank `TotalCharges` values belong to brand-new customers (tenure 0). The code verifies and logs the tenure of every blank row rather than assuming.
3. Each row is a customer snapshot; the file has no dates, so time-ordered validation is impossible.
4. Campaign parameters (contribution margin, retention horizon, save rate, contact cost, budget) are **illustrative placeholders** in `configs/default.yaml`. They are not estimated from data.
5. The retention-horizon value proxy (`monthly_charges x margin x months`) is a simplification (no discounting, no further churn inside the horizon).
6. Random seed 42 everywhere; results can still vary slightly across library versions.

## Limitations
* **Observational data, no causal claims.** A high churn probability does not show that an offer works. Estimating retention impact needs a randomised experiment or uplift modelling with intervention data.
* **Possible optimism from snapshot design.** For churned customers, `tenure` and `TotalCharges` reflect their whole time with the company, while active customers are still accruing. A real "predict before they leave" setting would use features measured before the churn event, so measured performance may overstate real-world performance.
* **Small dataset (7,043 rows).** Validation (about 1,409 rows) is noisy; small model differences may not be meaningful. Confidence intervals are bootstrapped on the test set.
* **Single company / single period.** No evidence of generalisation elsewhere.
* **Explanations are associations.** Permutation importance and occlusion describe the model, not the world.
* **Fairness is not fully audited.** `gender`, `SeniorCitizen`, `Partner`, `Dependents` are in the default model inputs (as the project spec requires). They are never used in retention rules, but the model may still encode proxies. A group-wise error audit and an exclusion experiment are recommended future work.
* **Simulation is not measurement.** Simulated retained customers, net value and ROI follow mechanically from assumptions.
* **Not production-ready.** No live monitoring, retraining, access control or privacy review is implemented. The optional API/Docker extension is a local demo and its monitoring plan is a plan only.
* **Novelty:** none is claimed. All techniques used are standard.
