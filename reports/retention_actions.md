# Rule-Based Retention Recommendations (rules-v1)

Implemented in `src/policy.py::recommend_actions`. These are **suggestions for a human reviewer**. They are not
predictions that an action will prevent churn, and their effectiveness has not been measured.

## Rules
| ID | Condition | Suggested action |
|---|---|---|
| R1 | High risk band **and** customer value proxy >= cutoff | Route to a human retention specialist for a personal review |
| R2 | High risk band **and** value proxy < cutoff | Low-cost automated message summarising plan options |
| R3 | Medium/High risk and MonthlyCharges >= 80 | Offer a plan review to check for a lower-cost suitable plan |
| R4 | Medium/High risk and Contract = Month-to-month | Review contract options (no pressure to switch) |
| R5 | Medium/High risk and tenure <= 12 months | Onboarding assistance / check-in call |
| R6 | Medium/High risk and has internet service but TechSupport = No | Assess whether technical support is needed |
| R0 | none of the above | No proactive action; keep monitoring |

Risk bands: Low < 0.20 (config `low_risk_max`); High >= the frozen decision threshold; Medium in between. Cut-offs are in
`configs/default.yaml` (`retention_rules`) and are placeholders to tune with business input.

## Safeguards
* **Only** MonthlyCharges, tenure, Contract, TechSupport, InternetService, the churn probability and the value proxy are read.
  `gender`, `SeniorCitizen`, `Partner`, `Dependents` are never accessed (unit-tested).
* No manipulative tactics (no fake urgency, no hidden fees, no cancellation obstruction); actions are offers to review or help.
* Every recommendation carries an **audit record**: rules version, inputs used, probability, value proxy, band, rules fired,
  `sensitive_attributes_used: false`. `append_audit_log()` writes JSON lines; use a pseudonymous reference, not personal data.
* **Why human review matters:** the model can be wrong, probabilities are estimates, rules are blunt, and expensive or
  customer-facing decisions deserve context, empathy and accountability that rules cannot supply.
