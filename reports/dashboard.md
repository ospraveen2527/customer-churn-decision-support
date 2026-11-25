# Streamlit Dashboard

Run: `streamlit run app/streamlit_app.py` (after the training commands in the README have created `artifacts/final_model.joblib`).
If artifacts are missing the app shows instructions instead of crashing.

| Page | What it does |
|---|---|
| Project overview | Explains the four evidence levels (measured / explained / simulated / causal-not-claimed) |
| Single-customer prediction | Validated form -> probability, risk band, threshold flag, top factors in plain language, rule-based actions with rationale and audit record, model version + timestamp |
| Batch CSV prediction | Upload CSV, schema/value validation with row-numbered errors, scored table, downloadable `predictions.csv` (required columns only) |
| Explanation | Global permutation importance + local explanation for the last scored customer, with a no-causation notice |
| Campaign simulator | Sliders for margin, horizon, save rate, cost, budget; compares three policies on the uploaded/scored batch; SIMULATED banner |
| Model evaluation | Validation tables/figures; test metrics appear only after the final evaluation has been run |
| Limitations | Plain-language limitations |

Privacy and safety: uploads are processed in memory only; nothing is written to disk or sent anywhere; exported CSV text
cells are protected against spreadsheet formula injection; no customer communications are sent.

Testing note: validation, scoring and export logic live in `src/app_utils.py` and are covered by `tests/test_predictions.py`.
The Streamlit UI itself was syntax-checked but needs to be run locally by you; please send screenshots/errors if anything looks off.
