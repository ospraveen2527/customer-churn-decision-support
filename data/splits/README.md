# data/splits

Created by `python -m src.data`. Contains only customer IDs, one file per split:

| File | Share | Purpose |
|---|---|---|
| `train_ids.csv` | 60% | Fit preprocessing and models; 5-fold CV happens inside this set |
| `val_ids.csv`   | 20% | Compare models, choose calibration + threshold, run simulations |
| `test_ids.csv`  | 20% | LOCKED. Used once, in the final evaluation (`python -m src.evaluate`) |
| `split_summary.json` | - | Row counts and churn rate per split |

How the split is made: stratified on `Churn` (so each split keeps ~the same churn rate),
`random_state=42`, in two steps (60 / 40, then the 40 is halved). IDs are saved separately so the
split is reproducible and auditable. No scaling, encoding, imputation, feature selection or
resampling is done here - those happen inside sklearn Pipelines fitted on the training split only.

`src/data.py::load_split("test")` raises `PermissionError` unless the final evaluation passes
`allow_test=True`.
