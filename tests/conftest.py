"""Shared fixtures. Tests use SYNTHETIC Telco-like data (never the real dataset)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import REQUIRED_COLUMNS, load_config


def make_synthetic_telco(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Create a fake dataset with the same 21 columns and quirks as the Telco file.

    Includes blank TotalCharges (tenure == 0) exactly like the real file. The data are
    random: any metric computed on them is meaningless and must never be reported.
    """
    rng = np.random.default_rng(seed)
    yn = lambda p=0.5: np.where(rng.random(n) < p, "Yes", "No")  # noqa: E731
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    phone = yn(0.9)

    def addon() -> np.ndarray:
        return np.where(internet == "No", "No internet service", yn(0.4))

    tenure = rng.integers(0, 73, n)
    tenure[:8] = 0
    monthly = np.round(rng.uniform(18, 115, n), 2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    logit = (-0.8 + 1.3 * (contract == "Month-to-month") - 0.03 * tenure
             + 0.012 * (monthly - 65) + 0.5 * (internet == "Fiber optic"))
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    total = np.round(tenure * monthly, 2).astype(str).astype(object)
    total[tenure == 0] = " "
    df = pd.DataFrame({
        "customerID": [f"{i:04d}-SYNTH" for i in range(n)],
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": yn(), "Dependents": yn(0.3), "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", yn(0.4)),
        "InternetService": internet,
        "OnlineSecurity": addon(), "OnlineBackup": addon(), "DeviceProtection": addon(),
        "TechSupport": addon(), "StreamingTV": addon(), "StreamingMovies": addon(),
        "Contract": contract, "PaperlessBilling": yn(0.6),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges": monthly, "TotalCharges": total, "Churn": churn,
    })
    return df[REQUIRED_COLUMNS]


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_telco()


@pytest.fixture()
def raw_csv(tmp_path: Path, synthetic_df: pd.DataFrame) -> Path:
    path = tmp_path / "raw.csv"
    synthetic_df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def fast_cfg() -> dict:
    """Default config shrunk so the whole pipeline runs in seconds."""
    cfg = load_config()
    cfg["models"]["random_forest"] = {"fixed": {"n_estimators": 25, "n_jobs": 1},
                                      "param_grid": {"model__max_depth": [3, None]}}
    cfg["models"]["logistic_regression"]["param_grid"] = {"model__C": [0.1, 1.0]}
    cfg["simulation"]["n_random_seeds"] = 3
    cfg["explain"]["n_repeats"] = 2
    return cfg


@pytest.fixture(scope="session")
def env(tmp_path_factory, fast_cfg) -> dict:
    """Run the whole pipeline (Stages 1-7) once on synthetic data in a temp folder."""
    from src import calibrate, data, explain, policy, train

    root = tmp_path_factory.mktemp("project")
    raw = root / "raw.csv"
    make_synthetic_telco().to_csv(raw, index=False)
    dirs = {k: root / k for k in ("processed", "splits", "reports", "artifacts", "figures")}
    for d in dirs.values():
        d.mkdir()
    data.run_data_pipeline(raw, dirs["processed"], dirs["splits"], dirs["reports"], fast_cfg)
    kw = {"processed_path": dirs["processed"] / "telco_clean.csv", "splits_dir": dirs["splits"]}
    train.run_baseline(fast_cfg, dirs["artifacts"], **kw)
    train.run_model_comparison(fast_cfg, dirs["artifacts"], dirs["figures"], dirs["reports"], **kw)
    calibrate.run_calibration(fast_cfg, dirs["artifacts"], dirs["figures"], dirs["reports"], **kw)
    explain.run_explainability(fast_cfg, dirs["artifacts"], dirs["figures"], dirs["reports"], **kw)
    policy.run_campaign_simulation(fast_cfg, dirs["artifacts"], dirs["figures"], dirs["reports"], **kw)
    return {"root": root, "kw": kw, "cfg": fast_cfg, **dirs}
