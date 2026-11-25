"""Central configuration: paths, column definitions, and YAML config loading.

Why this file exists
--------------------
Keeping column names, allowed category values and paths in ONE place means the
data, feature, dashboard and API code can never drift apart.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

RAW_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEAN_FILENAME = "telco_clean.csv"
RAW_DATA_URL = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"  # "Yes" in the raw file -> 1 after cleaning
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"

NUMERIC_FEATURES: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

REQUIRED_COLUMNS: list[str] = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

_YES_NO = ["Yes", "No"]
_YES_NO_NOINTERNET = ["Yes", "No", "No internet service"]

# Allowed values of every categorical feature in the original dataset.
CATEGORY_VALUES: dict[str, list[Any]] = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": _YES_NO_NOINTERNET,
    "OnlineBackup": _YES_NO_NOINTERNET,
    "DeviceProtection": _YES_NO_NOINTERNET,
    "TechSupport": _YES_NO_NOINTERNET,
    "StreamingTV": _YES_NO_NOINTERNET,
    "StreamingMovies": _YES_NO_NOINTERNET,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

# Attributes that describe the person rather than the service. They are never
# used to decide who receives retention treatment (see src/policy.py).
SENSITIVE_ATTRIBUTES: list[str] = ["gender", "SeniorCitizen", "Partner", "Dependents"]

# Plausibility ranges used for validation (not statistical outlier rules).
MAX_TENURE_MONTHS = 120
MAX_MONTHLY_CHARGES = 500.0


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
_REQUIRED_KEYS = [
    "project", "split", "cv", "features", "baseline", "models",
    "calibration", "threshold", "simulation", "retention_rules", "explain",
]


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate the YAML configuration.

    Args:
        path: Optional alternative config path. Defaults to configs/default.yaml.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if required sections are missing or split sizes are invalid.
    """
    cfg_path = Path(path) if path else CONFIG_PATH
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {cfg_path}") from exc
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Check that required sections exist and the split sizes sum to 1."""
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config is missing sections: {missing}")
    split = cfg["split"]
    total = split["train_size"] + split["val_size"] + split["test_size"]
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split sizes must sum to 1.0, got {total}.")


def config_hash(obj: Any) -> str:
    """Stable SHA-256 hash of any JSON-serialisable object (used to freeze assumptions)."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging once (safe to call repeatedly)."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
