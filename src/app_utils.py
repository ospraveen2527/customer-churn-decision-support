"""Logic shared by the Streamlit dashboard and the optional FastAPI service.

Keeping validation, scoring and export logic here (not in the UI) makes it
testable without a browser.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS_DIR, CATEGORY_VALUES, ID_COLUMN, MAX_MONTHLY_CHARGES, MAX_TENURE_MONTHS,
    NUMERIC_FEATURES, TARGET_COLUMN, load_config,
)
from src.explain import local_explanation, plain_language
from src.policy import CampaignAssumptions, recommend_actions, risk_band

logger = logging.getLogger(__name__)

MAX_BATCH_ROWS = 50_000
EXPORT_COLUMNS = [
    "customer_ref", "churn_probability", "risk_band", "flagged_at_threshold",
    "recommended_action", "model_version", "scoring_timestamp_utc",
]


class InputValidationError(ValueError):
    """Raised with a list of human-readable problems with user input."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass
class ModelBundle:
    """Everything needed to score customers."""

    model: Any
    metadata: dict[str, Any]
    frozen: dict[str, Any]
    cfg: dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.metadata["model_version"])

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def feature_columns(self) -> list[str]:
        order = self.metadata["feature_order"]
        return [*order["numeric"], *order["categorical"]]

    @property
    def assumptions(self) -> CampaignAssumptions:
        return CampaignAssumptions.from_config(self.cfg)


def load_bundle(artifacts_dir: Path | str = ARTIFACTS_DIR, cfg: dict[str, Any] | None = None) -> ModelBundle:
    """Load the saved model pipeline and its metadata.

    Raises:
        FileNotFoundError: with instructions if training has not been run.
    """
    artifacts_dir = Path(artifacts_dir)
    needed = ["final_model.joblib", "model_metadata.json", "frozen_config.json"]
    missing = [n for n in needed if not (artifacts_dir / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts {missing} in {artifacts_dir}. Run the training commands in the "
            "README (data -> baseline -> compare -> calibrate) first."
        )
    model = joblib.load(artifacts_dir / "final_model.joblib")
    meta = json.loads((artifacts_dir / "model_metadata.json").read_text(encoding="utf-8"))
    frozen = json.loads((artifacts_dir / "frozen_config.json").read_text(encoding="utf-8"))
    return ModelBundle(model, meta, frozen, cfg or load_config())


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_dataframe(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Validate and normalise user-provided customer rows.

    Required: every model feature. Optional: customerID (kept as reference) and
    Churn (kept as ``_actual_churn`` for simulations). Other columns are ignored.

    Raises:
        InputValidationError: listing every problem found.
    """
    errors: list[str] = []
    if len(df) == 0:
        raise InputValidationError(["The input has no rows."])
    if len(df) > MAX_BATCH_ROWS:
        raise InputValidationError([f"Too many rows ({len(df)}); the limit is {MAX_BATCH_ROWS}."])
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise InputValidationError([f"Missing required columns: {missing}"])

    out = pd.DataFrame(index=df.index)
    for col in feature_columns:
        s = df[col]
        if col in NUMERIC_FEATURES:
            raw = s.astype("string").str.strip().replace("", pd.NA)
            num = pd.to_numeric(raw, errors="coerce")
            bad = num.isna() & raw.notna()
            if bad.any():
                errors.append(f"{col}: non-numeric values in rows {_rows(bad)}.")
            if col == "TotalCharges":
                if (num < 0).any():
                    errors.append(f"{col}: negative values in rows {_rows(num < 0)}.")
            else:
                if num.isna().any() & raw.isna().any():
                    errors.append(f"{col}: missing values in rows {_rows(raw.isna())}.")
                upper = MAX_TENURE_MONTHS if col == "tenure" else MAX_MONTHLY_CHARGES
                outside = (num < 0) | (num > upper)
                if outside.any():
                    errors.append(f"{col}: values outside 0-{upper} in rows {_rows(outside)}.")
                if col == "tenure" and ((num % 1) != 0).any():
                    errors.append("tenure: must be a whole number of months.")
            out[col] = num.astype(float)
        else:
            allowed = CATEGORY_VALUES[col]
            if col == "SeniorCitizen":
                vals = pd.to_numeric(s, errors="coerce")
                bad = ~vals.isin([0, 1])
                out[col] = vals
            else:
                vals = s.astype("string").str.strip()
                bad = ~vals.isin([str(a) for a in allowed])
                out[col] = vals.astype(object)
            if bad.any():
                errors.append(f"{col}: invalid values in rows {_rows(bad)} (allowed: {allowed}).")
    if errors:
        raise InputValidationError(errors)
    if "SeniorCitizen" in out:
        out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)
    ref = df[ID_COLUMN].astype(str) if ID_COLUMN in df.columns else pd.Series(
        [f"row_{i + 1}" for i in range(len(df))], index=df.index)
    out["_customer_ref"] = ref
    if TARGET_COLUMN in df.columns:
        mapped = df[TARGET_COLUMN].map({"Yes": 1, "No": 0, 1: 1, 0: 0, "1": 1, "0": 0})
        if mapped.notna().all():
            out["_actual_churn"] = mapped.astype(int)
    return out


def _rows(mask: pd.Series, limit: int = 5) -> str:
    idx = [str(i + 1) for i in np.flatnonzero(mask.to_numpy())[:limit]]
    more = "..." if int(mask.sum()) > limit else ""
    return ", ".join(idx) + more


def validate_record(record: Mapping[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    """Validate a single customer dict; returns a one-row DataFrame."""
    return validate_dataframe(pd.DataFrame([dict(record)]), feature_columns)


# --------------------------------------------------------------------------- #
# Scoring and export
# --------------------------------------------------------------------------- #
def score_dataframe(bundle: ModelBundle, validated: pd.DataFrame) -> pd.DataFrame:
    """Score validated rows. Returns the EXPORT_COLUMNS (plus optional simulation helpers)."""
    cols = bundle.feature_columns
    X = validated[cols]
    proba = bundle.model.predict_proba(X)[:, 1]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rules = bundle.cfg["retention_rules"]
    actions, bands = [], []
    for (_, row), p in zip(X.iterrows(), proba):
        rec = recommend_actions(row.to_dict(), float(p), bundle.assumptions, rules, bundle.threshold)
        actions.append(rec.primary_action)
        bands.append(rec.risk_band)
    result = pd.DataFrame({
        "customer_ref": validated["_customer_ref"].to_numpy(),
        "churn_probability": np.round(proba, 4),
        "risk_band": bands,
        "flagged_at_threshold": proba >= bundle.threshold,
        "recommended_action": actions,
        "model_version": bundle.version,
        "scoring_timestamp_utc": now,
    })
    result["monthly_charges"] = X["MonthlyCharges"].to_numpy()
    if "_actual_churn" in validated:
        result["churned"] = validated["_actual_churn"].to_numpy()
    return result


def score_single(bundle: ModelBundle, record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and score one customer; include recommendation and explanation."""
    validated = validate_record(record, bundle.feature_columns)
    scored = score_dataframe(bundle, validated).iloc[0]
    X = validated[bundle.feature_columns]
    p = float(scored["churn_probability"])
    rec = recommend_actions(X.iloc[0].to_dict(), p, bundle.assumptions,
                            bundle.cfg["retention_rules"], bundle.threshold)
    expl = local_explanation(bundle.model, X, bundle.metadata["reference_values"],
                             bundle.cfg["explain"]["top_k"])
    return {
        "churn_probability": p,
        "risk_band": rec.risk_band,
        "flagged_at_threshold": bool(scored["flagged_at_threshold"]),
        "threshold": bundle.threshold,
        "recommendation": rec.to_dict(),
        "explanation_table": expl,
        "explanation_text": plain_language(expl),
        "model_version": bundle.version,
        "scoring_timestamp_utc": scored["scoring_timestamp_utc"],
    }


def sanitize_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralise spreadsheet formula injection in text cells before export."""
    out = df.copy()
    for col in out.select_dtypes(exclude="number").columns:
        out[col] = out[col].map(
            lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v)
    return out


def to_export_csv(scored: pd.DataFrame) -> bytes:
    """Predictions CSV (required columns only) as bytes for download. Nothing is written to disk."""
    return sanitize_for_csv(scored[EXPORT_COLUMNS]).to_csv(index=False).encode("utf-8")


def simulation_frame(scored: pd.DataFrame) -> pd.DataFrame:
    """Convert scored rows into the format expected by the campaign simulator."""
    out = pd.DataFrame({
        "customer_id": scored["customer_ref"].astype(str),
        "churn_probability": scored["churn_probability"],
        "monthly_charges": scored["monthly_charges"],
    })
    if "churned" in scored:
        out["churned"] = scored["churned"]
    return out
