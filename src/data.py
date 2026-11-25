"""Data loading, validation, cleaning, quality reporting and splitting.

Leakage policy
--------------
This module performs ONLY row-level, label-independent cleaning (type
conversion, whitespace stripping, invalid values -> NaN). It never scales,
encodes, imputes, selects features or resamples. Those steps are fitted inside
an sklearn Pipeline on the training split only (see src/features.py).

Documented cleaning policy
--------------------------
* Blank / non-numeric ``TotalCharges``  -> NaN (rows are KEPT, count is logged).
* Negative charges, negative / non-integer / implausible tenure -> NaN (logged).
* Unknown category values are reported but not altered.
* Duplicate ``customerID`` or invalid ``Churn`` labels -> hard error.
* ``Churn``: "Yes" -> 1, "No" -> 0.

Run: ``python -m src.data``
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CATEGORY_VALUES, CLEAN_FILENAME, ID_COLUMN, MAX_MONTHLY_CHARGES,
    MAX_TENURE_MONTHS, NEGATIVE_LABEL, NUMERIC_FEATURES, POSITIVE_LABEL,
    PROCESSED_DIR, RAW_DATA_URL, RAW_DIR, RAW_FILENAME, REPORTS_DIR,
    REQUIRED_COLUMNS, SPLITS_DIR, TARGET_COLUMN, load_config, setup_logging,
)

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


class DataValidationError(ValueError):
    """Raised when the dataset violates a hard requirement."""


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Read the raw Telco CSV exactly as stored (blank strings are preserved).

    Raises:
        FileNotFoundError: with download instructions if the file is missing.
    """
    csv_path = Path(path) if path else RAW_DIR / RAW_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {csv_path}. Download '{RAW_FILENAME}' from "
            f"{RAW_DATA_URL} and place it in data/raw/."
        )
    # keep_default_na=False: only truly empty cells are missing, so categories
    # such as "No" or "None" are never silently turned into NaN.
    df = pd.read_csv(csv_path, dtype={ID_COLUMN: str}, keep_default_na=False,
                     na_values=[""])
    logger.info("Loaded %d rows x %d columns from %s", *df.shape, csv_path)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Ensure every required column exists."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        logger.warning("Unexpected extra columns (ignored): %s", extra)


def validate_identifiers(df: pd.DataFrame) -> None:
    """customerID must be present and unique."""
    ids = df[ID_COLUMN]
    if ids.isna().any() or (ids.astype(str).str.strip() == "").any():
        raise DataValidationError("customerID contains missing or blank values.")
    n_dup = int(ids.duplicated().sum())
    if n_dup:
        raise DataValidationError(f"customerID is not unique: {n_dup} duplicates.")


def validate_target(df: pd.DataFrame) -> None:
    """Churn must contain only 'Yes' and 'No'."""
    values = set(df[TARGET_COLUMN].dropna().unique())
    bad = values - {POSITIVE_LABEL, NEGATIVE_LABEL}
    if bad or df[TARGET_COLUMN].isna().any():
        raise DataValidationError(
            f"Churn must contain only Yes/No (found extras: {sorted(map(str, bad))}, "
            f"missing: {int(df[TARGET_COLUMN].isna().sum())})."
        )


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the documented cleaning policy.

    Returns:
        (clean_df, cleaning_log). ``clean_df`` has Churn as 0/1 and numeric
        columns as floats/ints with NaN for invalid values.
    """
    validate_schema(raw)
    df = raw[REQUIRED_COLUMNS].copy()

    # Strip whitespace in text columns (blank " " becomes "").
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip().astype(object)
    df = df.replace({"": np.nan})

    validate_identifiers(df)
    validate_target(df)

    log: dict[str, Any] = {}

    # Numeric conversion with full transparency about what became NaN.
    raw_total = df["TotalCharges"]
    blank_mask = raw_total.isna()
    log["total_charges_blank_or_missing"] = int(blank_mask.sum())
    log["total_charges_blank_tenure_values"] = sorted(
        pd.to_numeric(df.loc[blank_mask, "tenure"], errors="coerce").dropna().unique().tolist()
    )
    for col in NUMERIC_FEATURES:
        converted = pd.to_numeric(df[col], errors="coerce")
        unparseable = int((converted.isna() & df[col].notna()).sum())
        log[f"{col}_unparseable_set_to_nan"] = unparseable
        df[col] = converted.astype(float)

    # Invalid values -> NaN (never silently dropped).
    invalid_tenure = (
        (df["tenure"] < 0)
        | (df["tenure"] > MAX_TENURE_MONTHS)
        | ((df["tenure"] % 1) != 0)
    )
    log["invalid_tenure_set_to_nan"] = int(invalid_tenure.sum())
    df.loc[invalid_tenure, "tenure"] = np.nan

    for col in ("MonthlyCharges", "TotalCharges"):
        bad = (df[col] < 0) | (df[col] > MAX_MONTHLY_CHARGES * MAX_TENURE_MONTHS)
        if col == "MonthlyCharges":
            bad = (df[col] < 0) | (df[col] > MAX_MONTHLY_CHARGES)
        log[f"{col}_invalid_set_to_nan"] = int(bad.sum())
        df.loc[bad, col] = np.nan

    # Target mapping: Yes -> 1, No -> 0.
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)

    # SeniorCitizen must be 0/1.
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")
    if not df["SeniorCitizen"].dropna().isin([0, 1]).all():
        raise DataValidationError("SeniorCitizen must be 0 or 1.")
    if df["SeniorCitizen"].notna().all():
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    log["rows_in"] = int(len(raw))
    log["rows_out"] = int(len(df))
    log["rows_dropped"] = int(len(raw) - len(df))
    logger.info("Cleaning complete: %s", log)
    return df.reset_index(drop=True), log


# --------------------------------------------------------------------------- #
# Data-quality report
# --------------------------------------------------------------------------- #
def build_quality_report(
    raw: pd.DataFrame,
    clean: pd.DataFrame | None = None,
    cleaning_log: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarise data quality. Computed from the raw data; never raises on bad data."""
    report: dict[str, Any] = {
        "n_rows": int(len(raw)),
        "n_columns": int(raw.shape[1]),
        "missing_columns": [c for c in REQUIRED_COLUMNS if c not in raw.columns],
    }
    present = [c for c in REQUIRED_COLUMNS if c in raw.columns]
    blank = lambda s: s.isna() | (s.astype("string").str.strip() == "")  # noqa: E731
    report["missing_or_blank_by_column"] = {
        c: int(blank(raw[c]).sum()) for c in present if blank(raw[c]).sum() > 0
    }
    report["duplicate_rows"] = int(raw.duplicated().sum())
    if ID_COLUMN in raw.columns:
        report["duplicate_customer_ids"] = int(raw[ID_COLUMN].duplicated().sum())
    if TARGET_COLUMN in raw.columns:
        report["target_distribution"] = {
            str(k): int(v) for k, v in raw[TARGET_COLUMN].value_counts(dropna=False).items()
        }
    numeric_checks: dict[str, int] = {}
    for col in NUMERIC_FEATURES:
        if col in raw.columns:
            nums = pd.to_numeric(raw[col], errors="coerce")
            numeric_checks[f"{col}_negative"] = int((nums < 0).sum())
    report["negative_values"] = numeric_checks
    if "tenure" in raw.columns:
        tenure = pd.to_numeric(raw["tenure"], errors="coerce")
        report["invalid_tenure"] = int(
            ((tenure < 0) | (tenure > MAX_TENURE_MONTHS) | (tenure % 1 != 0)).sum()
        )
    unexpected: dict[str, list[str]] = {}
    for col, allowed in CATEGORY_VALUES.items():
        if col in raw.columns:
            seen = set(raw[col].dropna().astype(str).str.strip())
            extra = seen - {str(a) for a in allowed}
            if extra:
                unexpected[col] = sorted(extra)
    report["unexpected_category_values"] = unexpected
    if cleaning_log is not None:
        report["cleaning_log"] = cleaning_log
    if clean is not None:
        report["missing_after_cleaning"] = {
            c: int(v) for c, v in clean.isna().sum().items() if v > 0
        }
        report["churn_rate_clean"] = float(clean[TARGET_COLUMN].mean())
    report["policy"] = (
        "Blank/invalid numeric values are set to NaN and kept; imputation happens "
        "later, inside the pipeline, using TRAINING data only."
    )
    return report


def write_quality_report(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    """Save the report as JSON and a readable Markdown file."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = ["# Data-Quality Report", "", "_Generated by `python -m src.data`._", ""]
    for key, value in report.items():
        lines.append(f"## {key}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(value, indent=2, default=str))
        lines.append("```")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Quality report written: %s, %s", json_path, md_path)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_data(
    df: pd.DataFrame,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Stratified train/validation/test split on the Churn label.

    Two stratified steps: (train) vs (validation+test), then validation vs test.
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-8:
        raise ValueError("train_size + val_size + test_size must equal 1.")
    train, rest = train_test_split(
        df, test_size=val_size + test_size, stratify=df[TARGET_COLUMN],
        random_state=seed, shuffle=True,
    )
    val, test = train_test_split(
        rest, test_size=test_size / (val_size + test_size),
        stratify=rest[TARGET_COLUMN], random_state=seed, shuffle=True,
    )
    splits = {"train": train.sort_index(), "val": val.sort_index(), "test": test.sort_index()}
    ids = [set(s[ID_COLUMN]) for s in splits.values()]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise DataValidationError("Split overlap detected - this should never happen.")
    return splits


def save_splits(splits: dict[str, pd.DataFrame], splits_dir: Path | str = SPLITS_DIR) -> None:
    """Save customer IDs of each split to separate files plus a small summary."""
    out = Path(splits_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for name, frame in splits.items():
        frame[[ID_COLUMN]].to_csv(out / f"{name}_ids.csv", index=False)
        summary[name] = {
            "rows": int(len(frame)),
            "churn_rate": float(frame[TARGET_COLUMN].mean()),
        }
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Split summary: %s", summary)


def load_split(
    name: str,
    processed_path: Path | str | None = None,
    splits_dir: Path | str = SPLITS_DIR,
    allow_test: bool = False,
) -> pd.DataFrame:
    """Load one cleaned split by joining saved IDs to the cleaned data.

    The test split is protected: it can only be loaded with ``allow_test=True``,
    which only src/evaluate.py does during the final locked evaluation.
    """
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split '{name}'. Use one of {SPLIT_NAMES}.")
    if name == "test" and not allow_test:
        raise PermissionError(
            "The test split is locked until the final evaluation (Stage 10)."
        )
    cpath = Path(processed_path) if processed_path else PROCESSED_DIR / CLEAN_FILENAME
    ipath = Path(splits_dir) / f"{name}_ids.csv"
    for p in (cpath, ipath):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run `python -m src.data` first.")
    clean = pd.read_csv(cpath, dtype={ID_COLUMN: str}, keep_default_na=False, na_values=[""])
    ids = pd.read_csv(ipath, dtype=str)[ID_COLUMN]
    return clean.set_index(ID_COLUMN, drop=False).loc[ids].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration / CLI
# --------------------------------------------------------------------------- #
def run_data_pipeline(
    raw_path: Path | str | None = None,
    processed_dir: Path | str = PROCESSED_DIR,
    splits_dir: Path | str = SPLITS_DIR,
    reports_dir: Path | str = REPORTS_DIR,
    cfg: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load -> validate -> clean -> report -> save -> split -> save splits."""
    cfg = cfg or load_config()
    raw = load_raw(raw_path)
    processed_dir, reports_dir = Path(processed_dir), Path(reports_dir)
    json_path = processed_dir / "data_quality_report.json"
    md_path = reports_dir / "data_quality_report.md"
    try:
        clean, log = clean_data(raw)
    except DataValidationError as exc:
        partial = build_quality_report(raw)
        partial["validation_error"] = str(exc)
        write_quality_report(partial, json_path, md_path)
        raise
    write_quality_report(build_quality_report(raw, clean, log), json_path, md_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(processed_dir / CLEAN_FILENAME, index=False)
    sp = cfg["split"]
    splits = split_data(clean, sp["train_size"], sp["val_size"], sp["test_size"],
                        cfg["project"]["random_seed"])
    save_splits(splits, splits_dir)
    return splits


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Validate, clean and split the Telco data.")
    parser.add_argument("--raw", default=None, help="Path to the raw CSV.")
    args = parser.parse_args()
    setup_logging()
    run_data_pipeline(args.raw)


if __name__ == "__main__":
    main()
