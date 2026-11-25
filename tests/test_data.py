import numpy as np
import pandas as pd
import pytest

from src import data
from src.config import REQUIRED_COLUMNS


def test_required_columns_exist_after_load(raw_csv):
    df = data.load_raw(raw_csv)
    assert set(REQUIRED_COLUMNS) <= set(df.columns)


def test_missing_column_raises(synthetic_df):
    with pytest.raises(data.DataValidationError, match="Missing required columns"):
        data.clean_data(synthetic_df.drop(columns=["Contract"]))


def test_customer_id_is_unique_and_duplicates_rejected(synthetic_df):
    clean, _ = data.clean_data(synthetic_df)
    assert clean["customerID"].is_unique
    dup = pd.concat([synthetic_df, synthetic_df.iloc[[0]]])
    with pytest.raises(data.DataValidationError, match="not unique"):
        data.clean_data(dup)


def test_churn_mapping_yes_1_no_0(synthetic_df):
    clean, _ = data.clean_data(synthetic_df)
    expected = (synthetic_df["Churn"] == "Yes").astype(int).to_numpy()
    assert (clean["Churn"].to_numpy() == expected).all()
    assert set(clean["Churn"].unique()) <= {0, 1}


def test_invalid_churn_value_rejected(synthetic_df):
    bad = synthetic_df.copy()
    bad.loc[0, "Churn"] = "Maybe"
    with pytest.raises(data.DataValidationError, match="Yes/No"):
        data.clean_data(bad)


def test_total_charges_numeric_and_blanks_kept_as_nan(synthetic_df):
    n_blank = int((synthetic_df["TotalCharges"].str.strip() == "").sum())
    clean, log = data.clean_data(synthetic_df)
    assert pd.api.types.is_float_dtype(clean["TotalCharges"])
    assert int(clean["TotalCharges"].isna().sum()) == n_blank == log["total_charges_blank_or_missing"]
    assert len(clean) == len(synthetic_df)           # nothing silently dropped
    assert log["total_charges_blank_tenure_values"] == [0.0]


def test_negative_and_invalid_values_become_nan_and_are_logged(synthetic_df):
    bad = synthetic_df.copy()
    bad.loc[1, "MonthlyCharges"] = -5.0
    bad.loc[2, "tenure"] = -3
    clean, log = data.clean_data(bad)
    assert np.isnan(clean.loc[1, "MonthlyCharges"]) and np.isnan(clean.loc[2, "tenure"])
    assert log["MonthlyCharges_invalid_set_to_nan"] == 1
    assert log["invalid_tenure_set_to_nan"] == 1
    assert len(clean) == len(bad)


def test_split_has_no_overlap_and_correct_proportions(synthetic_df):
    clean, _ = data.clean_data(synthetic_df)
    s = data.split_data(clean, seed=42)
    ids = {k: set(v["customerID"]) for k, v in s.items()}
    assert not ids["train"] & ids["val"] and not ids["train"] & ids["test"] and not ids["val"] & ids["test"]
    assert sum(len(v) for v in s.values()) == len(clean)
    n = len(clean)
    assert abs(len(s["train"]) / n - 0.6) < 0.01
    assert abs(len(s["val"]) / n - 0.2) < 0.01
    assert abs(len(s["test"]) / n - 0.2) < 0.01
    rates = [v["Churn"].mean() for v in s.values()]
    assert max(rates) - min(rates) < 0.01             # stratified


def test_split_is_reproducible(synthetic_df):
    clean, _ = data.clean_data(synthetic_df)
    a, b = data.split_data(clean, seed=42), data.split_data(clean, seed=42)
    assert list(a["val"]["customerID"]) == list(b["val"]["customerID"])


def test_pipeline_writes_files_and_report(env):
    for name in ("train_ids.csv", "val_ids.csv", "test_ids.csv", "split_summary.json"):
        assert (env["splits"] / name).exists()
    assert (env["processed"] / "telco_clean.csv").exists()
    assert (env["processed"] / "data_quality_report.json").exists()
    assert (env["reports"] / "data_quality_report.md").exists()


def test_test_split_is_locked_until_final_evaluation(env):
    with pytest.raises(PermissionError):
        data.load_split("test", **env["kw"])
    assert len(data.load_split("test", allow_test=True, **env["kw"])) > 0


def test_quality_report_contents(raw_csv):
    raw = data.load_raw(raw_csv)
    report = data.build_quality_report(raw)
    assert report["duplicate_customer_ids"] == 0
    assert report["missing_or_blank_by_column"]["TotalCharges"] == int((raw["tenure"] == 0).sum())
    assert report["unexpected_category_values"] == {}
