import io

import pandas as pd
import pytest

from src import app_utils, data
from src.app_utils import InputValidationError

GOOD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 5,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5,
    "TotalCharges": "",
}


@pytest.fixture(scope="module")
def bundle(env):
    return app_utils.load_bundle(env["artifacts"], env["cfg"])


def test_model_loads(bundle):
    assert bundle.version and 0 < bundle.threshold < 1
    assert "customerID" not in bundle.feature_columns


def test_missing_artifacts_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="training commands"):
        app_utils.load_bundle(tmp_path)


def test_single_prediction(bundle):
    r = app_utils.score_single(bundle, GOOD)
    assert 0 <= r["churn_probability"] <= 1
    assert r["risk_band"] in {"Low", "Medium", "High"}
    assert r["model_version"] == bundle.version and r["scoring_timestamp_utc"]
    assert r["recommendation"]["requires_human_review"]
    assert 1 <= len(r["explanation_text"]) <= 5


def test_batch_prediction_and_export_columns(bundle, env):
    df = data.load_split("val", **env["kw"]).drop(columns=["Churn"]).head(50)
    scored = app_utils.score_dataframe(bundle, app_utils.validate_dataframe(df, bundle.feature_columns))
    assert len(scored) == 50
    exported = pd.read_csv(io.BytesIO(app_utils.to_export_csv(scored)))
    assert list(exported.columns) == app_utils.EXPORT_COLUMNS
    assert exported["churn_probability"].between(0, 1).all()


def test_invalid_inputs_produce_clear_errors(bundle):
    cols = bundle.feature_columns
    with pytest.raises(InputValidationError, match="Contract"):
        app_utils.validate_record({**GOOD, "Contract": "Weekly"}, cols)
    with pytest.raises(InputValidationError, match="tenure"):
        app_utils.validate_record({**GOOD, "tenure": -4}, cols)
    with pytest.raises(InputValidationError, match="MonthlyCharges"):
        app_utils.validate_record({**GOOD, "MonthlyCharges": "abc"}, cols)
    with pytest.raises(InputValidationError, match="TotalCharges"):
        app_utils.validate_record({**GOOD, "TotalCharges": -1}, cols)


def test_uploaded_csv_schema_validated(bundle):
    with pytest.raises(InputValidationError, match="Missing required columns"):
        app_utils.validate_dataframe(pd.DataFrame([{"tenure": 1}]), bundle.feature_columns)
    with pytest.raises(InputValidationError, match="no rows"):
        app_utils.validate_dataframe(pd.DataFrame(columns=bundle.feature_columns), bundle.feature_columns)


def test_blank_total_charges_is_accepted(bundle):
    out = app_utils.validate_record(GOOD, bundle.feature_columns)
    assert out["TotalCharges"].isna().all()


def test_csv_formula_injection_is_neutralised():
    df = pd.DataFrame({"customer_ref": ["=cmd()", "ok"], "x": [1, 2]})
    assert app_utils.sanitize_for_csv(df)["customer_ref"].iloc[0] == "'=cmd()"
