import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src import data
from src.features import (build_pipeline, build_reference_values, get_feature_spec, split_xy)


@pytest.fixture(scope="module")
def frames(synthetic_df):
    clean, _ = data.clean_data(synthetic_df)
    s = data.split_data(clean)
    spec = get_feature_spec()
    return spec, split_xy(s["train"], spec), split_xy(s["val"], spec)


def test_customer_id_and_target_not_features():
    spec = get_feature_spec()
    assert "customerID" not in spec.all and "Churn" not in spec.all
    assert spec.numeric == ("tenure", "MonthlyCharges", "TotalCharges")
    assert len(spec.all) == 19


def test_exclude_columns_option():
    spec = get_feature_spec(["gender", "Partner"])
    assert "gender" not in spec.all and "Partner" not in spec.all


def test_preprocessing_fitted_on_train_only(frames, synthetic_df):
    spec, (X_tr, y_tr), (X_va, _) = frames
    pipe = build_pipeline(LogisticRegression(max_iter=1000), spec).fit(X_tr, y_tr)
    num = pipe.named_steps["preprocess"].named_transformers_["num"]
    assert np.isclose(num.named_steps["scale"].mean_[0], X_tr["tenure"].mean(), atol=1e-6)
    assert not np.isclose(num.named_steps["scale"].mean_[0],
                          np.concatenate([X_tr["tenure"], X_va["tenure"]]).mean(), atol=1e-9)
    assert np.isclose(num.named_steps["impute"].statistics_[2], X_tr["TotalCharges"].median())


def test_unseen_category_is_ignored_and_nan_imputed(frames):
    spec, (X_tr, y_tr), (X_va, _) = frames
    pipe = build_pipeline(LogisticRegression(max_iter=1000), spec).fit(X_tr, y_tr)
    row = X_va.iloc[[0]].copy()
    row["Contract"] = "Brand new contract"
    row["TotalCharges"] = np.nan
    p = pipe.predict_proba(row)[:, 1]
    assert 0 <= p[0] <= 1


def test_reference_values_cover_all_features(frames):
    spec, (X_tr, _), _ = frames
    ref = build_reference_values(X_tr, spec)
    assert set(ref) == set(spec.all)
