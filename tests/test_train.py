import json

import joblib
import numpy as np
import pandas as pd

from src import data, train
from src.features import get_feature_spec, split_xy


def test_metrics_file_has_required_columns(env):
    m = pd.read_csv(env["artifacts"] / "metrics.csv")
    for col in ["average_precision", "roc_auc", "precision", "recall", "f1", "brier",
                "log_loss", "tn", "fp", "fn", "tp"]:
        assert col in m.columns
    assert set(m["model"]) == {"dummy", "logistic_regression"}


def test_probabilities_between_0_and_1(env):
    model = joblib.load(env["artifacts"] / "baseline_logreg.joblib")
    spec = get_feature_spec()
    X, _ = split_xy(data.load_split("val", **env["kw"]), spec)
    p = model.predict_proba(X)[:, 1]
    assert p.min() >= 0 and p.max() <= 1


def test_saved_and_reloaded_model_predicts_identically(env, tmp_path):
    model = joblib.load(env["artifacts"] / "baseline_logreg.joblib")
    joblib.dump(model, tmp_path / "m.joblib")
    reloaded = joblib.load(tmp_path / "m.joblib")
    X, _ = split_xy(data.load_split("val", **env["kw"]), get_feature_spec())
    assert np.array_equal(model.predict_proba(X), reloaded.predict_proba(X))


def test_metadata_records_features_seed_versions_without_id(env):
    meta = json.loads((env["artifacts"] / "baseline_logreg_metadata.json").read_text())
    feats = meta["feature_order"]["numeric"] + meta["feature_order"]["categorical"]
    assert "customerID" not in feats and "Churn" not in feats
    assert meta["random_seed"] == 42
    assert "scikit-learn" in meta["package_versions"]


def test_single_row_and_batch_prediction(env):
    model = joblib.load(env["artifacts"] / "baseline_logreg.joblib")
    X, _ = split_xy(data.load_split("val", **env["kw"]), get_feature_spec())
    assert model.predict_proba(X.iloc[[0]]).shape == (1, 2)
    assert model.predict_proba(X).shape == (len(X), 2)


def test_baseline_is_reproducible(env):
    a = train.run_baseline(env["cfg"], env["artifacts"] / "rep_a", **env["kw"])
    b = train.run_baseline(env["cfg"], env["artifacts"] / "rep_b", **env["kw"])
    assert np.allclose(a["average_precision"], b["average_precision"])


def test_dummy_is_no_better_than_logistic_regression_on_signal(env):
    m = pd.read_csv(env["artifacts"] / "metrics.csv").set_index("model")
    assert m.loc["logistic_regression", "roc_auc"] > m.loc["dummy", "roc_auc"]


def test_select_model_prefers_simpler_within_tolerance():
    table = pd.DataFrame({"model": ["dummy", "logistic_regression", "random_forest"],
                          "val_average_precision": [0.27, 0.640, 0.645]})
    assert train.select_model(table, tolerance=0.01) == "logistic_regression"
    assert train.select_model(table, tolerance=0.001) == "random_forest"


def test_model_comparison_outputs(env):
    t = pd.read_csv(env["artifacts"] / "model_comparison.csv")
    assert t["selected"].sum() == 1 and "dummy" not in t[t["selected"]]["model"].values
    assert (env["figures"] / "val_pr_curves.png").exists()
    assert (env["reports"] / "model_comparison.md").exists()
