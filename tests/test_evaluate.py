import json

import numpy as np
import pytest

from src import calibrate, evaluate


def test_compute_metrics_perfect_and_validation():
    y = np.array([0, 0, 1, 1])
    m = evaluate.compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert m["roc_auc"] == 1 and m["precision"] == 1 and m["recall"] == 1 and m["tp"] == 2
    with pytest.raises(ValueError):
        evaluate.compute_metrics(y, np.array([0.1, 0.2, 1.8, 0.9]))


def test_precision_recall_at_k():
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    r = evaluate.precision_recall_at_k(y, p, 2)
    assert r["precision_at_k"] == 0.5 and r["recall_at_k"] == 0.5


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    ci = evaluate.bootstrap_ci(y, p, 0.5, n_boot=200)
    point = evaluate.compute_metrics(y, p, 0.5)["roc_auc"]
    assert ci["roc_auc"][0] <= point <= ci["roc_auc"][1]


def test_threshold_rule_max_recall_subject_to_precision():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.4, 0.45, 0.3, 0.2, 0.1, 0.05])
    out = calibrate.select_threshold(y, p, "max_recall_min_precision", np.arange(0.05, 0.96, 0.05),
                                     min_precision=0.7)
    t = out["table"].set_index("threshold").loc[out["value"]]
    assert t["precision"] >= 0.7 and not out["fallback_used"]
    assert t["recall"] == out["table"][out["table"]["precision"] >= 0.7]["recall"].max()


def test_threshold_fallback_when_precision_unreachable():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.4, 0.6, 0.3, 0.7])
    out = calibrate.select_threshold(y, p, "max_recall_min_precision", np.arange(0.05, 0.96, 0.05), 0.99)
    assert out["fallback_used"]


def test_frozen_config_is_validation_only_and_complete(env):
    frozen = json.loads((env["artifacts"] / "frozen_config.json").read_text())
    for key in ("model_name", "calibration", "threshold", "campaign_policy",
                "simulation_assumptions_sha256", "model_version"):
        assert key in frozen
    assert 0.05 <= frozen["threshold"]["value"] <= 0.95
    assert "test set has not been used" in frozen["note"]


def test_final_evaluation_runs_once_then_locks(env):
    res = evaluate.final_test_evaluation(env["cfg"], env["artifacts"], env["reports"], env["figures"],
                                         **env["kw"])
    assert 0 <= res["metrics"]["roc_auc"] <= 1
    assert (env["reports"] / "final_evaluation.md").exists()
    with pytest.raises(RuntimeError, match="already evaluated"):
        evaluate.final_test_evaluation(env["cfg"], env["artifacts"], env["reports"], env["figures"],
                                       **env["kw"])


def test_final_evaluation_refuses_changed_assumptions(env, tmp_path):
    import copy
    cfg = copy.deepcopy(env["cfg"])
    cfg["simulation"]["save_rate"] = 0.99
    # remove lock in a copy so the assumption check (which runs first) is what triggers
    with pytest.raises(RuntimeError, match="differ from the frozen"):
        evaluate.final_test_evaluation(cfg, env["artifacts"], env["reports"], env["figures"], **env["kw"])
