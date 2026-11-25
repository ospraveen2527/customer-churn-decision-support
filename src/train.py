"""Baseline training (Stage 3) and model comparison / tuning (Stage 4).

All fitting uses the TRAINING split. Cross-validation is 5-fold stratified on
training data only. The validation split is used to compare/select models. The
test split is never loaded here (``data.load_split`` blocks it).

Run:
    python -m src.train baseline
    python -m src.train compare
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src import data as data_mod
from src.config import ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR, load_config, setup_logging
from src.evaluate import compute_metrics, plot_pr_curves, plot_roc_curves
from src.features import FeatureSpec, build_pipeline, get_feature_spec, split_xy

logger = logging.getLogger(__name__)

CV_SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "neg_brier_score": "neg_brier_score",
    "neg_log_loss": "neg_log_loss",
}
# Lower number = simpler model (used for the "prefer simpler" tie rule).
COMPLEXITY_RANK = {"dummy": 0, "logistic_regression": 1, "random_forest": 2, "xgboost": 3}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def package_versions() -> dict[str, str]:
    """Versions of the key packages (for reproducibility metadata)."""
    out = {"python": platform.python_version()}
    for pkg in ("pandas", "numpy", "scikit-learn", "joblib", "xgboost"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    return out


def make_estimator(name: str, seed: int, fixed: Mapping[str, Any] | None = None) -> BaseEstimator:
    """Factory for the supported estimators."""
    fixed = dict(fixed or {})
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=seed, **fixed)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **fixed)
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("xgboost is not installed (pip install xgboost).") from exc
        return XGBClassifier(random_state=seed, **fixed)
    raise ValueError(f"Unknown model '{name}'.")


def make_cv(cfg: Mapping[str, Any]) -> StratifiedKFold:
    """5-fold stratified CV with a fixed seed."""
    return StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True,
                           random_state=cfg["project"]["random_seed"])


def load_train_val(cfg: Mapping[str, Any], **load_kwargs: Any):
    """Load train and validation frames (never test) and the feature spec."""
    spec = get_feature_spec(cfg["features"]["exclude_columns"])
    train_df = data_mod.load_split("train", **load_kwargs)
    val_df = data_mod.load_split("val", **load_kwargs)
    X_tr, y_tr = split_xy(train_df, spec)
    X_va, y_va = split_xy(val_df, spec)
    return spec, X_tr, y_tr, X_va, y_va


def cv_summary(cv_result: Mapping[str, Any]) -> dict[str, float]:
    """Mean/std of cross-validation scores (brier/log-loss sign-flipped to be positive)."""
    out: dict[str, float] = {}
    for key in CV_SCORING:
        scores = np.asarray(cv_result[f"test_{key}"], dtype=float)
        sign = -1.0 if key.startswith("neg_") else 1.0
        name = key.replace("neg_", "")
        out[f"cv_{name}_mean"] = float(sign * scores.mean())
        out[f"cv_{name}_std"] = float(scores.std(ddof=1))
    return out


def save_model(
    model: Any, name: str, spec: FeatureSpec, cfg: Mapping[str, Any],
    artifacts_dir: Path, extra: Mapping[str, Any] | None = None,
) -> Path:
    """Save the model with joblib and a metadata JSON next to it."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / f"{name}.joblib"
    joblib.dump(model, model_path)
    meta = {
        "model_name": name,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": cfg["project"]["random_seed"],
        "feature_order": spec.to_dict(),
        "excluded_columns": list(cfg["features"]["exclude_columns"]),
        "identifier_excluded": "customerID",
        "package_versions": package_versions(),
        **(extra or {}),
    }
    (artifacts_dir / f"{name}_metadata.json").write_text(json.dumps(meta, indent=2, default=str),
                                                           encoding="utf-8")
    logger.info("Saved %s and metadata", model_path)
    return model_path


# --------------------------------------------------------------------------- #
# Stage 3: baselines
# --------------------------------------------------------------------------- #
def run_baseline(
    cfg: dict[str, Any] | None = None,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
    **load_kwargs: Any,
) -> pd.DataFrame:
    """Train Dummy and Logistic Regression; CV on train, evaluate on validation."""
    cfg = cfg or load_config()
    artifacts_dir = Path(artifacts_dir)
    seed = cfg["project"]["random_seed"]
    spec, X_tr, y_tr, X_va, y_va = load_train_val(cfg, **load_kwargs)
    cv = make_cv(cfg)
    threshold = cfg["baseline"]["decision_threshold"]
    rows: list[dict[str, Any]] = []

    for name in ("dummy", "logistic_regression"):
        pipe = build_pipeline(make_estimator(name, seed), spec)
        cv_res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=CV_SCORING, n_jobs=1)
        start = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        fit_seconds = time.perf_counter() - start
        proba = pipe.predict_proba(X_va)[:, 1]
        val = compute_metrics(y_va, proba, threshold)
        rows.append({"model": name, "split": "validation", **val, **cv_summary(cv_res),
                     "fit_seconds": fit_seconds})
        if name == "logistic_regression":
            save_model(pipe, "baseline_logreg", spec, cfg, artifacts_dir,
                       {"stage": "baseline", "decision_threshold": threshold,
                        "train_rows": int(len(X_tr))})
        logger.info("%s validation AP=%.4f ROC-AUC=%.4f", name, val["average_precision"], val["roc_auc"])

    metrics = pd.DataFrame(rows)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(artifacts_dir / "metrics.csv", index=False)
    return metrics


# --------------------------------------------------------------------------- #
# Stage 4: comparison and tuning
# --------------------------------------------------------------------------- #
def _model_plan(cfg: Mapping[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, list[Any]]]]:
    """(name, fixed params, param grid) for each model in the comparison."""
    plan: list[tuple[str, dict[str, Any], dict[str, list[Any]]]] = [("dummy", {}, {})]
    m = cfg["models"]
    plan.append(("logistic_regression", {}, dict(m["logistic_regression"]["param_grid"])))
    rf = m["random_forest"]
    plan.append(("random_forest", dict(rf.get("fixed", {})), dict(rf["param_grid"])))
    xgb = m.get("xgboost", {})
    if xgb.get("enabled"):
        plan.append(("xgboost", dict(xgb.get("fixed", {})), dict(xgb["param_grid"])))
    return plan


def select_model(table: pd.DataFrame, tolerance: float) -> str:
    """Pick the final candidate on VALIDATION average precision.

    Dummy is excluded. Any model within ``tolerance`` of the best validation AP
    counts as "similar"; among those, the simplest model wins.
    """
    cands = table[table["model"] != "dummy"]
    best_ap = cands["val_average_precision"].max()
    similar = cands[cands["val_average_precision"] >= best_ap - tolerance].copy()
    similar["rank"] = similar["model"].map(COMPLEXITY_RANK)
    similar = similar.sort_values(["rank", "val_average_precision"], ascending=[True, False])
    return str(similar.iloc[0]["model"])


def run_model_comparison(
    cfg: dict[str, Any] | None = None,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    reports_dir: Path | str = REPORTS_DIR,
    **load_kwargs: Any,
) -> pd.DataFrame:
    """Tune each model with training-only CV, compare on validation, select one."""
    cfg = cfg or load_config()
    artifacts_dir, figures_dir, reports_dir = Path(artifacts_dir), Path(figures_dir), Path(reports_dir)
    seed = cfg["project"]["random_seed"]
    spec, X_tr, y_tr, X_va, y_va = load_train_val(cfg, **load_kwargs)
    cv = make_cv(cfg)
    threshold = cfg["baseline"]["decision_threshold"]

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    val_probs: dict[str, np.ndarray] = {}
    for name, fixed, grid in _model_plan(cfg):
        pipe = build_pipeline(make_estimator(name, seed, fixed), spec)
        search = GridSearchCV(pipe, grid or {}, scoring=CV_SCORING, refit="average_precision",
                              cv=cv, n_jobs=1)
        start = time.perf_counter()
        search.fit(X_tr, y_tr)
        train_seconds = time.perf_counter() - start
        best = search.best_estimator_
        i = search.best_index_
        res = search.cv_results_
        cv_stats = {
            "cv_average_precision_mean": float(res["mean_test_average_precision"][i]),
            "cv_average_precision_std": float(res["std_test_average_precision"][i]),
            "cv_roc_auc_mean": float(res["mean_test_roc_auc"][i]),
            "cv_brier_mean": float(-res["mean_test_neg_brier_score"][i]),
            "cv_log_loss_mean": float(-res["mean_test_neg_log_loss"][i]),
        }
        proba = best.predict_proba(X_va)[:, 1]
        val = compute_metrics(y_va, proba, threshold)
        rows.append({
            "model": name,
            "best_params": json.dumps(search.best_params_, default=str),
            **cv_stats,
            **{f"val_{k}": v for k, v in val.items() if k in
               ("average_precision", "roc_auc", "precision", "recall", "f1", "brier", "log_loss")},
            "train_seconds_incl_cv": train_seconds,
            "n_configs_searched": int(len(res["params"])),
            "random_seed": seed,
        })
        fitted[name], val_probs[name] = best, proba
        logger.info("%s: CV AP=%.4f | val AP=%.4f", name, cv_stats["cv_average_precision_mean"],
                    val["average_precision"])

    table = pd.DataFrame(rows)
    chosen = select_model(table, cfg["models"]["selection_tolerance_ap"])
    table["selected"] = table["model"] == chosen
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(artifacts_dir / "model_comparison.csv", index=False)

    best_row = table[table["model"] == chosen].iloc[0]
    save_model(fitted[chosen], "selected_model", spec, cfg, artifacts_dir, {
        "stage": "model_selection", "selected": chosen,
        "best_params": json.loads(best_row["best_params"]),
        "selection_rule": f"highest validation AP; simpler model preferred within "
                          f"{cfg['models']['selection_tolerance_ap']} AP",
    })
    (artifacts_dir / "selection.json").write_text(json.dumps({
        "model_name": chosen, "best_params": json.loads(best_row["best_params"]),
        "tolerance_ap": cfg["models"]["selection_tolerance_ap"],
        "package_versions": package_versions(),
    }, indent=2), encoding="utf-8")

    curves = {n: (y_va, p) for n, p in val_probs.items()}
    plot_pr_curves(curves, figures_dir / "val_pr_curves.png", "Validation set: precision-recall curves")
    plot_roc_curves(curves, figures_dir / "val_roc_curves.png", "Validation set: ROC curves")
    write_comparison_report(table, chosen, cfg, reports_dir / "model_comparison.md")
    return table


def write_comparison_report(table: pd.DataFrame, chosen: str, cfg: Mapping[str, Any], path: Path) -> None:
    """Write reports/model_comparison.md from the measured table."""
    cols = ["model", "cv_average_precision_mean", "cv_average_precision_std", "val_average_precision",
            "val_roc_auc", "val_brier", "val_log_loss", "train_seconds_incl_cv", "best_params"]
    body = "\n".join(
        "| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |"
        for _, r in table.iterrows()
    )
    text = f"""# Model Comparison

Generated by `python -m src.train compare` (seed {cfg['project']['random_seed']}).

* Tuning: GridSearchCV, 5-fold stratified CV on the TRAINING split only.
* Selection: highest VALIDATION average precision; if a simpler model is within
  {cfg['models']['selection_tolerance_ap']} AP of the best, the simpler model is preferred.
* The test set was not loaded.

| {' | '.join(cols)} |
|{'---|' * len(cols)}
{body}

**Selected model:** `{chosen}`

Reading the table: a large gap between CV and validation AP would suggest the
validation split is unrepresentative or the search overfit; small gaps are expected.
Validation scores come from one 1,409-row split, so small differences between models
may be noise. Metrics at threshold 0.5 are shown only for comparability; the real
operating threshold is chosen in Stage 5.

Figures: `reports/figures/val_pr_curves.png`, `reports/figures/val_roc_curves.png`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    """CLI: `python -m src.train baseline|compare`."""
    parser = argparse.ArgumentParser(description="Train baseline models or run model comparison.")
    parser.add_argument("stage", choices=["baseline", "compare"])
    args = parser.parse_args()
    setup_logging()
    result = run_baseline() if args.stage == "baseline" else run_model_comparison()
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(result.drop(columns=["best_params"], errors="ignore").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
