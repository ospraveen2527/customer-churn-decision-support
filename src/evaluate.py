"""Metrics, curves, bootstrap intervals and the locked final test evaluation.

The test set is touched in exactly one place: ``final_test_evaluation`` below,
which requires a frozen configuration and refuses to run twice (lock file).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from src.config import (
    ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR, config_hash, load_config, setup_logging,
)
from src.plotting import new_figure, save_figure

logger = logging.getLogger(__name__)

METRIC_KEYS = ["average_precision", "roc_auc", "precision", "recall", "f1", "brier", "log_loss"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: Any, proba: Any, threshold: float = 0.5) -> dict[str, float]:
    """All required metrics for binary churn predictions.

    Args:
        y_true: 0/1 labels.
        proba: predicted probability of churn (must be within [0, 1]).
        threshold: probability cut-off for the binary decision.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and proba must have the same shape.")
    if p.size == 0 or np.isnan(p).any() or p.min() < 0 or p.max() > 1:
        raise ValueError("Probabilities must be non-empty, non-NaN and within [0, 1].")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_table(y_true: Any, proba: Any, thresholds: Any) -> pd.DataFrame:
    """Precision / recall / F1 / flagged share across candidate thresholds."""
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, proba, float(t))
        rows.append({
            "threshold": round(float(t), 4), "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "n_flagged": m["tp"] + m["fp"],
            "flagged_rate": (m["tp"] + m["fp"]) / len(np.asarray(y_true)),
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        })
    return pd.DataFrame(rows)


def precision_recall_at_k(y_true: Any, proba: Any, k: int) -> dict[str, float]:
    """Precision@k and Recall@k when the k highest-probability customers are contacted."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    k = int(max(0, min(k, len(y))))
    if k == 0:
        return {"k": 0, "precision_at_k": float("nan"), "recall_at_k": 0.0}
    top = np.argsort(-p, kind="mergesort")[:k]
    hits = int(y[top].sum())
    return {"k": k, "precision_at_k": hits / k,
            "recall_at_k": hits / max(1, int(y.sum()))}


def bootstrap_ci(
    y_true: Any, proba: Any, threshold: float, n_boot: int = 1000, seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap confidence intervals for the headline metrics."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = compute_metrics(y[idx], p[idx], threshold)
        for k in METRIC_KEYS:
            draws[k].append(m[k])
    return {
        k: (float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2))))
        for k, v in draws.items() if v
    }


# --------------------------------------------------------------------------- #
# Plots (each has a title, axis labels)
# --------------------------------------------------------------------------- #
def plot_pr_curves(curves: dict[str, tuple[Any, Any]], path: Path | str, title: str) -> Path:
    """Precision-recall curves for several models. ``curves`` maps name -> (y, proba)."""
    fig, ax = new_figure()
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y, p):.3f})")
    base = float(np.mean(next(iter(curves.values()))[0]))
    ax.axhline(base, color="grey", linestyle="--", label=f"Base rate {base:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8)
    return save_figure(fig, path)


def plot_roc_curves(curves: dict[str, tuple[Any, Any]], path: Path | str, title: str) -> Path:
    """ROC curves for several models."""
    fig, ax = new_figure()
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Chance")
    ax.set_title(title)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8)
    return save_figure(fig, path)


def plot_confusion_matrix(m: dict[str, float], path: Path | str, title: str) -> Path:
    """Confusion matrix heat-map from a metrics dict."""
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = new_figure((4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], ["Stay (0)", "Churn (1)"])
    ax.set_yticks([0, 1], ["Stay (0)", "Churn (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return save_figure(fig, path)


def plot_reliability(
    curves: dict[str, tuple[Any, Any]], path: Path | str, title: str, n_bins: int = 10
) -> Path:
    """Reliability diagram: mean predicted probability vs observed churn rate."""
    fig, ax = new_figure()
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Perfect calibration")
    for name, (y, p) in curves.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel("Mean predicted churn probability (per bin)")
    ax.set_ylabel("Observed churn rate (per bin)")
    ax.legend(fontsize=8)
    return save_figure(fig, path)


# --------------------------------------------------------------------------- #
# Frozen configuration + locked final evaluation
# --------------------------------------------------------------------------- #
def load_frozen_config(artifacts_dir: Path | str = ARTIFACTS_DIR) -> dict[str, Any]:
    """Load artifacts/frozen_config.json (written by `python -m src.calibrate`)."""
    path = Path(artifacts_dir) / "frozen_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Complete Stages 4-5 (model selection, calibration, "
            "threshold) before the final evaluation."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def final_test_evaluation(
    cfg: dict[str, Any] | None = None,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
    reports_dir: Path | str = REPORTS_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    processed_path: Path | str | None = None,
    splits_dir: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate the FROZEN model on the test set exactly once.

    Nothing computed here may be used to change the model, calibration method,
    threshold, campaign policy or assumptions.
    """
    from src import data as data_mod  # local import keeps module dependencies light
    from src.policy import CampaignAssumptions, compare_policies, plot_policy_comparison

    cfg = cfg or load_config()
    artifacts_dir, reports_dir, figures_dir = Path(artifacts_dir), Path(reports_dir), Path(figures_dir)
    frozen = load_frozen_config(artifacts_dir)

    assumptions = CampaignAssumptions.from_config(cfg)
    if config_hash(assumptions.to_dict()) != frozen["simulation_assumptions_sha256"]:
        raise RuntimeError(
            "Simulation assumptions in the config differ from the frozen ones. "
            "Do not change assumptions after freezing; restore the config or re-run Stage 5."
        )
    lock = artifacts_dir / "test_evaluation_lock.json"
    if lock.exists() and not force:
        raise RuntimeError(
            f"The test set was already evaluated ({lock}). Re-running would turn the test "
            "set into a tuning set. Use --force only to reproduce results without changes."
        )

    from src.features import get_feature_spec, split_xy
    spec = get_feature_spec(cfg["features"]["exclude_columns"])
    kwargs: dict[str, Any] = {"allow_test": True}
    if processed_path:
        kwargs["processed_path"] = processed_path
    if splits_dir:
        kwargs["splits_dir"] = splits_dir
    test_df = data_mod.load_split("test", **kwargs)
    X_test, y_test = split_xy(test_df, spec)

    model = joblib.load(artifacts_dir / "final_model.joblib")
    proba = model.predict_proba(X_test)[:, 1]
    threshold = float(frozen["threshold"]["value"])
    metrics = compute_metrics(y_test, proba, threshold)
    ci = bootstrap_ci(y_test, proba, threshold, n_boot=1000, seed=cfg["project"]["random_seed"])

    k = int(np.floor(assumptions.total_budget / assumptions.contact_cost)) if assumptions.contact_cost else len(y_test)
    at_budget = precision_recall_at_k(y_test, proba, k)

    sim_df = pd.DataFrame({
        "customer_id": test_df["customerID"].astype(str),
        "churn_probability": proba,
        "monthly_charges": test_df["MonthlyCharges"].fillna(X_test["MonthlyCharges"].median()),
        "churned": y_test.to_numpy(),
    })
    campaign = compare_policies(sim_df, assumptions, cfg["simulation"]["n_random_seeds"],
                                cfg["project"]["random_seed"])

    figs = {
        "pr": plot_pr_curves({"final model": (y_test, proba)}, figures_dir / "final_pr_curve.png",
                             "Test set: precision-recall curve"),
        "roc": plot_roc_curves({"final model": (y_test, proba)}, figures_dir / "final_roc_curve.png",
                               "Test set: ROC curve"),
        "cm": plot_confusion_matrix(metrics, figures_dir / "final_confusion_matrix.png",
                                    f"Test set: confusion matrix (threshold={threshold:.2f})"),
        "cal": plot_reliability({"final model": (y_test, proba)}, figures_dir / "final_calibration_curve.png",
                                "Test set: reliability diagram"),
        "campaign": plot_policy_comparison(campaign, figures_dir / "final_campaign_comparison.png"),
    }

    results = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": frozen.get("model_version"),
        "n_test": int(len(y_test)),
        "test_churn_rate": float(y_test.mean()),
        "metrics": metrics,
        "confidence_intervals_95": {k_: list(v) for k_, v in ci.items()},
        "at_budget": at_budget,
        "campaign_simulation": campaign.drop(columns=["label"]).to_dict(orient="records"),
        "figures": {k_: str(v) for k_, v in figs.items()},
    }
    (artifacts_dir / "test_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lock.write_text(json.dumps({"evaluated_at_utc": results["evaluated_at_utc"]}), encoding="utf-8")
    write_final_report(results, frozen, assumptions, reports_dir / "final_evaluation.md")
    logger.info("Final evaluation complete. Results saved; test set is now spent.")
    return results


def write_final_report(
    results: dict[str, Any], frozen: dict[str, Any], a: Any, path: Path
) -> None:
    """Write reports/final_evaluation.md from the measured numbers."""
    m, ci = results["metrics"], results["confidence_intervals_95"]
    rows = "\n".join(
        f"| {k} | {m[k]:.4f} | {ci[k][0]:.4f} - {ci[k][1]:.4f} |" for k in METRIC_KEYS if k in ci
    )
    camp = pd.DataFrame(results["campaign_simulation"])
    camp_md = camp.round(3).to_markdown(index=False) if _has_tabulate() else camp.round(3).to_string(index=False)
    text = f"""# Final Evaluation (locked test set)

Generated by `python -m src.evaluate` at {results['evaluated_at_utc']}.

**Frozen before testing:** model `{frozen['model_name']}`, calibration `{frozen['calibration']['method']}`
(applied: {frozen['calibration']['applied']}), threshold `{frozen['threshold']['value']}`
(rule: {frozen['threshold']['rule']}), campaign policy `{frozen['campaign_policy']}`,
model version `{results['model_version']}`.

Test rows: {results['n_test']} (churn rate {results['test_churn_rate']:.3f}).

## 1. Measured model performance (test set, 95% bootstrap CI)

| Metric | Value | 95% CI |
|---|---|---|
{rows}

Confusion matrix at the frozen threshold: TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']}.

At the campaign budget (top {results['at_budget']['k']} customers by probability):
Precision@budget = {results['at_budget']['precision_at_k']:.4f},
Recall@budget = {results['at_budget']['recall_at_k']:.4f}.

Figures: `reports/figures/final_pr_curve.png`, `final_roc_curve.png`,
`final_confusion_matrix.png`, `final_calibration_curve.png`, `final_campaign_comparison.png`.

## 2. Simulated campaign outcomes (ASSUMPTION-BASED, NOT MEASURED)

Assumptions (illustrative placeholders): {a.to_dict()}

{camp_md}

These are simulations. They do not show revenue actually saved or churn actually
reduced. "Historical churners targeted" counts customers who really churned in
the data; whether an offer would have kept them is unknown.

## 3. What this evaluation does NOT show

* Causal retention impact (needs a randomised experiment).
* Performance on a different company, period or customer base.
* Fairness across groups (not yet audited; see the model card).
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> None:
    """CLI: `python -m src.evaluate [--force]` runs the locked final evaluation."""
    parser = argparse.ArgumentParser(description="Final locked test-set evaluation.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even though the lock exists (reproduction only).")
    args = parser.parse_args()
    setup_logging()
    final_test_evaluation(force=args.force)


if __name__ == "__main__":
    main()
