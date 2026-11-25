"""Probability calibration (Stage 5), threshold selection, and freezing.

Calibration is done with ``CalibratedClassifierCV`` wrapped AROUND the whole
pipeline (preprocessing + model). Each internal calibration fold therefore
re-fits imputation, scaling and encoding on that fold's training part only, so no
information leaks from the held-out part of the fold.

Run: ``python -m src.calibrate``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from src.config import (
    ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR, config_hash, load_config, setup_logging,
)
from src.evaluate import compute_metrics, plot_reliability, threshold_table
from src.features import build_reference_values
from src.plotting import new_figure, save_figure
from src.policy import CampaignAssumptions
from src.train import load_train_val, package_versions

logger = logging.getLogger(__name__)


def build_calibrated(pipeline: Any, method: str, cv_folds: int, seed: int) -> CalibratedClassifierCV:
    """Wrap an UNFITTED full pipeline so preprocessing is refit inside every calibration fold."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(estimator=clone(pipeline), method=method, cv=cv)


def select_threshold(
    y_true: Any, proba: Any, rule: str, grid: np.ndarray,
    min_precision: float = 0.5, fn_cost: float = 5.0, fp_cost: float = 1.0,
) -> dict[str, Any]:
    """Choose an operating threshold on VALIDATION data using a predeclared rule.

    Rules:
        max_recall_min_precision: highest recall among thresholds with precision >= min_precision
            (ties -> higher precision, then higher threshold). If none qualifies, fall back to
            maximum F1 and record ``fallback_used=True``.
        min_cost: minimise fn_cost * FN + fp_cost * FP.
    """
    table = threshold_table(y_true, proba, grid)
    fallback = False
    if rule == "max_recall_min_precision":
        ok = table[table["precision"] >= min_precision]
        if ok.empty:
            fallback = True
            chosen = table.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
        else:
            chosen = ok.sort_values(["recall", "precision", "threshold"],
                                    ascending=[False, False, False]).iloc[0]
    elif rule == "min_cost":
        table["cost"] = fn_cost * table["fn"] + fp_cost * table["fp"]
        chosen = table.sort_values(["cost", "threshold"], ascending=[True, False]).iloc[0]
    else:
        raise ValueError(f"Unknown threshold rule '{rule}'.")
    return {"value": float(chosen["threshold"]), "rule": rule, "min_precision": min_precision,
            "fn_cost": fn_cost, "fp_cost": fp_cost, "fallback_used": fallback,
            "table": table}


def plot_threshold_curves(table: pd.DataFrame, chosen: float, path: Path | str) -> Path:
    """Precision, recall and F1 across thresholds with the chosen value marked."""
    fig, ax = new_figure()
    for col in ("precision", "recall", "f1"):
        ax.plot(table["threshold"], table[col], marker=".", label=col)
    ax.axvline(chosen, color="black", linestyle="--", label=f"chosen = {chosen:.2f}")
    ax.set_title("Validation set: metrics vs decision threshold")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.legend(fontsize=8)
    return save_figure(fig, path)


def run_calibration(
    cfg: dict[str, Any] | None = None,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    reports_dir: Path | str = REPORTS_DIR,
    **load_kwargs: Any,
) -> dict[str, Any]:
    """Compare uncalibrated vs sigmoid, pick the threshold, and write the frozen config."""
    cfg = cfg or load_config()
    artifacts_dir, figures_dir, reports_dir = Path(artifacts_dir), Path(figures_dir), Path(reports_dir)
    seed = cfg["project"]["random_seed"]
    sel_path = artifacts_dir / "selection.json"
    if not sel_path.exists():
        raise FileNotFoundError("selection.json not found. Run `python -m src.train compare` first.")
    selection = json.loads(sel_path.read_text(encoding="utf-8"))
    spec, X_tr, y_tr, X_va, y_va = load_train_val(cfg, **load_kwargs)

    uncal = joblib.load(artifacts_dir / "selected_model.joblib")  # already fitted on TRAIN only
    cal_cfg = cfg["calibration"]
    cal = build_calibrated(uncal, cal_cfg["method"], cal_cfg["cv"], seed).fit(X_tr, y_tr)

    p_uncal = uncal.predict_proba(X_va)[:, 1]
    p_cal = cal.predict_proba(X_va)[:, 1]
    m_uncal = compute_metrics(y_va, p_uncal, 0.5)
    m_cal = compute_metrics(y_va, p_cal, 0.5)

    use_cal = m_cal["log_loss"] < m_uncal["log_loss"]  # predeclared rule (validation log loss)
    deployed, p_dep = (cal, p_cal) if use_cal else (uncal, p_uncal)
    logger.info("Calibration applied: %s (val log loss %.4f -> %.4f)",
                use_cal, m_uncal["log_loss"], m_cal["log_loss"])

    plot_reliability({"uncalibrated": (y_va, p_uncal), f"{cal_cfg['method']} calibrated": (y_va, p_cal)},
                     figures_dir / "reliability_diagram.png",
                     "Validation set: reliability diagram")

    th = cfg["threshold"]
    grid = np.round(np.arange(th["grid_min"], th["grid_max"] + 1e-9, th["grid_step"]), 4)
    chosen = select_threshold(y_va, p_dep, th["rule"], grid, th["min_precision"],
                              th["fn_cost"], th["fp_cost"])
    table: pd.DataFrame = chosen.pop("table")
    table.to_csv(artifacts_dir / "threshold_analysis.csv", index=False)
    plot_threshold_curves(table, chosen["value"], figures_dir / "threshold_curves.png")
    val_metrics = compute_metrics(y_va, p_dep, chosen["value"])

    assumptions = CampaignAssumptions.from_config(cfg)
    core = {"model": selection["model_name"], "params": selection["best_params"],
            "calibrated": bool(use_cal), "threshold": chosen["value"], "seed": seed}
    version = f"{selection['model_name']}-{'cal' if use_cal else 'raw'}-{config_hash(core)[:8]}"
    frozen = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": version,
        "model_name": selection["model_name"],
        "best_params": selection["best_params"],
        "calibration": {"method": cal_cfg["method"], "applied": bool(use_cal), "cv": cal_cfg["cv"],
                        "rule": "apply only if validation log loss is lower",
                        "validation_log_loss_uncalibrated": m_uncal["log_loss"],
                        "validation_log_loss_calibrated": m_cal["log_loss"],
                        "validation_brier_uncalibrated": m_uncal["brier"],
                        "validation_brier_calibrated": m_cal["brier"]},
        "threshold": chosen,
        "campaign_policy": "expected_net_value",
        "simulation_assumptions": assumptions.to_dict(),
        "simulation_assumptions_sha256": config_hash(assumptions.to_dict()),
        "random_seed": seed,
        "validation_metrics_at_threshold": val_metrics,
        "package_versions": package_versions(),
        "note": "Frozen on VALIDATION data. The test set has not been used.",
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(deployed, artifacts_dir / "final_model.joblib")
    (artifacts_dir / "frozen_config.json").write_text(json.dumps(frozen, indent=2, default=str),
                                                      encoding="utf-8")
    metadata = {
        "model_version": version, "model_name": selection["model_name"], "calibrated": bool(use_cal),
        "threshold": chosen["value"], "feature_order": spec.to_dict(),
        "reference_values": build_reference_values(X_tr, spec),   # typical values, no customer rows
        "random_seed": seed, "package_versions": package_versions(),
        "created_at_utc": frozen["frozen_at_utc"],
    }
    (artifacts_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str),
                                                       encoding="utf-8")
    write_calibration_report(frozen, m_uncal, m_cal, table, reports_dir / "calibration.md")
    return frozen


def write_calibration_report(
    frozen: Mapping[str, Any], m_uncal: Mapping[str, float], m_cal: Mapping[str, float],
    table: pd.DataFrame, path: Path,
) -> None:
    """Write reports/calibration.md from measured validation numbers."""
    t = frozen["threshold"]
    text = f"""# Calibration and Threshold Analysis

Generated by `python -m src.calibrate`. All numbers are from the VALIDATION split; the test set was not used.

## Calibration (model: `{frozen['model_name']}`)

| Version | Brier | Log loss |
|---|---|---|
| Uncalibrated | {m_uncal['brier']:.4f} | {m_uncal['log_loss']:.4f} |
| Sigmoid (inside CV, preprocessing refit per fold) | {m_cal['brier']:.4f} | {m_cal['log_loss']:.4f} |

Rule (declared before looking): deploy the calibrated model only if validation log loss is lower.
**Calibration applied: {frozen['calibration']['applied']}.** See `reports/figures/reliability_diagram.png`.

## Threshold (rule: `{t['rule']}`, min precision {t['min_precision']})

**Frozen threshold: {t['value']}** (fallback used: {t['fallback_used']}).

{table.round(3).to_string(index=False)}

Figure: `reports/figures/threshold_curves.png`.

## Vocabulary

* **Churn probability** - the model's calibrated estimate of P(churn) for a customer.
* **Churn-risk score** - any ranking number (e.g. an uncalibrated output). It orders customers but need not be a true probability.
* **Binary churn decision** - probability >= threshold. Depends on the cost trade-off, not on "truth".
* **Retention-campaign selection** - who is contacted. Depends on budget, customer value and assumed offer effectiveness (Stage 7), not only on the threshold.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    setup_logging()
    frozen = run_calibration()
    print(json.dumps({k: frozen[k] for k in ("model_version", "calibration", "threshold")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
