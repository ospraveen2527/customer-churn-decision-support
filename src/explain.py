"""Explainability (Stage 6): permutation importance and per-customer occlusion.

Both methods call the SAME deployed ``predict_proba`` function (calibrated
pipeline), so explanations match what the dashboard scores.

* Global: permutation importance - how much validation average precision drops
  when a feature's values are shuffled.
* Local: occlusion - how much one customer's predicted probability changes when
  one feature is replaced by a typical training value (median / mode).

These show MODEL ASSOCIATIONS, not causes of churn. They never justify claiming
that changing a feature will prevent churn. SHAP is an optional alternative.

Run: ``python -m src.explain``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import (
    ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR, SENSITIVE_ATTRIBUTES, load_config, setup_logging,
)
from src.plotting import new_figure, save_figure
from src.train import load_train_val

logger = logging.getLogger(__name__)

CAUSAL_CAVEAT = (
    "These factors describe what the model associates with higher or lower predicted churn "
    "probability. They are not proven causes, and changing a factor will not necessarily "
    "change whether the customer leaves."
)


def global_importance(
    model: Any, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, seed: int = 42
) -> pd.DataFrame:
    """Permutation importance (drop in average precision) on the given data."""
    result = permutation_importance(model, X, y, scoring="average_precision",
                                    n_repeats=n_repeats, random_state=seed, n_jobs=1)
    out = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    out["sensitive_attribute"] = out["feature"].isin(SENSITIVE_ATTRIBUTES)
    return out


def local_explanation(
    model: Any, row: pd.DataFrame, reference: Mapping[str, Any], top_k: int = 5
) -> pd.DataFrame:
    """Occlusion explanation for ONE customer (a one-row DataFrame).

    ``delta`` = P(churn | actual value) - P(churn | feature set to its typical value).
    Positive delta: this feature value is associated with HIGHER predicted risk.
    """
    if len(row) != 1:
        raise ValueError("local_explanation expects exactly one row.")
    base_p = float(model.predict_proba(row)[:, 1][0])
    variants = []
    for col in row.columns:
        v = row.copy()
        v[col] = reference[col]
        variants.append(v)
    p_ref = model.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
    out = pd.DataFrame({
        "feature": list(row.columns),
        "value": [row[c].iloc[0] for c in row.columns],
        "typical_value": [reference[c] for c in row.columns],
        "delta_probability": base_p - p_ref,
    })
    out["sensitive_attribute"] = out["feature"].isin(SENSITIVE_ATTRIBUTES)
    out["abs_delta"] = out["delta_probability"].abs()
    out = out.sort_values("abs_delta", ascending=False).head(top_k).drop(columns="abs_delta")
    out.attrs["predicted_probability"] = base_p
    return out.reset_index(drop=True)


def plain_language(explanation: pd.DataFrame) -> list[str]:
    """Non-technical sentences describing a local explanation."""
    lines: list[str] = []
    for _, r in explanation.iterrows():
        points = r["delta_probability"] * 100
        direction = "higher" if points > 0 else "lower"
        lines.append(
            f"{r['feature']} = {r['value']} is associated with a {direction} predicted churn "
            f"probability ({points:+.1f} percentage points versus a typical value of "
            f"{r['typical_value']})."
        )
    return lines


def plot_global_importance(imp: pd.DataFrame, path: Path | str, top_n: int = 15) -> Path:
    """Horizontal bar chart of permutation importance."""
    top = imp.head(top_n).iloc[::-1]
    fig, ax = new_figure((7, 5.5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#4C78A8")
    ax.set_title("Global feature importance (validation set)\nassociation with model output, not causation")
    ax.set_xlabel("Drop in average precision when feature is shuffled")
    ax.set_ylabel("Feature")
    return save_figure(fig, path)


def plot_local_explanation(expl: pd.DataFrame, path: Path | str, title: str) -> Path:
    """Bar chart of per-feature probability change for one customer."""
    data = expl.iloc[::-1]
    fig, ax = new_figure((7, 4))
    colors = ["#E45756" if d > 0 else "#4C78A8" for d in data["delta_probability"]]
    ax.barh([f"{f} = {v}" for f, v in zip(data["feature"], data["value"])],
            data["delta_probability"] * 100, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Change in predicted churn probability (percentage points)")
    ax.set_ylabel("Feature value")
    return save_figure(fig, path)


def run_explainability(
    cfg: dict[str, Any] | None = None,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    reports_dir: Path | str = REPORTS_DIR,
    **load_kwargs: Any,
) -> dict[str, Any]:
    """Global importance + three example local explanations on VALIDATION data."""
    cfg = cfg or load_config()
    artifacts_dir, figures_dir, reports_dir = Path(artifacts_dir), Path(figures_dir), Path(reports_dir)
    spec, X_tr, y_tr, X_va, y_va = load_train_val(cfg, **load_kwargs)
    model = joblib.load(artifacts_dir / "final_model.joblib")
    meta = json.loads((artifacts_dir / "model_metadata.json").read_text(encoding="utf-8"))
    ref = meta["reference_values"]
    ex = cfg["explain"]

    imp = global_importance(model, X_va, y_va, ex["n_repeats"], cfg["project"]["random_seed"])
    imp.to_csv(artifacts_dir / "global_importance.csv", index=False)
    plot_global_importance(imp, figures_dir / "global_importance.png")

    proba = model.predict_proba(X_va)[:, 1]
    order = np.argsort(-proba, kind="mergesort")
    picks = {"highest_risk": order[0], "median_risk": order[len(order) // 2], "lowest_risk": order[-1]}
    local: dict[str, Any] = {}
    for label, idx in list(picks.items())[: ex["n_local_examples"]]:
        expl = local_explanation(model, X_va.iloc[[int(idx)]], ref, ex["top_k"])
        plot_local_explanation(expl, figures_dir / f"local_{label}.png",
                               f"Local explanation: {label.replace('_', ' ')} validation customer\n"
                               f"(predicted probability {expl.attrs['predicted_probability']:.2f})")
        local[label] = {"probability": expl.attrs["predicted_probability"],
                        "sentences": plain_language(expl),
                        "table": expl.to_dict(orient="records")}
    (artifacts_dir / "local_explanations_examples.json").write_text(
        json.dumps(local, indent=2, default=str), encoding="utf-8")
    write_explainability_report(imp, local, reports_dir / "explainability.md")
    return {"importance": imp, "local": local}


def write_explainability_report(imp: pd.DataFrame, local: Mapping[str, Any], path: Path) -> None:
    """Write reports/explainability.md from measured results."""
    parts = [
        "# Explainability", "",
        "Generated by `python -m src.explain` on the VALIDATION split.", "",
        f"> {CAUSAL_CAVEAT}", "",
        "## Global importance (permutation, drop in average precision)", "",
        imp.head(15).round(4).to_string(index=False), "",
        "Figure: `reports/figures/global_importance.png`. Rows with `sensitive_attribute=True` are "
        "attributes describing the person; they are never used to decide retention treatment.", "",
        "## Local examples (occlusion vs typical training value)", "",
    ]
    for label, item in local.items():
        parts += [f"### {label.replace('_', ' ').title()} (predicted p = {item['probability']:.3f})", ""]
        parts += [f"* {s}" for s in item["sentences"]] + [""]
    parts += [
        "## Method notes", "",
        "* Both methods call the deployed `predict_proba`, so they explain the model actually used.",
        "* Occlusion changes ONE feature at a time and can miss interactions; permutation importance "
        "can be distorted by correlated features. SHAP (optional) has its own caveats.",
        "* No demographic attribute is ever modified to build a 'lower-risk scenario'.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    setup_logging()
    result = run_explainability()
    print(result["importance"].head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
