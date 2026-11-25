"""Campaign simulation (Stage 7) and rule-based retention recommendations (Stage 8).

IMPORTANT BOUNDARY
------------------
Everything in this module is a SIMULATION or a RULE based on explicit
assumptions. A churn probability does not prove a customer will respond to an
offer. Nothing here measures real revenue saved or real churn reduction; causal
impact would need randomised intervention data.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.plotting import new_figure, save_figure

logger = logging.getLogger(__name__)

POLICIES = ("random", "highest_risk", "expected_net_value")
SIMULATION_LABEL = "SIMULATED - based on stated assumptions, not measured outcomes"
REQUIRED_SIM_COLUMNS = ("customer_id", "churn_probability", "monthly_charges")


# --------------------------------------------------------------------------- #
# Assumptions and formulas
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CampaignAssumptions:
    """User-visible assumptions. Defaults in configs/default.yaml are placeholders."""

    contribution_margin: float        # share of monthly charges that is profit (0-1)
    retention_horizon_months: float   # months of margin credited to a saved customer
    save_rate: float                  # assumed P(offer saves an otherwise-churning customer)
    contact_cost: float               # cost per contacted customer
    total_budget: float               # total campaign budget

    def __post_init__(self) -> None:
        if not 0 <= self.contribution_margin <= 1:
            raise ValueError("contribution_margin must be between 0 and 1.")
        if self.retention_horizon_months <= 0:
            raise ValueError("retention_horizon_months must be > 0.")
        if not 0 <= self.save_rate <= 1:
            raise ValueError("save_rate must be between 0 and 1.")
        if self.contact_cost < 0:
            raise ValueError("contact_cost must be >= 0.")
        if self.total_budget < 0:
            raise ValueError("total_budget must be >= 0.")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "CampaignAssumptions":
        sim = cfg["simulation"]
        return cls(
            contribution_margin=float(sim["contribution_margin"]),
            retention_horizon_months=float(sim["retention_horizon_months"]),
            save_rate=float(sim["save_rate"]),
            contact_cost=float(sim["contact_cost"]),
            total_budget=float(sim["total_budget"]),
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def customer_value_proxy(monthly_charges: Any, margin: float, horizon_months: float) -> Any:
    """monthly_charges * contribution_margin * retention_horizon_months."""
    return np.asarray(monthly_charges, dtype=float) * margin * horizon_months


def expected_net_value(
    churn_probability: Any, monthly_charges: Any, a: CampaignAssumptions
) -> Any:
    """churn_probability * save_rate * customer_value_proxy - contact_cost."""
    value = customer_value_proxy(monthly_charges, a.contribution_margin, a.retention_horizon_months)
    return np.asarray(churn_probability, dtype=float) * a.save_rate * value - a.contact_cost


def max_contacts(a: CampaignAssumptions, n_available: int) -> int:
    """Largest number of contacts whose total cost does not exceed the budget."""
    if a.contact_cost == 0:
        return n_available  # budget does not bind when contacting is free
    n = int(math.floor(a.total_budget / a.contact_cost + 1e-9))
    while n > 0 and n * a.contact_cost > a.total_budget + 1e-9:
        n -= 1
    return max(0, min(n, n_available))


# --------------------------------------------------------------------------- #
# Selection policies
# --------------------------------------------------------------------------- #
def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_SIM_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Simulation data is missing columns: {missing}")
    p = df["churn_probability"]
    if p.isna().any() or (p < 0).any() or (p > 1).any():
        raise ValueError("churn_probability must be in [0, 1] with no missing values.")
    if df["monthly_charges"].isna().any() or (df["monthly_charges"] < 0).any():
        raise ValueError("monthly_charges must be non-negative with no missing values.")


def select_customers(
    df: pd.DataFrame, policy: str, a: CampaignAssumptions, seed: int | None = None
) -> pd.DataFrame:
    """Choose customers to contact under a policy, respecting the budget.

    * random:              uniformly random order, first N that fit the budget.
    * highest_risk:        descending churn probability, first N that fit.
    * expected_net_value:  descending expected net value, first N that fit,
                           only customers with expected net value > 0.

    Ties are broken deterministically by customer_id (ascending).
    Returns the selected rows with row-level value columns added.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy '{policy}'. Choose from {POLICIES}.")
    _check_columns(df)
    data = df.copy()
    data["customer_value"] = customer_value_proxy(
        data["monthly_charges"], a.contribution_margin, a.retention_horizon_months
    )
    data["expected_net_value"] = expected_net_value(
        data["churn_probability"], data["monthly_charges"], a
    )
    data["expected_saves"] = data["churn_probability"] * a.save_rate
    data["customer_id"] = data["customer_id"].astype(str)
    n_budget = max_contacts(a, len(data))

    if policy == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(data))
        chosen = data.iloc[order[:n_budget]]
    elif policy == "highest_risk":
        chosen = data.sort_values(
            ["churn_probability", "customer_id"], ascending=[False, True], kind="mergesort"
        ).head(n_budget)
    else:
        eligible = data[data["expected_net_value"] > 0]
        chosen = eligible.sort_values(
            ["expected_net_value", "customer_id"], ascending=[False, True], kind="mergesort"
        ).head(n_budget)
    return chosen.copy()


def summarize_selection(selected: pd.DataFrame, a: CampaignAssumptions) -> dict[str, float]:
    """Campaign totals computed from row-level values.

    ``simulated_retained`` is probability-based: sum(p) * save_rate.
    ``label_based_retained`` applies the save rate to customers who really
    churned historically (only if a ``churned`` column is present).
    """
    n = int(len(selected))
    cost = n * a.contact_cost
    gross = float((selected["expected_saves"] * selected["customer_value"]).sum()) if n else 0.0
    churners = float(selected["churned"].sum()) if "churned" in selected and n else (
        float("nan") if "churned" not in selected else 0.0
    )
    return {
        "n_contacted": n,
        "historical_churners_targeted": churners,
        "simulated_retained": float(selected["expected_saves"].sum()) if n else 0.0,
        "label_based_retained": churners * a.save_rate,
        "total_cost": cost,
        "simulated_gross_value": gross,
        "simulated_net_value": gross - cost,
        "simulated_roi": (gross - cost) / cost if cost > 0 else float("nan"),
    }


def compare_policies(
    df: pd.DataFrame, a: CampaignAssumptions, n_random_seeds: int = 20, base_seed: int = 42
) -> pd.DataFrame:
    """Compare the three policies. Random targeting is averaged over many seeds."""
    rows: list[dict[str, Any]] = []
    random_runs = [
        summarize_selection(select_customers(df, "random", a, seed=base_seed + i), a)
        for i in range(n_random_seeds)
    ]
    rand = pd.DataFrame(random_runs)
    row = rand.mean(numeric_only=True).to_dict()
    row["simulated_net_value_std"] = float(rand["simulated_net_value"].std(ddof=1)) if len(rand) > 1 else 0.0
    row["policy"] = "random"
    row["note"] = f"mean of {n_random_seeds} seeds"
    rows.append(row)
    for policy in ("highest_risk", "expected_net_value"):
        r = summarize_selection(select_customers(df, policy, a), a)
        r["simulated_net_value_std"] = 0.0
        r["policy"] = policy
        r["note"] = "deterministic"
        rows.append(r)
    out = pd.DataFrame(rows)
    out["label"] = SIMULATION_LABEL
    cols = ["policy", "n_contacted", "historical_churners_targeted", "simulated_retained",
            "label_based_retained", "total_cost", "simulated_gross_value",
            "simulated_net_value", "simulated_net_value_std", "simulated_roi", "note", "label"]
    return out[cols]


def sensitivity_analysis(
    df: pd.DataFrame,
    base: CampaignAssumptions,
    grid: Mapping[str, list[float]],
    n_random_seeds: int = 10,
    base_seed: int = 42,
) -> pd.DataFrame:
    """One-at-a-time sweep of each assumption; returns long-form results."""
    records: list[pd.DataFrame] = []
    for param, values in grid.items():
        if not hasattr(base, param):
            raise ValueError(f"Unknown assumption '{param}'.")
        for v in values:
            varied = CampaignAssumptions(**{**base.to_dict(), param: v})
            table = compare_policies(df, varied, n_random_seeds, base_seed)
            table.insert(0, "value", v)
            table.insert(0, "parameter", param)
            records.append(table)
    result = pd.concat(records, ignore_index=True)
    return result


def best_policy_by_setting(sens: pd.DataFrame) -> pd.DataFrame:
    """For each (parameter, value) report the policy with the highest simulated net value."""
    idx = sens.groupby(["parameter", "value"])["simulated_net_value"].idxmax()
    best = sens.loc[idx, ["parameter", "value", "policy", "simulated_net_value"]]
    return best.reset_index(drop=True)


def plot_policy_comparison(table: pd.DataFrame, path: Path | str) -> Path:
    """Bar chart of simulated net value by policy."""
    fig, ax = new_figure((6.5, 4.5))
    ax.bar(table["policy"], table["simulated_net_value"],
           yerr=table["simulated_net_value_std"], capsize=4, color="#4C78A8")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Simulated campaign net value by policy\n(SIMULATED - assumption-based)")
    ax.set_xlabel("Targeting policy")
    ax.set_ylabel("Simulated net value (currency units)")
    return save_figure(fig, path)


def plot_sensitivity(sens: pd.DataFrame, path: Path | str) -> Path:
    """Small multiples: simulated net value vs each varied assumption."""
    params = list(sens["parameter"].unique())
    fig = new_figure((5 * len(params) / 2 + 3, 8))[0]
    fig.clf()
    axes = fig.subplots(2, math.ceil(len(params) / 2), squeeze=False).ravel()
    for ax, param in zip(axes, params):
        sub = sens[sens["parameter"] == param]
        for policy, g in sub.groupby("policy"):
            ax.plot(g["value"], g["simulated_net_value"], marker="o", label=policy)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"Vary {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("Simulated net value")
    axes[0].legend(fontsize=8)
    for ax in axes[len(params):]:
        ax.axis("off")
    fig.suptitle("Sensitivity of simulated net value (SIMULATED)")
    return save_figure(fig, path)


# --------------------------------------------------------------------------- #
# Rule-based retention recommendations (Stage 8)
# --------------------------------------------------------------------------- #
RECOMMENDATION_DISCLAIMER = (
    "Suggestions are rule-based prompts for a human reviewer. They are NOT "
    "predictions that an action will prevent churn."
)
RULES_VERSION = "rules-v1"

# Only these (non-sensitive) fields may influence a recommendation.
RULE_INPUT_FIELDS = ("MonthlyCharges", "tenure", "Contract", "TechSupport", "InternetService")


@dataclass
class Recommendation:
    """Recommendation with a per-rule audit trail."""

    risk_band: str
    churn_probability: float
    customer_value: float
    actions: list[dict[str, str]]
    audit: dict[str, Any]
    requires_human_review: bool = True
    disclaimer: str = RECOMMENDATION_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def primary_action(self) -> str:
        return self.actions[0]["action"] if self.actions else "No action"


def risk_band(probability: float, low_max: float, high_min: float) -> str:
    """Map a probability to Low / Medium / High using explicit cut-offs."""
    if probability >= high_min:
        return "High"
    if probability >= low_max:
        return "Medium"
    return "Low"


def recommend_actions(
    record: Mapping[str, Any],
    churn_probability: float,
    assumptions: CampaignAssumptions,
    rule_cfg: Mapping[str, float],
    high_risk_threshold: float,
) -> Recommendation:
    """Transparent, rule-based suggestions with an audit trail.

    Only RULE_INPUT_FIELDS, the churn probability and the customer-value proxy
    are read. Sensitive attributes (gender, SeniorCitizen, Partner, Dependents)
    are deliberately never accessed.
    """
    value = float(customer_value_proxy(
        record["MonthlyCharges"], assumptions.contribution_margin,
        assumptions.retention_horizon_months))
    band = risk_band(churn_probability, rule_cfg["low_risk_max"], high_risk_threshold)
    high_value = value >= rule_cfg["high_value_cutoff"]
    actions: list[dict[str, str]] = []

    def add(rule_id: str, action: str, why: str) -> None:
        actions.append({"rule_id": rule_id, "action": action, "rationale": why})

    if band == "High" and high_value:
        add("R1", "Route to a human retention specialist for a personal review.",
            f"High risk band (p={churn_probability:.2f}) and customer value proxy "
            f"{value:.0f} >= {rule_cfg['high_value_cutoff']:.0f}.")
    elif band == "High":
        add("R2", "Send a low-cost automated message summarising plan options.",
            f"High risk band (p={churn_probability:.2f}) and customer value proxy "
            f"{value:.0f} < {rule_cfg['high_value_cutoff']:.0f}.")

    if band != "Low":
        if float(record["MonthlyCharges"]) >= rule_cfg["high_monthly_charges"]:
            add("R3", "Offer a plan review to check whether a lower-cost suitable plan fits.",
                f"MonthlyCharges {float(record['MonthlyCharges']):.2f} >= "
                f"{rule_cfg['high_monthly_charges']:.2f}.")
        if record["Contract"] == "Month-to-month":
            add("R4", "Review contract options with the customer (no pressure to switch).",
                "Contract is Month-to-month.")
        tenure = record["tenure"]
        if tenure is not None and not pd.isna(tenure) and float(tenure) <= rule_cfg["low_tenure_months"]:
            add("R5", "Offer onboarding assistance / a check-in call.",
                f"tenure {float(tenure):.0f} <= {rule_cfg['low_tenure_months']:.0f} months.")
        if record["TechSupport"] == "No" and record["InternetService"] != "No":
            add("R6", "Assess whether the customer needs technical support.",
                "Has internet service but no TechSupport add-on.")

    if not actions:
        add("R0", "No proactive action suggested; keep monitoring.",
            f"Risk band is {band} and no service/billing rule applied.")

    audit = {
        "rules_version": RULES_VERSION,
        "inputs_used": {f: _jsonable(record.get(f)) for f in RULE_INPUT_FIELDS},
        "churn_probability": round(float(churn_probability), 4),
        "customer_value_proxy": round(value, 2),
        "risk_band": band,
        "high_risk_threshold": high_risk_threshold,
        "rules_fired": [x["rule_id"] for x in actions],
        "sensitive_attributes_used": False,
    }
    return Recommendation(band, float(churn_probability), value, actions, audit)


def _jsonable(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def append_audit_log(path: Path | str, recommendation: Recommendation, customer_ref: str = "") -> None:
    """Append one JSON line per recommendation. Pass a pseudonymous reference, not PII."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    entry = {"customer_ref": customer_ref, **recommendation.audit}
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------- #
# CLI: run the simulation on the VALIDATION split (never the test split)
# --------------------------------------------------------------------------- #
def run_campaign_simulation(
    cfg: dict[str, Any] | None = None,
    artifacts_dir: Path | str | None = None,
    figures_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    **load_kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Score the validation split with the final model and simulate the three policies."""
    import joblib

    from src.config import ARTIFACTS_DIR, FIGURES_DIR, REPORTS_DIR, load_config
    from src.train import load_train_val

    cfg = cfg or load_config()
    artifacts_dir = Path(artifacts_dir or ARTIFACTS_DIR)
    figures_dir = Path(figures_dir or FIGURES_DIR)
    reports_dir = Path(reports_dir or REPORTS_DIR)
    a = CampaignAssumptions.from_config(cfg)
    _, _, _, X_va, y_va = load_train_val(cfg, **load_kwargs)
    from src import data as data_mod
    ids = data_mod.load_split("val", **{k: v for k, v in load_kwargs.items()})["customerID"]
    model = joblib.load(artifacts_dir / "final_model.joblib")
    proba = model.predict_proba(X_va)[:, 1]
    df = pd.DataFrame({
        "customer_id": ids.astype(str).to_numpy(),
        "churn_probability": proba,
        "monthly_charges": X_va["MonthlyCharges"].fillna(X_va["MonthlyCharges"].median()).to_numpy(),
        "churned": y_va.to_numpy(),
    })
    seeds = cfg["simulation"]["n_random_seeds"]
    seed = cfg["project"]["random_seed"]
    comparison = compare_policies(df, a, seeds, seed)
    sens = sensitivity_analysis(df, a, cfg["simulation"]["sensitivity"], max(5, seeds // 2), seed)
    best = best_policy_by_setting(sens)
    comparison.to_csv(artifacts_dir / "campaign_comparison.csv", index=False)
    sens.to_csv(artifacts_dir / "campaign_sensitivity.csv", index=False)
    plot_policy_comparison(comparison, figures_dir / "campaign_policy_comparison.png")
    plot_sensitivity(sens, figures_dir / "campaign_sensitivity.png")
    write_campaign_report(comparison, best, a, reports_dir / "campaign_simulation.md")
    return {"comparison": comparison, "sensitivity": sens, "best": best}


def write_campaign_report(
    comparison: pd.DataFrame, best: pd.DataFrame, a: CampaignAssumptions, path: Path
) -> None:
    """Write reports/campaign_simulation.md (all numbers are simulated)."""
    show = comparison.drop(columns=["label"]).round(3)
    text = f"""# Campaign Simulation (SIMULATED)

> **All outcomes below are simulations based on the stated assumptions. They are not measured
> revenue, not measured churn reduction, and not causal estimates.** Generated on the VALIDATION
> split by `python -m src.policy`.

## Assumptions (illustrative placeholders - replace with your own)

{json.dumps(a.to_dict(), indent=2)}

* Customer value proxy = monthly_charges x contribution_margin x retention_horizon_months
* Expected net value = churn_probability x save_rate x value proxy - contact_cost
* Customers with expected net value <= 0 are never selected by the expected-net-value policy.
* Random targeting is averaged over several seeds.
* "Simulated retained" = sum of churn_probability x save_rate over contacted customers.
* "Historical churners targeted" counts people who actually churned in the data; it says nothing
  about whether an offer would have kept them.

## Policy comparison

{show.to_string(index=False)}

## Which policy has the highest simulated net value as assumptions change?

{best.round(3).to_string(index=False)}

The expected-net-value policy is a transparent heuristic (rank by an expected-value formula under a
budget). It is **not claimed to be optimal**; no optimisation method was implemented or validated.
Figures: `reports/figures/campaign_policy_comparison.png`, `campaign_sensitivity.png`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    from src.config import setup_logging
    setup_logging()
    out = run_campaign_simulation()
    print(out["comparison"].drop(columns=["label"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
