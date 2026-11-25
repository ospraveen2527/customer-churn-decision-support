"""Streamlit dashboard: ``streamlit run app/streamlit_app.py``.

Seven pages: overview, single prediction, batch CSV, explanation, campaign
simulator, model evaluation, limitations. No customer data is stored; uploads
live in memory for the session only. No communications are ever sent.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.app_utils import (  # noqa: E402
    InputValidationError, ModelBundle, load_bundle, score_dataframe, score_single,
    simulation_frame, to_export_csv, validate_dataframe,
)
from src.config import ARTIFACTS_DIR, CATEGORY_VALUES, FIGURES_DIR  # noqa: E402
from src.policy import (  # noqa: E402
    RECOMMENDATION_DISCLAIMER, SIMULATION_LABEL, CampaignAssumptions, compare_policies,
)

st.set_page_config(page_title="Churn Decision Support", page_icon=":bar_chart:", layout="wide")

SIM_BANNER = ("SIMULATED: campaign outcomes are calculated from the assumptions you set. They are "
              "not measured revenue saved or churn reduction, and not causal estimates.")


@st.cache_resource(show_spinner="Loading model...")
def get_bundle() -> ModelBundle:
    return load_bundle(ARTIFACTS_DIR)


def footer(bundle: ModelBundle) -> None:
    st.caption(f"Model version: `{bundle.version}` - decision threshold {bundle.threshold:.2f}. "
               "No communications are sent and no uploaded data is stored.")


def page_overview(bundle: ModelBundle) -> None:
    st.title("Customer Churn Decision-Support System")
    st.write("Estimates churn risk, shows model-associated factors, suggests rule-based actions "
             "for human review, and simulates campaign scenarios under explicit assumptions.")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**1. Measured performance**  \nHeld-out metrics (Model evaluation page).")
    c2.markdown("**2. Explanation**  \nModel associations, not causes.")
    c3.markdown("**3. Simulation**  \nAssumption-based scenarios.")
    c4.markdown("**4. Causal impact**  \nNot claimed; needs a randomised test.")
    st.info("This is a portfolio/demo project trained on a public telecom dataset. It is not "
            "validated for production use.")
    footer(bundle)


def single_form(bundle: ModelBundle) -> dict:
    cols = st.columns(3)
    record: dict = {}
    record["tenure"] = cols[0].number_input("Tenure (months)", 0, 120, 12)
    record["MonthlyCharges"] = cols[1].number_input("Monthly charges", 0.0, 500.0, 70.0, step=0.5)
    total = cols[2].text_input("Total charges (blank if unknown)", "")
    record["TotalCharges"] = total
    for i, (name, options) in enumerate(CATEGORY_VALUES.items()):
        record[name] = cols[i % 3].selectbox(name, options)
    return record


def page_single(bundle: ModelBundle) -> None:
    st.header("Single-customer prediction")
    with st.form("single"):
        record = single_form(bundle)
        submitted = st.form_submit_button("Score customer")
    if not submitted:
        return
    try:
        result = score_single(bundle, record)
    except InputValidationError as exc:
        for err in exc.errors:
            st.error(err)
        return
    st.session_state["last_record"] = record
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated churn probability", f"{result['churn_probability']:.1%}")
    c2.metric("Risk band", result["risk_band"])
    c3.metric("At/above threshold?", "Yes" if result["flagged_at_threshold"] else "No")
    st.subheader("Top model-associated factors")
    for line in result["explanation_text"]:
        st.write("- " + line)
    st.caption("Associations in the model, not proven causes of churn.")
    st.subheader("Suggested actions (rule-based, for human review)")
    for a in result["recommendation"]["actions"]:
        st.write(f"**{a['rule_id']}** - {a['action']}  \n_{a['rationale']}_")
    st.warning(RECOMMENDATION_DISCLAIMER)
    with st.expander("Audit record"):
        st.json(result["recommendation"]["audit"])
    st.caption(f"Scored at {result['scoring_timestamp_utc']} UTC - model `{result['model_version']}`")


def page_batch(bundle: ModelBundle) -> None:
    st.header("Batch CSV prediction")
    st.write("Upload a CSV with the columns of the original Telco file (customerID optional). "
             "The file is processed in memory and not saved.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is None:
        return
    try:
        raw = pd.read_csv(up, keep_default_na=False, na_values=[""])
        validated = validate_dataframe(raw, bundle.feature_columns)
    except InputValidationError as exc:
        for err in exc.errors:
            st.error(err)
        return
    except Exception as exc:  # noqa: BLE001 - show a clear message instead of a traceback
        st.error(f"Could not read the file as CSV: {exc}")
        return
    scored = score_dataframe(bundle, validated)
    st.session_state["scored"] = scored
    st.success(f"Scored {len(scored)} customers.")
    st.dataframe(scored.drop(columns=["monthly_charges", "churned"], errors="ignore").head(200))
    st.download_button("Download predictions CSV", to_export_csv(scored), "predictions.csv", "text/csv")
    footer(bundle)


def page_explanation(bundle: ModelBundle) -> None:
    st.header("Explanation")
    st.info("Explanations show what the model associates with its output. They do not prove "
            "causes and do not show that changing a factor will prevent churn.")
    imp_path = ARTIFACTS_DIR / "global_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        st.subheader("Global importance (validation data)")
        st.bar_chart(imp.head(15).set_index("feature")["importance_mean"])
    else:
        st.write("Run `python -m src.explain` to create global importance.")
    st.subheader("Local explanation")
    if "last_record" in st.session_state:
        result = score_single(bundle, st.session_state["last_record"])
        st.write(f"Last scored customer: {result['churn_probability']:.1%}")
        for line in result["explanation_text"]:
            st.write("- " + line)
    else:
        st.write("Score a customer on the 'Single-customer prediction' page first.")


def page_campaign(bundle: ModelBundle) -> None:
    st.header("Campaign simulator")
    st.error(SIM_BANNER)
    if "scored" not in st.session_state:
        st.write("Score a CSV on the 'Batch CSV prediction' page first (the simulator needs "
                 "probabilities and monthly charges for many customers).")
        return
    base = bundle.assumptions
    c = st.columns(5)
    a = CampaignAssumptions(
        contribution_margin=c[0].slider("Contribution margin", 0.0, 1.0, float(base.contribution_margin), 0.05),
        retention_horizon_months=c[1].slider("Retention horizon (months)", 1, 60, int(base.retention_horizon_months)),
        save_rate=c[2].slider("Assumed save rate", 0.0, 1.0, float(base.save_rate), 0.01),
        contact_cost=c[3].number_input("Cost per contact", 0.0, 1000.0, float(base.contact_cost)),
        total_budget=c[4].number_input("Total budget", 0.0, 10_000_000.0, float(base.total_budget), step=100.0),
    )
    table = compare_policies(simulation_frame(st.session_state["scored"]), a)
    st.dataframe(table.drop(columns=["label"]).round(3))
    st.bar_chart(table.set_index("policy")["simulated_net_value"])
    st.caption(SIMULATION_LABEL + ". Expected-net-value targeting is a heuristic, not proven optimal.")


def page_evaluation(bundle: ModelBundle) -> None:
    st.header("Model evaluation")
    st.write("Validation-set results (used for model selection). Test results appear only after the "
             "final locked evaluation has been run.")
    for name in ("model_comparison.csv", "threshold_analysis.csv"):
        p = ARTIFACTS_DIR / name
        if p.exists():
            st.subheader(name)
            st.dataframe(pd.read_csv(p).drop(columns=["best_params"], errors="ignore").round(4))
    for fig in ("val_pr_curves.png", "reliability_diagram.png", "threshold_curves.png"):
        if (FIGURES_DIR / fig).exists():
            st.image(str(FIGURES_DIR / fig), caption=fig)
    test_path = ARTIFACTS_DIR / "test_metrics.json"
    if test_path.exists():
        st.subheader("Final test-set metrics (evaluated once)")
        st.json(json.loads(test_path.read_text(encoding="utf-8"))["metrics"])
    else:
        st.caption("Final test evaluation: not run yet.")


def page_limitations(bundle: ModelBundle) -> None:
    st.header("Limitations")
    st.markdown("""
- Public, single-snapshot telecom dataset; no timestamps, so time-based validation was not possible.
- Churners' `tenure` and `TotalCharges` reflect their full time as customers, which may make the task look easier than a real forward-looking prediction.
- Predicting churn is not the same as knowing an offer will work; causal impact needs a randomised experiment.
- Simulation assumptions (margin, save rate, cost) are placeholders you must replace.
- Explanations show associations, not causes.
- Fairness across demographic groups has not been fully audited.
- Not deployed or monitored in production.
""")


PAGES = {
    "Project overview": page_overview, "Single-customer prediction": page_single,
    "Batch CSV prediction": page_batch, "Explanation": page_explanation,
    "Campaign simulator": page_campaign, "Model evaluation": page_evaluation,
    "Limitations": page_limitations,
}


def main() -> None:
    try:
        bundle = get_bundle()
    except FileNotFoundError as exc:
        st.title("Model not found")
        st.error(str(exc))
        st.stop()
    choice = st.sidebar.radio("Page", list(PAGES))
    st.sidebar.caption(f"Model `{bundle.version}`")
    PAGES[choice](bundle)


main()
