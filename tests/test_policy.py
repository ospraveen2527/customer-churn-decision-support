import numpy as np
import pandas as pd
import pytest

from src import policy
from src.policy import CampaignAssumptions, compare_policies, select_customers, summarize_selection

A = CampaignAssumptions(contribution_margin=0.3, retention_horizon_months=12, save_rate=0.2,
                        contact_cost=5.0, total_budget=100.0)


@pytest.fixture()
def pool():
    rng = np.random.default_rng(1)
    n = 200
    return pd.DataFrame({
        "customer_id": [f"C{i:03d}" for i in range(n)],
        "churn_probability": rng.random(n),
        "monthly_charges": rng.uniform(20, 120, n),
        "churned": rng.integers(0, 2, n),
    })


@pytest.mark.parametrize("pol", policy.POLICIES)
def test_campaign_cost_never_exceeds_budget(pool, pol):
    for budget in (0, 4.99, 5, 99.9, 100, 1e6):
        a = CampaignAssumptions(**{**A.to_dict(), "total_budget": budget})
        sel = select_customers(pool, pol, a, seed=0)
        assert len(sel) * a.contact_cost <= budget + 1e-9


@pytest.mark.parametrize("pol", policy.POLICIES)
def test_zero_budget_selects_nobody(pool, pol):
    a = CampaignAssumptions(**{**A.to_dict(), "total_budget": 0.0})
    assert len(select_customers(pool, pol, a, seed=0)) == 0


def test_negative_expected_net_value_never_selected(pool):
    a = CampaignAssumptions(**{**A.to_dict(), "total_budget": 1e9, "contact_cost": 5.0})
    sel = select_customers(pool, "expected_net_value", a)
    assert (sel["expected_net_value"] > 0).all()
    assert len(sel) < len(pool)  # some customers were excluded for negative value


def test_expected_net_value_ordering_is_descending(pool):
    sel = select_customers(pool, "expected_net_value", A)
    assert sel["expected_net_value"].is_monotonic_decreasing


def test_tie_breaking_is_deterministic():
    df = pd.DataFrame({"customer_id": ["B", "A", "C", "D"], "churn_probability": [0.5] * 4,
                       "monthly_charges": [100.0] * 4})
    a = CampaignAssumptions(**{**A.to_dict(), "total_budget": 10.0})
    first = list(select_customers(df, "expected_net_value", a)["customer_id"])
    again = list(select_customers(df.sample(frac=1, random_state=3), "expected_net_value", a)["customer_id"])
    assert first == again == ["A", "B"]


def test_higher_contact_cost_cannot_increase_expected_net_value(pool):
    low = policy.expected_net_value(pool["churn_probability"], pool["monthly_charges"], A)
    hi_a = CampaignAssumptions(**{**A.to_dict(), "contact_cost": 9.0})
    high = policy.expected_net_value(pool["churn_probability"], pool["monthly_charges"], hi_a)
    assert (high <= low).all()


def test_totals_match_row_level_calculations(pool):
    sel = select_customers(pool, "highest_risk", A)
    s = summarize_selection(sel, A)
    manual_gross = sum(p * A.save_rate * (m * A.contribution_margin * A.retention_horizon_months)
                       for p, m in zip(sel["churn_probability"], sel["monthly_charges"]))
    assert s["n_contacted"] == len(sel) == 20
    assert np.isclose(s["simulated_net_value"], manual_gross - 20 * A.contact_cost)
    assert np.isclose(s["simulated_net_value"], sel["expected_net_value"].sum())
    assert s["historical_churners_targeted"] == sel["churned"].sum()
    assert np.isclose(s["simulated_roi"], s["simulated_net_value"] / s["total_cost"])


def test_compare_policies_table_and_random_seed_reproducibility(pool):
    t1 = compare_policies(pool, A, n_random_seeds=5)
    t2 = compare_policies(pool, A, n_random_seeds=5)
    assert list(t1["policy"]) == ["random", "highest_risk", "expected_net_value"]
    assert t1.equals(t2)
    assert t1.loc[t1["policy"] == "expected_net_value", "simulated_net_value"].iloc[0] >= \
        t1.loc[t1["policy"] == "random", "simulated_net_value"].iloc[0]
    assert t1["label"].str.contains("SIMULATED").all()


def test_sensitivity_analysis_shapes(pool):
    s = policy.sensitivity_analysis(pool, A, {"save_rate": [0.1, 0.3], "contact_cost": [2.0, 8.0]}, 3)
    assert set(s["parameter"]) == {"save_rate", "contact_cost"} and len(s) == 12
    assert len(policy.best_policy_by_setting(s)) == 4


def test_invalid_assumptions_rejected():
    with pytest.raises(ValueError):
        CampaignAssumptions(1.5, 12, 0.2, 5, 100)
    with pytest.raises(ValueError):
        CampaignAssumptions(0.3, 12, -0.1, 5, 100)


# ----- rule-based recommendations ------------------------------------------------------------
RULES = {"low_risk_max": 0.2, "high_monthly_charges": 80.0, "low_tenure_months": 12,
         "high_value_cutoff": 250.0}
REC = {"MonthlyCharges": 95.0, "tenure": 3, "Contract": "Month-to-month",
       "TechSupport": "No", "InternetService": "Fiber optic"}


def test_high_value_high_risk_routes_to_human():
    r = policy.recommend_actions(REC, 0.8, A, RULES, 0.5)
    assert r.risk_band == "High" and r.actions[0]["rule_id"] == "R1"
    assert {"R3", "R4", "R5", "R6"} <= {a["rule_id"] for a in r.actions}
    assert r.requires_human_review and "NOT" in r.disclaimer


def test_low_value_high_risk_gets_low_cost_automation():
    r = policy.recommend_actions({**REC, "MonthlyCharges": 30.0}, 0.8, A, RULES, 0.5)
    assert r.actions[0]["rule_id"] == "R2"


def test_low_risk_gets_no_proactive_action():
    r = policy.recommend_actions(REC, 0.05, A, RULES, 0.5)
    assert r.risk_band == "Low" and [a["rule_id"] for a in r.actions] == ["R0"]


def test_recommendations_ignore_sensitive_attributes():
    base = policy.recommend_actions({**REC, "gender": "Male", "SeniorCitizen": 0, "Partner": "No"}, 0.7, A, RULES, 0.5)
    other = policy.recommend_actions({**REC, "gender": "Female", "SeniorCitizen": 1, "Partner": "Yes"}, 0.7, A, RULES, 0.5)
    assert base.actions == other.actions
    assert base.audit["sensitive_attributes_used"] is False
    assert not {"gender", "SeniorCitizen", "Partner", "Dependents"} & set(base.audit["inputs_used"])


def test_audit_log_written_as_jsonl(tmp_path):
    r = policy.recommend_actions(REC, 0.7, A, RULES, 0.5)
    policy.append_audit_log(tmp_path / "audit.jsonl", r, "ref-1")
    policy.append_audit_log(tmp_path / "audit.jsonl", r, "ref-2")
    assert len((tmp_path / "audit.jsonl").read_text().strip().splitlines()) == 2


def test_policy_outputs_generated(env):
    assert (env["artifacts"] / "campaign_comparison.csv").exists()
    assert "SIMULATED" in (env["reports"] / "campaign_simulation.md").read_text()
