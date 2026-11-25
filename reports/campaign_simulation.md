# Campaign Simulation (SIMULATED)

> **All outcomes below are simulations based on the stated assumptions. They are not measured
> revenue, not measured churn reduction, and not causal estimates.** Generated on the VALIDATION
> split by `python -m src.policy`.

## Assumptions (illustrative placeholders - replace with your own)

{
  "contribution_margin": 0.3,
  "retention_horizon_months": 12.0,
  "save_rate": 0.2,
  "contact_cost": 5.0,
  "total_budget": 2500.0
}

* Customer value proxy = monthly_charges x contribution_margin x retention_horizon_months
* Expected net value = churn_probability x save_rate x value proxy - contact_cost
* Customers with expected net value <= 0 are never selected by the expected-net-value policy.
* Random targeting is averaged over several seeds.
* "Simulated retained" = sum of churn_probability x save_rate over contacted customers.
* "Historical churners targeted" counts people who actually churned in the data; it says nothing
  about whether an offer would have kept them.

## Policy comparison

            policy  n_contacted  historical_churners_targeted  simulated_retained  label_based_retained  total_cost  simulated_gross_value  simulated_net_value  simulated_net_value_std  simulated_roi             note
            random        500.0                         134.7              27.033                 26.94      2500.0               7175.512             4675.512                  317.716          1.870 mean of 20 seeds
      highest_risk        500.0                         283.0              56.180                 56.60      2500.0              15699.822            13199.822                    0.000          5.280    deterministic
expected_net_value        500.0                         268.0              54.056                 53.60      2500.0              16181.170            13681.170                    0.000          5.472    deterministic

## Which policy has the highest simulated net value as assumptions change?

               parameter  value             policy  simulated_net_value
            contact_cost   1.00 expected_net_value            19033.378
            contact_cost   2.50 expected_net_value            17456.701
            contact_cost   5.00 expected_net_value            13681.170
            contact_cost  10.00 expected_net_value             7696.462
            contact_cost  20.00 expected_net_value             3212.468
     contribution_margin   0.10 expected_net_value             2893.723
     contribution_margin   0.20 expected_net_value             8287.447
     contribution_margin   0.30 expected_net_value            13681.170
     contribution_margin   0.40 expected_net_value            19074.894
     contribution_margin   0.60 expected_net_value            29862.340
retention_horizon_months   3.00 expected_net_value             1590.364
retention_horizon_months   6.00 expected_net_value             5590.585
retention_horizon_months  12.00 expected_net_value            13681.170
retention_horizon_months  24.00 expected_net_value            29862.340
               save_rate   0.05 expected_net_value             1590.364
               save_rate   0.10 expected_net_value             5590.585
               save_rate   0.20 expected_net_value            13681.170
               save_rate   0.30 expected_net_value            21771.755
               save_rate   0.50 expected_net_value            37952.926

The expected-net-value policy is a transparent heuristic (rank by an expected-value formula under a
budget). It is **not claimed to be optimal**; no optimisation method was implemented or validated.
Figures: `reports/figures/campaign_policy_comparison.png`, `campaign_sensitivity.png`.
