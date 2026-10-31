"""Unit & Integration Tests for Predictive Risk, Causal Uplift & Stackelberg Optimizer.

Verifies:
- Baseline Churn Model calibration and TreeSHAP top-3 drivers
- Multi-treatment T-Learner CATE and 90% Lower Confidence Bound (LCB)
- Eligibility matrix guardrails (0 Sleeping Dogs permitted)
- Stackelberg MILP knapsack solution under budget constraint B
- Memory consumption strictly < 600 MB RSS
"""

import numpy as np
import pandas as pd
import pytest

try:
    import psutil
    def get_process_ram_mb() -> float:
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
except ImportError:
    import resource
    def get_process_ram_mb() -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

from src.config import ACTION_NAMES
from src.data_engineering.ingest import generate_calibrated_rct_telemetry
from src.decisioning.eligibility import evaluate_eligibility_matrix
from src.decisioning.stackelberg_optimizer import StackelbergBudgetOptimizer
from src.modeling.causal_uplift import MultiTreatmentTLearner
from src.modeling.evaluation import evaluate_policy_comparison
from src.modeling.predictive_risk import BaselineRiskModel


@pytest.fixture(scope="module")
def sample_gold_data():
    """Generate consistent 3,000-subscriber cohort for rapid testing."""
    df = generate_calibrated_rct_telemetry(n_subscribers=3000, random_seed=42)
    # Add point-in-time engineered features
    df["arpu_tenure_velocity"] = (df["monthly_arpu_idr"] / (df["tenure_months"] + 1.0)).astype(np.float32)
    df["high_drop_rate_flag"] = (df["drop_call_rate"] >= 5.0).astype(np.int8)
    df["high_complaint_flag"] = (df["complaint_count_90d"] >= 2).astype(np.int8)
    return df


def test_baseline_risk_and_treeshap(sample_gold_data):
    """Test calibrated classifier and TreeSHAP explanation drivers."""
    ram_start = get_process_ram_mb()

    model = BaselineRiskModel(random_state=42)
    model.fit(sample_gold_data)

    metrics = model.evaluate(sample_gold_data)
    assert metrics["roc_auc"] >= 0.70, f"ROC-AUC {metrics['roc_auc']} is below 0.70"
    assert metrics["brier_score"] <= 0.20, f"Brier Score {metrics['brier_score']} is above 0.20"

    # Test individual TreeSHAP drivers
    sample_subset = sample_gold_data.iloc[:5]
    drivers = model.explain_drivers(sample_subset, top_k=3)
    assert len(drivers) == 5
    for sub_drivers in drivers:
        assert len(sub_drivers) == 3
        for d in sub_drivers:
            assert "feature_name" in d
            assert "feature_value" in d
            assert "shap_contribution" in d
            assert d["direction"] in ["increases_churn_risk", "reduces_churn_risk"]

    ram_current = get_process_ram_mb()
    assert ram_current < 600.0, f"RAM {ram_current:.1f} MB exceeds 600 MB"


def test_multi_treatment_causal_uplift(sample_gold_data):
    """Test T-Learner estimation, LCB, and strategic quadrant classification."""
    uplift_model = MultiTreatmentTLearner(random_state=42)
    uplift_model.fit(sample_gold_data)

    tau_p, tau_se, tau_lcb = uplift_model.estimate_cate_and_lcb(sample_gold_data)

    for arm in ACTION_NAMES:
        assert arm in tau_p
        assert arm in tau_lcb
        if arm != "no_action":
            # LCB should strictly be <= Point CATE
            assert np.all(tau_lcb[arm] <= tau_p[arm] + 1e-5)

    quadrants = uplift_model.classify_quadrants(sample_gold_data, tau_p, tau_lcb)
    assert len(quadrants) == len(sample_gold_data)
    valid_quads = {"Persuadable", "Sure_Thing", "Lost_Cause", "Sleeping_Dog"}
    assert set(quadrants).issubset(valid_quads)


def test_eligibility_and_stackelberg_optimizer(sample_gold_data):
    """Test SciPy HiGHS MILP Knapsack solver and budget enforcement."""
    # 1. Fit models
    risk_model = BaselineRiskModel(random_state=42).fit(sample_gold_data)
    uplift_model = MultiTreatmentTLearner(random_state=42).fit(sample_gold_data)

    base_churn = risk_model.predict_proba(sample_gold_data)
    tau_p, tau_se, tau_lcb = uplift_model.estimate_cate_and_lcb(sample_gold_data)

    # 2. Evaluate eligibility matrix
    elig_mat, reasons = evaluate_eligibility_matrix(sample_gold_data, tau_p, tau_lcb)
    assert elig_mat.shape == (len(sample_gold_data), len(ACTION_NAMES))
    # Control action must always be eligible
    assert np.all(elig_mat[:, 0] == 1.0)

    # 3. Solve MILP Knapsack
    budget = 10_000_000.0  # Rp 10 Juta budget
    optimizer = StackelbergBudgetOptimizer(budget_idr=budget)
    payoffs = optimizer.compute_payoff_matrix(sample_gold_data, tau_lcb)

    chosen_indices, meta = optimizer.solve_milp_highs(payoffs, elig_mat, budget=budget)
    assert len(chosen_indices) == len(sample_gold_data)

    # Check budget constraint strictly respected
    assert meta["budget_spent_idr"] <= budget + 1e-4

    chosen_actions = [ACTION_NAMES[idx] for idx in chosen_indices]

    # Verify zero sleeping dogs triggered in GARUDA policy
    comp = evaluate_policy_comparison(
        sample_gold_data,
        chosen_actions,
        tau_p,
        tau_lcb,
        base_churn,
        budget,
    )
    assert comp["garuda_policy"]["sleeping_dogs_triggered"] == 0
    assert comp["garuda_policy"]["net_incremental_value_idr"] >= comp["naive_churn_policy"]["net_incremental_value_idr"]

    ram_current = get_process_ram_mb()
    assert ram_current < 600.0, f"Peak RAM {ram_current:.1f} MB exceeds 600 MB limit"

