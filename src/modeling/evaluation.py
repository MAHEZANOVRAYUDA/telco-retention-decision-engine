"""Comprehensive Evaluation Module: Qini Curve, AUUC, Uplift@K% & Policy Comparison.

Compares:
1. GARUDA-NBA (Stackelberg Game-Theoretic Prescriptive Allocation)
2. Traditional Churn-Risk Targeting (Naive ranking on P(Churn) without causality)
3. Random Intervention Baseline
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import ACTION_COSTS, ACTION_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("garuda_nba.evaluation")


def compute_qini_curve(
    y_true: np.ndarray,
    uplift_score: np.ndarray,
    is_treated: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute Qini Curve and Normalized Qini Coefficient.

    Qini(t) = Y_treated(t) - (Y_control(t) * N_treated(t) / N_control(t))
    """
    order = np.argsort(-uplift_score)
    y_sorted = y_true[order]
    t_sorted = is_treated[order]

    cum_y_treated = np.cumsum(y_sorted * t_sorted)
    cum_n_treated = np.cumsum(t_sorted)
    cum_y_ctrl = np.cumsum(y_sorted * (1 - t_sorted))
    cum_n_ctrl = np.cumsum(1 - t_sorted)

    # Avoid zero division
    scale = np.where(cum_n_ctrl > 0, cum_n_treated / np.maximum(cum_n_ctrl, 1), 1.0)
    qini_curve = cum_y_treated - (cum_y_ctrl * scale)

    # Normalized Qini score
    total_qini = float(qini_curve[-1]) if len(qini_curve) > 0 else 0.0
    auuc = float(np.mean(qini_curve) / np.maximum(total_qini, 1.0)) if total_qini > 0 else 0.0

    bins = np.linspace(0, 1, n_bins + 1)
    indices = (bins * (len(y_true) - 1)).astype(int)
    return bins, qini_curve[indices], auuc


def evaluate_policy_comparison(
    df: pd.DataFrame,
    garuda_chosen_actions: List[str],
    tau_preds: Dict[str, np.ndarray],
    tau_lcb: Dict[str, np.ndarray],
    baseline_churn_prob: np.ndarray,
    budget_idr: float,
) -> Dict[str, Any]:
    """Run an economic ablation study comparing GARUDA-NBA vs Naive Churn Policy.

    - GARUDA-NBA: Stackelberg MILP with LCB and eligibility constraints.
    - Naive Churn Policy: Sort subscribers by churn risk descending, allocate
      combo_discount_20pct until budget is exhausted.
    """
    n_sub = len(df)
    cltvs = df["cltv_idr"].values

    # 1. Evaluate GARUDA-NBA
    garuda_spend = 0.0
    garuda_net_ev = 0.0
    garuda_sleeping_dogs = 0
    garuda_persuadables = 0

    for i in range(n_sub):
        act = garuda_chosen_actions[i]
        cost = ACTION_COSTS[act]
        garuda_spend += cost
        if act != "no_action":
            uplift_val = tau_lcb[act][i]
            net_val = (uplift_val * cltvs[i]) - cost
            garuda_net_ev += net_val
            if tau_preds[act][i] < 0:
                garuda_sleeping_dogs += 1
            if uplift_val >= 0.01:
                garuda_persuadables += 1

    # 2. Evaluate Naive Churn-Risk Policy
    naive_action = "combo_discount_20pct"
    naive_cost = ACTION_COSTS[naive_action]
    risk_order = np.argsort(-baseline_churn_prob)
    max_naive_targets = int(budget_idr // naive_cost)
    naive_targeted_indices = risk_order[:max_naive_targets]

    naive_spend = len(naive_targeted_indices) * naive_cost
    naive_net_ev = 0.0
    naive_sleeping_dogs = 0
    naive_persuadables = 0

    for idx in naive_targeted_indices:
        uplift_val = tau_lcb[naive_action][idx]
        net_val = (uplift_val * cltvs[idx]) - naive_cost
        naive_net_ev += net_val
        if tau_preds[naive_action][idx] < 0:
            naive_sleeping_dogs += 1
        if uplift_val >= 0.01:
            naive_persuadables += 1

    ev_advantage = garuda_net_ev - naive_net_ev

    result = {
        "budget_limit_idr": budget_idr,
        "garuda_policy": {
            "budget_spent_idr": garuda_spend,
            "net_incremental_value_idr": round(garuda_net_ev, 0),
            "sleeping_dogs_triggered": garuda_sleeping_dogs,
            "persuadables_retained": garuda_persuadables,
        },
        "naive_churn_policy": {
            "budget_spent_idr": naive_spend,
            "net_incremental_value_idr": round(naive_net_ev, 0),
            "sleeping_dogs_triggered": naive_sleeping_dogs,
            "persuadables_retained": naive_persuadables,
        },
        "advantage": {
            "net_value_gain_idr": round(ev_advantage, 0),
            "sleeping_dogs_eliminated": naive_sleeping_dogs - garuda_sleeping_dogs,
        },
    }
    logger.info(
        f"Ablation Result: GARUDA EV Rp {garuda_net_ev:,.0f} vs Naive EV Rp {naive_net_ev:,.0f} "
        f"(Advantage: +Rp {ev_advantage:,.0f})"
    )
    return result
