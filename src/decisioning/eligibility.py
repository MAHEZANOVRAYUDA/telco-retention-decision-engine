"""Business, Technical & Behavioral Eligibility Matrix for Retention Interventions.

Enforces:
1. No negative uplift interventions (Zero Sleeping Dog triggers)
2. Technical suitability (QoS ticket only when drop rate or complaints present)
3. Commercial guardrails (ARPU thresholds and usage decay triggers)
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    ACTION_NAMES,
    MIN_UPLIFT_THRESHOLD,
    SLEEPING_DOG_UPLIFT_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("garuda_nba.eligibility")


def evaluate_eligibility_matrix(
    df: pd.DataFrame,
    tau_point: Dict[str, np.ndarray],
    tau_lcb: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """Evaluate eligibility mask (N x K) and audit reasons for all subscribers.

    Returns:
    - elig_matrix: np.ndarray of shape (N, K) with 1.0 (eligible) or 0.0 (ineligible)
    - reasons: list of dicts mapping action -> eligibility reason for each subscriber
    """
    n_sub = len(df)
    n_arms = len(ACTION_NAMES)
    elig_matrix = np.zeros((n_sub, n_arms), dtype=np.float64)
    reasons_list: List[Dict[str, str]] = []

    drop_rates = df["drop_call_rate"].values
    complaints = df["complaint_count_90d"].values
    decay_ratios = df["usage_decay_ratio"].values
    data_usages = df["data_usage_gb_m1"].values
    arpus = df["monthly_arpu_idr"].values

    for i in range(n_sub):
        sub_reasons: Dict[str, str] = {}
        for k, arm in enumerate(ACTION_NAMES):
            if arm == "no_action":
                # Control is always eligible
                elig_matrix[i, k] = 1.0
                sub_reasons[arm] = "Default baseline holdout."
                continue

            tau_p = float(tau_point[arm][i])
            tau_l = float(tau_lcb[arm][i])

            # Hard Guardrail 1: Prohibit Sleeping Dogs (negative uplift)
            if tau_p < SLEEPING_DOG_UPLIFT_THRESHOLD or tau_l <= 0.0:
                elig_matrix[i, k] = 0.0
                sub_reasons[arm] = f"Ineligible: Non-positive causal uplift ({tau_p*100:.1f}%), risk of sleeping dog reaction."
                continue

            # Hard Guardrail 2: Minimum positive threshold
            if tau_l < MIN_UPLIFT_THRESHOLD:
                elig_matrix[i, k] = 0.0
                sub_reasons[arm] = f"Ineligible: Lower confidence bound {tau_l*100:.2f}% below threshold {MIN_UPLIFT_THRESHOLD*100:.1f}%."
                continue

            # Technical Guardrail: QoS Priority Care
            if arm == "qos_priority_care":
                is_tech_candidate = (drop_rates[i] >= 3.5) or (complaints[i] >= 1)
                if not is_tech_candidate:
                    elig_matrix[i, k] = 0.0
                    sub_reasons[arm] = f"Ineligible: Network drop rate ({drop_rates[i]:.1f}%) and complaints ({complaints[i]}) are within normal parameters."
                    continue

            # Commercial Guardrail: Data Booster
            if arm == "data_booster_15gb":
                is_data_candidate = (decay_ratios[i] <= 0.95) or (data_usages[i] >= 3.0)
                if not is_data_candidate:
                    elig_matrix[i, k] = 0.0
                    sub_reasons[arm] = "Ineligible: Data usage and decay ratio do not warrant booster inject."
                    continue

            # Commercial Guardrail: Combo Discount 20%
            if arm == "combo_discount_20pct":
                if arpus[i] < 45000.0:
                    elig_matrix[i, k] = 0.0
                    sub_reasons[arm] = f"Ineligible: ARPU (Rp {arpus[i]:,.0f}) too low for contract discount subsidy."
                    continue

            # Passed all filters
            elig_matrix[i, k] = 1.0
            sub_reasons[arm] = f"Eligible: Positive uplift ({tau_p*100:.1f}%) and meets technical/commercial criteria."

        reasons_list.append(sub_reasons)

    return elig_matrix, reasons_list
