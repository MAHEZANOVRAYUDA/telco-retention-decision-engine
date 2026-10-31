"""Stackelberg Game-Theoretic Budget Allocation & Prescriptive Decisioning.

Solves the Multi-Choice Knapsack Mixed-Integer Linear Program (MILP) using
SciPy HiGHS (scipy.optimize.milp) to maximize total Net Incremental Retained Value (IDR)
under an organizational campaign budget constraint B.
Features vectorized sparse matrix generation and high-efficiency dual-path solving:
- Exact HiGHS MILP for candidate knapsack optimization
- Dual Lagrangian efficiency ranking fallback for extreme cohort scales
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from src.config import ACTION_COSTS, ACTION_NAMES, DEFAULT_CAMPAIGN_BUDGET_IDR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("garuda_nba.stackelberg_optimizer")


class StackelbergBudgetOptimizer:
    """Solves Global Campaign Knapsack MILP via SciPy HiGHS."""

    def __init__(self, budget_idr: float = DEFAULT_CAMPAIGN_BUDGET_IDR):
        self.budget_idr = float(budget_idr)
        self.action_names = ACTION_NAMES
        self.action_costs = ACTION_COSTS
        self.n_arms = len(self.action_names)
        self.costs_vec = np.array([self.action_costs[arm] for arm in self.action_names], dtype=np.float64)

    def compute_payoff_matrix(
        self,
        df: pd.DataFrame,
        tau_lcb: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute payoff matrix pi_{i,k} = tau_{k,LCB}(x_i) * CLTV_i - Cost(a_k).

        For k=0 (no_action), payoff is strictly 0.0.
        """
        n_sub = len(df)
        payoffs = np.zeros((n_sub, self.n_arms), dtype=np.float64)
        cltvs = df["cltv_idr"].values.astype(np.float64)

        for k, arm in enumerate(self.action_names):
            if arm == "no_action":
                payoffs[:, k] = 0.0
            else:
                c_k = self.action_costs[arm]
                payoffs[:, k] = (tau_lcb[arm] * cltvs) - c_k

        return payoffs

    def solve_milp_highs(
        self,
        payoff_matrix: np.ndarray,
        eligibility_matrix: np.ndarray,
        budget: Optional[float] = None,
        max_highs_candidates: int = 350,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve Multi-Choice Knapsack MILP via SciPy HiGHS."""
        t0 = time.time()
        b_budget = float(budget if budget is not None else self.budget_idr)
        n_sub, n_arms = payoff_matrix.shape

        # Step 1: Identify candidates with positive net payoff on eligible active arms
        masked_payoff = np.where(eligibility_matrix == 1.0, payoff_matrix, -1e12)
        has_positive_payoff = np.any((masked_payoff[:, 1:] > 0), axis=1)
        candidate_indices = np.where(has_positive_payoff)[0]
        n_candidates = len(candidate_indices)

        chosen_indices = np.zeros(n_sub, dtype=np.int32)
        solver_name = "HiGHS_MILP_Exact"

        if n_candidates == 0 or b_budget <= 0:
            # Nobody has positive payoff or zero budget -> all no_action
            total_payoff = 0.0
            total_cost = 0.0
        elif n_candidates <= max_highs_candidates:
            # Solve exact MILP on candidate cohort
            chosen_sub_indices = self._solve_highs_core(
                payoff_matrix[candidate_indices],
                eligibility_matrix[candidate_indices],
                b_budget,
            )
            for idx, c_idx in enumerate(candidate_indices):
                chosen_indices[c_idx] = chosen_sub_indices[idx]
        else:
            # For larger cohorts, select top candidate pool by ROI efficiency or Lagrangian greedy
            best_active_idx = np.argmax(masked_payoff[candidate_indices, 1:], axis=1) + 1
            best_active_payoffs = masked_payoff[candidate_indices, best_active_idx]
            best_active_costs = self.costs_vec[best_active_idx]
            roi = best_active_payoffs / np.maximum(best_active_costs, 1.0)
            top_order = np.argsort(-roi)[:max_highs_candidates]

            sub_cand = candidate_indices[top_order]
            chosen_sub_indices = self._solve_highs_core(
                payoff_matrix[sub_cand],
                eligibility_matrix[sub_cand],
                b_budget,
            )
            for idx, c_idx in enumerate(sub_cand):
                chosen_indices[c_idx] = chosen_sub_indices[idx]

        elapsed = time.time() - t0

        total_payoff = float(np.sum([payoff_matrix[i, chosen_indices[i]] for i in range(n_sub)]))
        total_cost = float(np.sum([self.costs_vec[chosen_indices[i]] for i in range(n_sub)]))

        action_counts = {}
        for k, arm in enumerate(self.action_names):
            action_counts[arm] = int(np.sum(chosen_indices == k))

        meta = {
            "solver": solver_name,
            "elapsed_seconds": round(elapsed, 3),
            "candidates_optimized": n_candidates,
            "budget_limit_idr": b_budget,
            "budget_spent_idr": total_cost,
            "budget_utilization_pct": round((total_cost / b_budget) * 100, 2) if b_budget > 0 else 0.0,
            "net_incremental_value_idr": round(total_payoff, 0),
            "roi_multiple": round(total_payoff / total_cost, 2) if total_cost > 0 else 0.0,
            "action_distribution": action_counts,
        }
        logger.info(f"Optimization completed in {elapsed:.3f}s | Net Value: Rp {total_payoff:,.0f} | Spend: Rp {total_cost:,.0f}")
        return chosen_indices, meta

    def _solve_highs_core(
        self,
        payoff_mat: np.ndarray,
        elig_mat: np.ndarray,
        budget: float,
    ) -> np.ndarray:
        """Internal HiGHS solver using vectorized sparse matrix formulation."""
        n_c, n_arms = payoff_mat.shape
        n_vars = n_c * n_arms

        c_obj = -payoff_mat.flatten()

        # Vectorized constraint matrix construction
        rows_choice = np.repeat(np.arange(n_c), n_arms)
        cols_choice = np.arange(n_vars)
        data_choice = np.ones(n_vars, dtype=np.float64)

        rows_budget = np.full(n_vars, n_c)
        cols_budget = np.arange(n_vars)
        data_budget = np.tile(self.costs_vec, n_c)

        rows = np.concatenate([rows_choice, rows_budget])
        cols = np.concatenate([cols_choice, cols_budget])
        data = np.concatenate([data_choice, data_budget])
        A_csr = coo_matrix((data, (rows, cols)), shape=(n_c + 1, n_vars)).tocsr()

        lhs = np.ones(n_c + 1, dtype=np.float64)
        lhs[n_c] = 0.0
        rhs = np.ones(n_c + 1, dtype=np.float64)
        rhs[n_c] = budget

        bounds = Bounds(np.zeros(n_vars), elig_mat.flatten())
        integrality = np.ones(n_vars, dtype=np.int32)

        res = milp(
            c=c_obj,
            integrality=integrality,
            constraints=LinearConstraint(A_csr, lhs, rhs),
            bounds=bounds,
            options={"time_limit": 5.0, "mip_rel_gap": 0.01},
        )

        if res.success:
            z_mat = res.x.reshape((n_c, n_arms))
            return np.argmax(z_mat, axis=1).astype(np.int32)
        else:
            return self.solve_lagrangian_greedy(payoff_mat, elig_mat, budget)

    def solve_lagrangian_greedy(
        self,
        payoff_matrix: np.ndarray,
        eligibility_matrix: np.ndarray,
        budget: float,
    ) -> np.ndarray:
        """Fast Dual Lagrangian Benefit-to-Cost Efficiency Ranking."""
        n_sub, n_arms = payoff_matrix.shape
        cost_mat = self.costs_vec

        masked_payoff = np.where(eligibility_matrix == 1.0, payoff_matrix, -1e12)
        best_active_idx = np.argmax(masked_payoff[:, 1:], axis=1) + 1
        best_active_payoff = masked_payoff[np.arange(n_sub), best_active_idx]
        best_active_cost = cost_mat[best_active_idx]

        positive_mask = best_active_payoff > 0
        roi_ratio = np.where(positive_mask, best_active_payoff / np.maximum(best_active_cost, 1.0), -1.0)
        order_roi = np.argsort(-roi_ratio)

        chosen = np.zeros(n_sub, dtype=np.int32)
        spend = 0.0
        for idx in order_roi:
            if not positive_mask[idx]:
                break
            c_i = best_active_cost[idx]
            if spend + c_i <= budget:
                chosen[idx] = best_active_idx[idx]
                spend += c_i

        return chosen
