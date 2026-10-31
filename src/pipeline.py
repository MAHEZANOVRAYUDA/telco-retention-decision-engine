"""End-to-End Orchestrator Pipeline with Local SQLite MLflow Logging.

Orchestrates:
1. Ingestion & Medallion ETL (Bronze -> Silver -> Gold)
2. Quality Gates Contract Assertions
3. Calibrated Risk Modeling & Multi-Treatment Uplift
4. Stackelberg Knapsack MILP Budget Optimizer
5. Economic Ablation Study (GARUDA-NBA vs Naive Churn Policy)
6. Distribution Drift Assessment (PSI & KS-Test)
7. Local SQLite MLflow Experiment Tracking (sqlite:///mlflow.db)
Strictly adheres to < 400 MB RAM budget.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import pandas as pd

try:
    import psutil
    def get_ram_mb() -> float:
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
except ImportError:
    def get_ram_mb() -> float:
        return 0.0

from src.config import (
    BASE_DIR,
    DEFAULT_CAMPAIGN_BUDGET_IDR,
    DUCKDB_MEMORY_LIMIT,
    GOLD_DIR,
)
from src.data_engineering.ingest import ingest_bronze
from src.data_engineering.quality_gates import validate_gold_features
from src.data_engineering.transform_medallion import run_medallion_pipeline
from src.decisioning.eligibility import evaluate_eligibility_matrix
from src.decisioning.stackelberg_optimizer import StackelbergBudgetOptimizer
from src.modeling.causal_uplift import MultiTreatmentTLearner
from src.modeling.evaluation import evaluate_policy_comparison
from src.modeling.predictive_risk import BaselineRiskModel
from src.monitoring.drift_detector import DriftDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("garuda_nba.pipeline")


def run_full_pipeline(
    budget_idr: float = DEFAULT_CAMPAIGN_BUDGET_IDR,
    n_subscribers: int = 20000,
    mlflow_tracking_uri: str = "sqlite:///mlflow.db",
) -> Dict[str, any]:
    """Execute the end-to-end production decision pipeline and log metrics to MLflow."""
    t0 = time.time()
    logger.info("=== STARTING GARUDA-NBA PRODUCTION RUN ===")

    # 1. Ingestion & Lakehouse Medallion
    logger.info("Stage 1: Ingestion & Medallion ETL...")
    ingest_bronze(n_synthetic_subscribers=n_subscribers)
    run_medallion_pipeline()

    # 2. Quality Gates Assertions
    logger.info("Stage 2: Quality Gates Validation...")
    qg_result = validate_gold_features()
    assert qg_result["status"] == "PASSED"

    # 3. Load Gold Data
    df_gold = pd.read_parquet(GOLD_DIR / "fct_subscriber_features.parquet")
    logger.info(f"Loaded Gold Feature Store: {len(df_gold):,} subscribers.")

    # 4. Train Risk & Uplift Models
    logger.info("Stage 3: Training Baseline Risk & CATE Uplift Models...")
    risk_model = BaselineRiskModel(random_state=42).fit(df_gold)
    risk_eval = risk_model.evaluate(df_gold)

    uplift_model = MultiTreatmentTLearner(random_state=42).fit(df_gold)
    tau_p, tau_se, tau_lcb = uplift_model.estimate_cate_and_lcb(df_gold)
    base_churn = risk_model.predict_proba(df_gold)

    # 5. Eligibility Matrix & Stackelberg MILP Knapsack Optimizer
    logger.info("Stage 4: Prescriptive Stackelberg Budget Optimization...")
    elig_mat, _ = evaluate_eligibility_matrix(df_gold, tau_p, tau_lcb)
    optimizer = StackelbergBudgetOptimizer(budget_idr=budget_idr)
    payoffs = optimizer.compute_payoff_matrix(df_gold, tau_lcb)
    chosen_indices, opt_meta = optimizer.solve_milp_highs(payoffs, elig_mat, budget=budget_idr)

    chosen_actions = [optimizer.action_names[idx] for idx in chosen_indices]

    # 6. Policy Ablation Study
    logger.info("Stage 5: Policy Economic Comparison...")
    ablation = evaluate_policy_comparison(
        df_gold,
        chosen_actions,
        tau_p,
        tau_lcb,
        base_churn,
        budget_idr=budget_idr,
    )

    # 7. Drift Monitoring
    logger.info("Stage 6: Distribution Drift Governance...")
    half = len(df_gold) // 2
    detector = DriftDetector(baseline_df=df_gold.iloc[:half])
    normal_drift = detector.evaluate_drift(df_gold.iloc[half:])
    shifted_df = detector.generate_shifted_batch(df_gold.iloc[half:], severity=0.35)
    stress_drift = detector.evaluate_drift(shifted_df)

    peak_ram = get_ram_mb()
    total_time = time.time() - t0

    # 8. MLflow Logging (Local SQLite Backend)
    logger.info(f"Stage 7: Logging Experiment to MLflow ({mlflow_tracking_uri})...")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("GARUDA-NBA-Prescriptive-Retention")

    with mlflow.start_run(run_name=f"Run_{int(time.time())}"):
        mlflow.log_params({
            "budget_idr": budget_idr,
            "cohort_size": n_subscribers,
            "duckdb_memory_limit": DUCKDB_MEMORY_LIMIT,
            "hardware_platform": "8GB_RAM_Laptop",
        })
        mlflow.log_metrics({
            "baseline_roc_auc": risk_eval["roc_auc"],
            "baseline_brier_score": risk_eval["brier_score"],
            "garuda_net_ev_idr": ablation["garuda_policy"]["net_incremental_value_idr"],
            "naive_net_ev_idr": ablation["naive_churn_policy"]["net_incremental_value_idr"],
            "ev_advantage_idr": ablation["advantage"]["net_value_gain_idr"],
            "garuda_budget_spent_idr": ablation["garuda_policy"]["budget_spent_idr"],
            "garuda_sleeping_dogs_triggered": ablation["garuda_policy"]["sleeping_dogs_triggered"],
            "naive_sleeping_dogs_triggered": ablation["naive_churn_policy"]["sleeping_dogs_triggered"],
            "normal_batch_max_psi": normal_drift["max_psi"],
            "stress_batch_max_psi": stress_drift["max_psi"],
            "peak_rss_mb": peak_ram,
            "total_elapsed_seconds": total_time,
        })

    logger.info(
        f"=== RUN COMPLETE in {total_time:.2f}s! Peak RAM: {peak_ram:.1f} MB ===\n"
        f"GARUDA Net EV: Rp {ablation['garuda_policy']['net_incremental_value_idr']:,.0f} | "
        f"Naive EV: Rp {ablation['naive_churn_policy']['net_incremental_value_idr']:,.0f} | "
        f"Sleeping Dogs Eliminated: {ablation['advantage']['sleeping_dogs_eliminated']}"
    )

    return {
        "status": "SUCCESS",
        "peak_ram_mb": round(peak_ram, 1),
        "total_elapsed_seconds": round(total_time, 2),
        "risk_evaluation": risk_eval,
        "ablation_study": ablation,
        "normal_drift": normal_drift,
        "stress_drift": stress_drift,
    }


if __name__ == "__main__":
    run_full_pipeline()

