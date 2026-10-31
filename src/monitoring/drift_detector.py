"""MLOps Drift Monitoring & Automated Retraining Governance.

Implements:
1. Population Stability Index (PSI) with robust quantile binning
2. Kolmogorov-Smirnov 2-sample distributional hypothesis test
3. Automated RETRAIN_REQUIRED alert when max(PSI) >= 0.25
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import PSI_MODERATE_SHIFT, PSI_RETRAIN_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("garuda_nba.drift_detector")

DEFAULT_MONITORED_FEATURES = [
    "usage_decay_ratio",
    "drop_call_rate",
    "monthly_arpu_idr",
    "data_usage_gb_m1",
    "complaint_count_90d",
]


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between baseline and production distributions.

    PSI = sum( (Actual% - Expected%) * ln(Actual% / Expected%) )
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Ensure 1D float arrays
    exp = np.asarray(expected, dtype=np.float64)
    act = np.asarray(actual, dtype=np.float64)

    # Compute quantile breakpoints from expected distribution
    breakpoints = np.quantile(exp, np.linspace(0, 1, bins + 1))
    breakpoints[0] -= 1e-5
    breakpoints[-1] += 1e-5

    # Eliminate identical breakpoints in discrete / zero-heavy distributions
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 2:
        breakpoints = np.linspace(np.min(exp) - 1e-5, np.max(exp) + 1e-5, bins + 1)

    exp_counts, _ = np.histogram(exp, bins=breakpoints)
    act_counts, _ = np.histogram(act, bins=breakpoints)

    exp_percents = exp_counts / float(len(exp))
    act_percents = act_counts / float(len(act))

    # Epsilon clipping to prevent log(0) or division by zero
    exp_percents = np.clip(exp_percents, 1e-4, None)
    act_percents = np.clip(act_percents, 1e-4, None)

    psi_val = np.sum((act_percents - exp_percents) * np.log(act_percents / exp_percents))
    return float(psi_val)


class DriftDetector:
    """Monitors feature and prediction distributions for drift and issues retrain signals."""

    def __init__(
        self,
        baseline_df: pd.DataFrame,
        monitored_features: Optional[List[str]] = None,
        retrain_threshold: float = PSI_RETRAIN_THRESHOLD,
    ):
        self.baseline_df = baseline_df.copy()
        self.features = monitored_features or [f for f in DEFAULT_MONITORED_FEATURES if f in baseline_df.columns]
        self.retrain_threshold = retrain_threshold

    def evaluate_drift(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate PSI and KS-test across monitored features."""
        feature_reports: Dict[str, Any] = {}
        max_psi = 0.0
        retrain_flags: List[str] = []

        for f in self.features:
            if f not in current_df.columns:
                logger.warning(f"Feature '{f}' missing in production batch, skipping.")
                continue

            exp_vals = self.baseline_df[f].values
            act_vals = current_df[f].values

            psi_score = calculate_psi(exp_vals, act_vals)
            ks_res = ks_2samp(exp_vals, act_vals)
            ks_stat = float(ks_res.statistic)
            ks_pval = float(ks_res.pvalue)

            max_psi = max(max_psi, psi_score)

            if psi_score >= self.retrain_threshold:
                feature_status = "DRIFT_CRITICAL"
                retrain_flags.append(f)
            elif psi_score >= PSI_MODERATE_SHIFT:
                feature_status = "MODERATE_SHIFT"
            else:
                feature_status = "STABLE"

            feature_reports[f] = {
                "psi": round(psi_score, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": round(ks_pval, 4),
                "status": feature_status,
            }

        overall_status = "RETRAIN_REQUIRED" if max_psi >= self.retrain_threshold else (
            "MODERATE_SHIFT" if max_psi >= PSI_MODERATE_SHIFT else "STABLE"
        )

        logger.info(f"Drift assessment: Status={overall_status} | Max PSI={max_psi:.4f} (Threshold={self.retrain_threshold})")
        return {
            "overall_status": overall_status,
            "max_psi": round(max_psi, 4),
            "retrain_threshold": self.retrain_threshold,
            "drifting_features": retrain_flags,
            "baseline_samples": len(self.baseline_df),
            "current_samples": len(current_df),
            "feature_details": feature_reports,
        }

    def generate_shifted_batch(self, current_df: pd.DataFrame, severity: float = 0.40) -> pd.DataFrame:
        """Simulate a severe network & macroeconomic shock batch for drift testing."""
        shifted = current_df.copy()
        if "usage_decay_ratio" in shifted.columns:
            # Significant usage decay shock (e.g. competitor promo launch)
            shifted["usage_decay_ratio"] = np.clip(shifted["usage_decay_ratio"].values * (1.0 - severity), 0.05, 3.0)
        if "drop_call_rate" in shifted.columns:
            # BTS network congestion anomaly (+12% dropped calls)
            shifted["drop_call_rate"] = np.clip(shifted["drop_call_rate"].values + (severity * 30.0), 0.0, 100.0)
        if "complaint_count_90d" in shifted.columns:
            shifted["complaint_count_90d"] = np.clip(shifted["complaint_count_90d"].values + 3, 0, None)
        return shifted

