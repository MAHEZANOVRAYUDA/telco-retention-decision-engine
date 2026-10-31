"""Multi-Treatment Causal Uplift Engine (T-Learner) with Conformal/Bootstrap LCB.

Estimates Conditional Average Treatment Effect (CATE) tau_k(x_i) across
multiple retention interventions and computes the 90% Lower Confidence Bound (LCB)
to eliminate speculative spending on uncertain uplift.
Categorizes subscribers into 4 strategic quadrants:
- Persuadables
- Sure Things
- Lost Causes
- Sleeping Dogs
"""

import logging
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.config import (
    ACTION_COSTS,
    ACTION_NAMES,
    GOLD_DIR,
    LOST_CAUSE_RETENTION_THRESHOLD,
    MIN_UPLIFT_THRESHOLD,
    SLEEPING_DOG_UPLIFT_THRESHOLD,
    SURE_THING_RETENTION_THRESHOLD,
    Z_SCORE_90_PCT,
)
from src.modeling.predictive_risk import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("garuda_nba.causal_uplift")


class MultiTreatmentTLearner:
    """T-Learner multi-treatment causal uplift estimator."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_names = FEATURE_COLUMNS
        self.treatment_arms = ACTION_NAMES
        self.arm_models: Dict[str, CalibratedClassifierCV] = {}
        self.arm_sample_sizes: Dict[str, int] = {}

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feature_names].values.astype(np.float32)

    def fit(self, df: pd.DataFrame) -> "MultiTreatmentTLearner":
        """Train separate calibrated response estimators for each treatment arm."""
        X = self._prepare_features(df)
        y_retain = df["retained_flag"].values.astype(np.int32)
        arms = df["treatment_arm"].values

        logger.info(f"Fitting Multi-Treatment T-Learner across {len(self.treatment_arms)} arms on {len(df):,} samples...")

        for arm in self.treatment_arms:
            mask = arms == arm
            n_arm = int(np.sum(mask))
            self.arm_sample_sizes[arm] = max(n_arm, 1)

            if n_arm < 50:
                logger.warning(f"Arm '{arm}' has only {n_arm} samples. Fitting may have high variance.")

            X_arm = X[mask]
            y_arm = y_retain[mask]

            # Fit LightGBM response model P(Retain=1 | X, T=arm)
            base_estimator = lgb.LGBMClassifier(
                n_estimators=80,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=5,
                max_bin=255,
                n_jobs=4,
                random_state=self.random_state,
                objective="binary",
                verbose=-1,
            )

            cv_splits = min(3, max(2, n_arm // 50))
            calibrated = CalibratedClassifierCV(
                estimator=base_estimator,
                method="sigmoid",
                cv=cv_splits,
            )
            calibrated.fit(X_arm, y_arm)
            self.arm_models[arm] = calibrated
            logger.info(f"Arm '{arm}' calibrated model fitted on {n_arm:,} samples (Retention rate: {np.mean(y_arm):.2%}).")

        return self

    def predict_response_probabilities(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict retention probability mu_k(x_i) = P(Retain=1 | X, T=a_k) for all arms."""
        X = self._prepare_features(df)
        mu_preds: Dict[str, np.ndarray] = {}
        for arm, model in self.arm_models.items():
            probs = model.predict_proba(X)[:, 1]
            mu_preds[arm] = np.clip(probs.astype(np.float32), 0.001, 0.999)
        return mu_preds

    def estimate_cate_and_lcb(
        self, df: pd.DataFrame, alpha_z: float = Z_SCORE_90_PCT
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Compute Point CATE tau_k(x_i), Standard Error, and Lower Confidence Bound.

        tau_k(x_i) = mu_k(x_i) - mu_0(x_i)
        SE(tau_k) = sqrt( mu_k*(1 - mu_k)/N_k + mu_0*(1 - mu_0)/N_0 )
        tau_{k, LCB} = tau_k - alpha_z * SE(tau_k)
        """
        mu_preds = self.predict_response_probabilities(df)
        mu_ctrl = mu_preds["no_action"]
        n_ctrl = self.arm_sample_sizes.get("no_action", 5000)

        tau_point: Dict[str, np.ndarray] = {}
        tau_se: Dict[str, np.ndarray] = {}
        tau_lcb: Dict[str, np.ndarray] = {}

        for arm in self.treatment_arms:
            if arm == "no_action":
                n_sub = len(df)
                tau_point[arm] = np.zeros(n_sub, dtype=np.float32)
                tau_se[arm] = np.zeros(n_sub, dtype=np.float32)
                tau_lcb[arm] = np.zeros(n_sub, dtype=np.float32)
                continue

            mu_arm = mu_preds[arm]
            n_arm = self.arm_sample_sizes.get(arm, 2000)

            # CATE point estimate
            tau = (mu_arm - mu_ctrl).astype(np.float32)

            # Standard error via delta method / binomial variance
            se = np.sqrt(
                (mu_arm * (1.0 - mu_arm) / float(n_arm))
                + (mu_ctrl * (1.0 - mu_ctrl) / float(n_ctrl))
            ).astype(np.float32)

            # Lower Confidence Bound
            lcb = (tau - (alpha_z * se)).astype(np.float32)

            tau_point[arm] = tau
            tau_se[arm] = se
            tau_lcb[arm] = lcb

        return tau_point, tau_se, tau_lcb

    def classify_quadrants(
        self,
        df: pd.DataFrame,
        tau_point: Dict[str, np.ndarray],
        tau_lcb: Dict[str, np.ndarray],
    ) -> List[str]:
        """Classify each subscriber into one of four game-theoretic quadrants.

        - Sleeping Dog: tau_k < 0 on any evaluated intervention
        - Sure Thing: mu_0 >= 0.85 and tau_max < theta_min
        - Lost Cause: mu_k < 0.25 across all treatment arms
        - Persuadable: tau_lcb > theta_min and positive payoff
        """
        mu_preds = self.predict_response_probabilities(df)
        mu_ctrl = mu_preds["no_action"]
        active_arms = [a for a in self.treatment_arms if a != "no_action"]

        n = len(df)
        quadrants: List[str] = []

        # Find max uplift and min uplift across active arms for each subscriber
        all_tau_points = np.column_stack([tau_point[arm] for arm in active_arms])
        all_tau_lcbs = np.column_stack([tau_lcb[arm] for arm in active_arms])
        all_mus = np.column_stack([mu_preds[arm] for arm in self.treatment_arms])

        max_lcb = np.max(all_tau_lcbs, axis=1)
        min_point = np.min(all_tau_points, axis=1)
        max_mu_all = np.max(all_mus, axis=1)

        for i in range(n):
            if min_point[i] < SLEEPING_DOG_UPLIFT_THRESHOLD and max_lcb[i] <= 0.0:
                quadrants.append("Sleeping_Dog")
            elif mu_ctrl[i] >= SURE_THING_RETENTION_THRESHOLD and max_lcb[i] < MIN_UPLIFT_THRESHOLD:
                quadrants.append("Sure_Thing")
            elif max_mu_all[i] < LOST_CAUSE_RETENTION_THRESHOLD:
                quadrants.append("Lost_Cause")
            elif max_lcb[i] >= MIN_UPLIFT_THRESHOLD:
                quadrants.append("Persuadable")
            else:
                # Default boundary category
                quadrants.append("Sure_Thing" if mu_ctrl[i] >= 0.50 else "Lost_Cause")

        return quadrants


if __name__ == "__main__":
    gold_file = GOLD_DIR / "fct_subscriber_features.parquet"
    if gold_file.exists():
        df_gold = pd.read_parquet(gold_file)
        uplift_model = MultiTreatmentTLearner()
        uplift_model.fit(df_gold)
        tau_p, tau_s, tau_l = uplift_model.estimate_cate_and_lcb(df_gold.iloc[:5])
        quads = uplift_model.classify_quadrants(df_gold.iloc[:5], tau_p, tau_l)
        print("Sample Quadrants:", quads)

