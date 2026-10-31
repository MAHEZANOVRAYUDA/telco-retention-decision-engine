"""Baseline Churn Risk Classifier & TreeSHAP Feature Attribution.

Uses LightGBM with Isotonic / Platt probability calibration and
shap.TreeExplainer to produce individual top-3 key driver attributions.
Strictly optimized for < 300 MB RAM using float32 downcasting.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.config import GOLD_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("garuda_nba.predictive_risk")

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_arpu_idr",
    "data_usage_gb_m1",
    "usage_decay_ratio",
    "drop_call_rate",
    "complaint_count_90d",
    "is_prepaid",
    "is_annual",
    "arpu_tenure_velocity",
    "high_drop_rate_flag",
    "high_complaint_flag",
]


class BaselineRiskModel:
    """Calibrated LightGBM baseline churn classifier with TreeSHAP explanations."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_names = FEATURE_COLUMNS
        self.lgb_model: Optional[lgb.LGBMClassifier] = None
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.tree_explainer: Optional[shap.TreeExplainer] = None

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and downcast features to float32."""
        missing = [col for col in self.feature_names if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        return df[self.feature_names].values.astype(np.float32)

    def fit(self, df: pd.DataFrame, control_only: bool = True) -> "BaselineRiskModel":
        """Fit calibrated classifier on training data.

        If control_only=True, trains strictly on treatment_arm == 'no_action'
        to isolate natural baseline risk P(Churn=1 | X, T=a0).
        """
        if control_only and "treatment_arm" in df.columns:
            subset = df[df["treatment_arm"] == "no_action"].copy()
            if len(subset) < 100:
                logger.warning("Control sample size is small (<100), training on full population.")
                subset = df.copy()
        else:
            subset = df.copy()

        X = self._prepare_features(subset)
        # Target: 1 = Churned, 0 = Retained
        y_churn = (1 - subset["retained_flag"].values).astype(np.int32)

        logger.info(f"Training Baseline Churn Model on {len(X):,} samples (churn rate: {np.mean(y_churn):.2%})...")

        # Base LightGBM classifier with memory-friendly histogram settings
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=5,
            max_bin=255,
            n_jobs=4,
            random_state=self.random_state,
            objective="binary",
            importance_type="gain",
            verbose=-1,
        )

        # Probability calibration via 3-fold cross validation with isotonic regression
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.lgb_model,
            method="sigmoid",
            cv=3,
        )
        self.calibrated_model.fit(X, y_churn)

        # Fit a standalone LGBM model on all control data for TreeSHAP exact explanations
        self.lgb_model.fit(X, y_churn)
        self.tree_explainer = shap.TreeExplainer(self.lgb_model)
        logger.info("Baseline Churn Model & TreeSHAP Explainer successfully fitted.")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict calibrated probability of churn P(Churn=1 | X)."""
        if self.calibrated_model is None:
            raise RuntimeError("Model is not fitted yet.")
        X = self._prepare_features(df)
        probs = self.calibrated_model.predict_proba(X)[:, 1]
        return np.clip(probs.astype(np.float32), 0.0001, 0.9999)

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute ROC-AUC and Brier score against ground truth."""
        X = self._prepare_features(df)
        y_true = (1 - df["retained_flag"].values).astype(np.int32)
        y_pred = self.predict_proba(df)

        auc = float(roc_auc_score(y_true, y_pred))
        brier = float(brier_score_loss(y_true, y_pred))
        logger.info(f"Evaluation -> ROC-AUC: {auc:.4f} (target >= 0.80) | Brier Score: {brier:.4f} (target <= 0.15)")
        return {"roc_auc": auc, "brier_score": brier}

    def explain_drivers(self, df: pd.DataFrame, top_k: int = 3) -> List[List[Dict[str, Any]]]:
        """Compute individual TreeSHAP attributions and extract Top-K drivers."""
        if self.tree_explainer is None:
            raise RuntimeError("TreeExplainer is not fitted.")
        X = self._prepare_features(df)
        try:
            explanation = self.tree_explainer(X)
            shap_mat = explanation.values
        except Exception:
            shap_values = self.tree_explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_mat = shap_values[1]
            elif len(shap_values.shape) == 3:
                shap_mat = shap_values[:, :, 1]
            else:
                shap_mat = shap_values

        results: List[List[Dict[str, Any]]] = []
        for i in range(len(df)):
            row_shap = shap_mat[i]
            row_features = X[i]
            # Sort by absolute SHAP contribution descending
            ranked_indices = np.argsort(-np.abs(row_shap))[:top_k]

            drivers = []
            for idx in ranked_indices:
                contrib = float(row_shap[idx])
                val = float(row_features[idx])
                fname = self.feature_names[idx]
                direction = "increases_churn_risk" if contrib > 0 else "reduces_churn_risk"
                drivers.append(
                    {
                        "feature_name": fname,
                        "feature_value": round(val, 3),
                        "shap_contribution": round(contrib, 4),
                        "direction": direction,
                    }
                )
            results.append(drivers)
        return results


if __name__ == "__main__":
    gold_file = GOLD_DIR / "fct_subscriber_features.parquet"
    if gold_file.exists():
        df_gold = pd.read_parquet(gold_file)
        model = BaselineRiskModel()
        model.fit(df_gold)
        metrics = model.evaluate(df_gold)
        sample_drivers = model.explain_drivers(df_gold.iloc[:3], top_k=3)
        print("Sample Drivers for first subscriber:", sample_drivers[0])

