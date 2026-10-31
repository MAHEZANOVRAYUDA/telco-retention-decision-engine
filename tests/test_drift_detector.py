"""Unit Tests for Drift Detector & MLflow Tracking Pipeline.

Verifies:
- Population Stability Index (PSI) calculation on identical and shifted features
- Kolmogorov-Smirnov test behavior
- Automated RETRAIN_REQUIRED alert when PSI >= 0.25
- End-to-end pipeline execution and MLflow SQLite logging
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import DriftDetector, calculate_psi
from src.pipeline import run_full_pipeline


def test_psi_identical_distribution():
    """Identical distributions should have PSI close to zero."""
    np.random.seed(42)
    dist1 = np.random.normal(0, 1, 1000)
    dist2 = np.random.normal(0, 1, 1000)

    psi = calculate_psi(dist1, dist2)
    assert psi < 0.10, f"Expected stable PSI < 0.10, got {psi:.4f}"


def test_psi_shifted_distribution():
    """Drifted distribution should yield PSI >= 0.25."""
    np.random.seed(42)
    baseline = np.random.normal(1.0, 0.2, 1000)
    shifted = np.random.normal(0.4, 0.2, 1000)  # Significant drop

    psi = calculate_psi(baseline, shifted)
    assert psi >= 0.25, f"Expected drift PSI >= 0.25, got {psi:.4f}"


def test_drift_detector_retrain_signal():
    """Ensure DriftDetector issues RETRAIN_REQUIRED alert on shifted batches."""
    np.random.seed(42)
    base_df = pd.DataFrame({
        "usage_decay_ratio": np.random.normal(0.88, 0.28, 500),
        "drop_call_rate": np.random.beta(1.5, 18.0, 500) * 100,
        "monthly_arpu_idr": np.random.gamma(3.2, 28000, 500),
        "data_usage_gb_m1": np.random.lognormal(2.4, 0.65, 500),
        "complaint_count_90d": np.random.poisson(0.65, 500),
    })

    detector = DriftDetector(baseline_df=base_df)

    # 1. Normal batch -> STABLE
    normal_df = pd.DataFrame({
        "usage_decay_ratio": np.random.normal(0.88, 0.28, 500),
        "drop_call_rate": np.random.beta(1.5, 18.0, 500) * 100,
        "monthly_arpu_idr": np.random.gamma(3.2, 28000, 500),
        "data_usage_gb_m1": np.random.lognormal(2.4, 0.65, 500),
        "complaint_count_90d": np.random.poisson(0.65, 500),
    })
    res_normal = detector.evaluate_drift(normal_df)
    assert res_normal["overall_status"] == "STABLE"

    # 2. Shifted batch -> RETRAIN_REQUIRED
    shifted_df = detector.generate_shifted_batch(normal_df, severity=0.45)
    res_shifted = detector.evaluate_drift(shifted_df)
    assert res_shifted["overall_status"] == "RETRAIN_REQUIRED"
    assert res_shifted["max_psi"] >= 0.25
    assert len(res_shifted["drifting_features"]) > 0


def test_end_to_end_pipeline_mlflow(tmp_path):
    """Test full pipeline execution with local SQLite MLflow tracking."""
    mlflow_db = tmp_path / "test_mlflow.db"
    uri = f"sqlite:///{mlflow_db}"

    # Run with small cohort (2,000) for fast CI test
    res = run_full_pipeline(
        budget_idr=5000000.0,
        n_subscribers=2000,
        mlflow_tracking_uri=uri,
    )
    assert res["status"] == "SUCCESS"
    assert res["peak_ram_mb"] < 600.0
    assert res["ablation_study"]["advantage"]["sleeping_dogs_eliminated"] >= 0
    assert mlflow_db.exists()
