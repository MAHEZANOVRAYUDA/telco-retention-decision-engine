"""Unit Tests for Data Engineering Lakehouse & Quality Gates.

Validates:
- Bronze ingestion (Hugging Face / calibrated RCT)
- Silver & Gold Medallion transformations
- Quality gates (0 duplicates, 0 negative values, 0 target leakage)
- Memory profile < 400 MB RSS
"""

import os
from pathlib import Path

import duckdb
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

from src.config import BRONZE_DIR, GOLD_DIR, SILVER_DIR, TARGET_RAM_MB
from src.data_engineering.ingest import ingest_bronze
from src.data_engineering.quality_gates import (
    QualityGateViolation,
    validate_gold_features,
)
from src.data_engineering.transform_medallion import run_medallion_pipeline


def test_ingestion_and_medallion_pipeline(tmp_path):
    """Test full pipeline from ingestion to gold feature store."""
    ram_initial = get_process_ram_mb()

    # 1. Ingest bronze (test size 2,000 for fast testing)
    b_test = tmp_path / "raw_subscribers.parquet"
    ingest_bronze(destination_file=b_test, n_synthetic_subscribers=2000, force_synthetic=True)
    assert b_test.exists()

    # 2. Run medallion transforms
    s_test = tmp_path / "stg_validated_subscribers.parquet"
    g_test = tmp_path / "fct_subscriber_features.parquet"

    from src.data_engineering.transform_medallion import (
        transform_bronze_to_silver,
        transform_silver_to_gold,
    )

    transform_bronze_to_silver(bronze_path=b_test, silver_path=s_test)
    assert s_test.exists()

    transform_silver_to_gold(silver_path=s_test, gold_path=g_test)
    assert g_test.exists()

    # 3. Quality gate validation
    result = validate_gold_features(gold_path=g_test)
    assert result["status"] == "PASSED"
    assert result["total_subscribers"] == 2000
    assert result["distinct_subscribers"] == 2000
    assert result["null_count"] == 0

    # 4. Check RAM
    ram_current = get_process_ram_mb()
    assert ram_current < TARGET_RAM_MB, f"RAM {ram_current:.1f} MB exceeds target {TARGET_RAM_MB} MB"


def test_quality_gate_detects_duplicates(tmp_path):
    """Ensure quality gates fail if duplicate subscriber_id is present."""
    bad_df = pd.DataFrame(
        {
            "subscriber_id": ["SUB-001", "SUB-001"],  # Duplicate!
            "tenure_months": [12, 24],
            "monthly_arpu_idr": [50000.0, 60000.0],
            "data_usage_gb_m1": [10.0, 15.0],
            "usage_decay_ratio": [1.0, 0.9],
            "drop_call_rate": [1.2, 2.3],
            "complaint_count_90d": [0, 1],
            "cltv_idr": [200000.0, 300000.0],
            "treatment_arm": ["no_action", "data_booster_15gb"],
            "retained_flag": [1, 1],
        }
    )
    bad_path = tmp_path / "bad_dup.parquet"
    bad_df.to_parquet(bad_path)

    with pytest.raises(QualityGateViolation, match="Duplicate subscriber_id"):
        validate_gold_features(gold_path=bad_path)


def test_quality_gate_detects_negative_arpu(tmp_path):
    """Ensure quality gates fail if negative ARPU is present."""
    bad_df = pd.DataFrame(
        {
            "subscriber_id": ["SUB-001", "SUB-002"],
            "tenure_months": [12, 24],
            "monthly_arpu_idr": [-500.0, 60000.0],  # Negative!
            "data_usage_gb_m1": [10.0, 15.0],
            "usage_decay_ratio": [1.0, 0.9],
            "drop_call_rate": [1.2, 2.3],
            "complaint_count_90d": [0, 1],
            "cltv_idr": [200000.0, 300000.0],
            "treatment_arm": ["no_action", "data_booster_15gb"],
            "retained_flag": [1, 1],
        }
    )
    bad_path = tmp_path / "bad_arpu.parquet"
    bad_df.to_parquet(bad_path)

    with pytest.raises(QualityGateViolation, match="Negative monthly_arpu_idr"):
        validate_gold_features(gold_path=bad_path)

