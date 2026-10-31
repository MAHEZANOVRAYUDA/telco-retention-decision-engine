"""Data Quality Gates and Boundary Contract Assertions for GARUDA-NBA.

Validates:
1. Zero duplicate subscriber_id
2. Zero negative values on monthly_arpu_idr, tenure_months, cltv_idr
3. Zero nulls in essential columns
4. Value domain boundaries (drop_call_rate in [0, 100], retained_flag in {0, 1})
5. Zero feature target leakage
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import pyarrow.parquet as pq

from src.config import ACTION_NAMES, GOLD_DIR, SILVER_DIR

logger = logging.getLogger("garuda_nba.quality_gates")


class QualityGateViolation(Exception):
    """Raised when a data contract quality gate fails."""
    pass


def validate_gold_features(gold_path: Optional[Path] = None) -> Dict[str, any]:
    """Execute deterministic quality gates against the Gold feature table.

    Returns summary metrics if all assertions pass.
    Raises QualityGateViolation if any contract is breached.
    """
    g_path = gold_path or (GOLD_DIR / "fct_subscriber_features.parquet")
    if not g_path.exists():
        raise FileNotFoundError(f"Gold feature store not found at {g_path}")

    g_str = str(g_path).replace("\\", "/")
    conn = duckdb.connect(database=":memory:")

    # 1. Row count & schema
    meta = conn.execute(f"SELECT COUNT(*) AS total_rows, COUNT(DISTINCT subscriber_id) AS distinct_subs FROM read_parquet('{g_str}')").fetchone()
    total_rows, distinct_subs = meta[0], meta[1]

    if total_rows == 0:
        raise QualityGateViolation("Gold feature table is empty (0 rows).")

    # 2. Assert Zero duplicate subscriber_id
    if total_rows != distinct_subs:
        dup_count = total_rows - distinct_subs
        raise QualityGateViolation(f"Duplicate subscriber_id detected! Duplicates: {dup_count}")

    # 3. Assert Zero nulls in core columns
    core_cols = [
        "subscriber_id",
        "tenure_months",
        "monthly_arpu_idr",
        "data_usage_gb_m1",
        "usage_decay_ratio",
        "drop_call_rate",
        "complaint_count_90d",
        "cltv_idr",
        "treatment_arm",
        "retained_flag",
    ]
    null_expr = " + ".join([f"CASE WHEN {c} IS NULL THEN 1 ELSE 0 END" for c in core_cols])
    null_count = conn.execute(f"SELECT SUM({null_expr}) FROM read_parquet('{g_str}')").fetchone()[0] or 0
    if null_count > 0:
        raise QualityGateViolation(f"Null values detected in core columns: {null_count}")

    # 4. Assert non-negative values for financial and duration columns
    neg_checks = conn.execute(f"""
        SELECT
            COUNT(CASE WHEN monthly_arpu_idr < 0 THEN 1 END) AS neg_arpu,
            COUNT(CASE WHEN tenure_months < 0 THEN 1 END) AS neg_tenure,
            COUNT(CASE WHEN cltv_idr < 0 THEN 1 END) AS neg_cltv,
            COUNT(CASE WHEN data_usage_gb_m1 < 0 THEN 1 END) AS neg_data,
            COUNT(CASE WHEN drop_call_rate < 0 OR drop_call_rate > 100 THEN 1 END) AS invalid_drop_rate,
            COUNT(CASE WHEN retained_flag NOT IN (0, 1) THEN 1 END) AS invalid_target
        FROM read_parquet('{g_str}')
    """).fetchone()

    neg_arpu, neg_tenure, neg_cltv, neg_data, invalid_drop, invalid_target = neg_checks
    if neg_arpu > 0:
        raise QualityGateViolation(f"Negative monthly_arpu_idr detected: {neg_arpu} rows")
    if neg_tenure > 0:
        raise QualityGateViolation(f"Negative tenure_months detected: {neg_tenure} rows")
    if neg_cltv > 0:
        raise QualityGateViolation(f"Negative cltv_idr detected: {neg_cltv} rows")
    if neg_data > 0:
        raise QualityGateViolation(f"Negative data_usage_gb_m1 detected: {neg_data} rows")
    if invalid_drop > 0:
        raise QualityGateViolation(f"Out-of-bounds drop_call_rate detected: {invalid_drop} rows")
    if invalid_target > 0:
        raise QualityGateViolation(f"Invalid retained_flag values detected: {invalid_target} rows")

    # 5. Assert valid treatment arms
    valid_arms = set(ACTION_NAMES)
    found_arms = set(r[0] for r in conn.execute(f"SELECT DISTINCT treatment_arm FROM read_parquet('{g_str}')").fetchall())
    unknown_arms = found_arms - valid_arms
    if unknown_arms:
        raise QualityGateViolation(f"Unknown treatment arms detected: {unknown_arms}")

    # 6. Assert Zero Target Leakage:
    # Feature columns must not have perfect 1:1 deterministic correlation with target
    leakage_check = conn.execute(f"""
        SELECT
            CORR(retained_flag, monthly_arpu_idr) AS r_arpu,
            CORR(retained_flag, tenure_months) AS r_tenure,
            CORR(retained_flag, usage_decay_ratio) AS r_decay,
            CORR(retained_flag, drop_call_rate) AS r_drop
        FROM read_parquet('{g_str}')
    """).fetchone()

    for val, name in zip(leakage_check, ["monthly_arpu_idr", "tenure_months", "usage_decay_ratio", "drop_call_rate"]):
        if val is not None and abs(val) > 0.95:
            raise QualityGateViolation(f"Suspected target leakage on feature '{name}' with correlation {val:.4f}")

    conn.close()

    result = {
        "status": "PASSED",
        "total_subscribers": total_rows,
        "distinct_subscribers": distinct_subs,
        "null_count": null_count,
        "features_verified": core_cols,
        "treatment_arms": sorted(list(found_arms)),
    }
    logger.info(f"Quality gates passed successfully for {total_rows:,} records.")
    return result


if __name__ == "__main__":
    validate_gold_features()

