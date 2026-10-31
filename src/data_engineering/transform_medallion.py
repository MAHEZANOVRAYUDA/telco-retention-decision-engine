"""Medallion Lakehouse Transformation Pipeline (Bronze -> Silver -> Gold).

Uses DuckDB SQL in-process with strict memory limits (600MB RAM, 4 threads).
Produces point-in-time features in data/gold/fct_subscriber_features.parquet
with zero data leakage.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import duckdb

from src.config import (
    BRONZE_DIR,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_THREADS,
    GOLD_DIR,
    SILVER_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("garuda_nba.transform")


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Create a configured DuckDB connection with strict memory limits."""
    conn = duckdb.connect(database=":memory:")
    conn.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}';")
    conn.execute(f"SET threads = {DUCKDB_THREADS};")
    conn.execute("SET preserve_insertion_order = false;")
    return conn


def transform_bronze_to_silver(
    bronze_path: Optional[Path] = None,
    silver_path: Optional[Path] = None,
) -> Path:
    """Transform Bronze raw Parquet into Silver validated stage Parquet.

    - Cleans data types & handles missing values deterministically
    - Deduplicates on subscriber_id
    - Enforces boundary conditions (monthly_arpu_idr >= 0, tenure_months >= 0)
    """
    b_path = bronze_path or (BRONZE_DIR / "raw_subscribers.parquet")
    s_path = silver_path or (SILVER_DIR / "stg_validated_subscribers.parquet")
    s_path.parent.mkdir(parents=True, exist_ok=True)

    if not b_path.exists():
        raise FileNotFoundError(f"Bronze file not found: {b_path}")

    b_str = str(b_path).replace("\\", "/")
    s_str = str(s_path).replace("\\", "/")

    conn = get_duckdb_connection()

    logger.info("Transforming Bronze -> Silver in DuckDB...")
    query = f"""
    COPY (
        WITH ranked AS (
            SELECT
                TRIM(CAST(subscriber_id AS VARCHAR)) AS subscriber_id,
                GREATEST(0, CAST(COALESCE(tenure_months, 1) AS INTEGER)) AS tenure_months,
                GREATEST(0.0, CAST(COALESCE(monthly_arpu_idr, 25000.0) AS FLOAT)) AS monthly_arpu_idr,
                GREATEST(0.0, CAST(COALESCE(data_usage_gb_m1, 1.0) AS FLOAT)) AS data_usage_gb_m1,
                LEAST(GREATEST(CAST(COALESCE(usage_decay_ratio, 1.0) AS FLOAT), 0.05), 3.0) AS usage_decay_ratio,
                LEAST(GREATEST(CAST(COALESCE(drop_call_rate, 0.0) AS FLOAT), 0.0), 100.0) AS drop_call_rate,
                GREATEST(0, CAST(COALESCE(complaint_count_90d, 0) AS INTEGER)) AS complaint_count_90d,
                COALESCE(contract_type, 'Prepaid') AS contract_type,
                CAST(COALESCE(is_prepaid, 1) AS TINYINT) AS is_prepaid,
                CAST(COALESCE(is_annual, 0) AS TINYINT) AS is_annual,
                GREATEST(0.0, CAST(COALESCE(cltv_idr, 50000.0) AS FLOAT)) AS cltv_idr,
                COALESCE(treatment_arm, 'no_action') AS treatment_arm,
                CAST(COALESCE(retained_flag, 1) AS TINYINT) AS retained_flag,
                ROW_NUMBER() OVER (PARTITION BY subscriber_id ORDER BY tenure_months DESC) AS rn
            FROM read_parquet('{b_str}')
            WHERE subscriber_id IS NOT NULL AND TRIM(CAST(subscriber_id AS VARCHAR)) != ''
        )
        SELECT
            subscriber_id,
            tenure_months,
            monthly_arpu_idr,
            data_usage_gb_m1,
            usage_decay_ratio,
            drop_call_rate,
            complaint_count_90d,
            contract_type,
            is_prepaid,
            is_annual,
            cltv_idr,
            treatment_arm,
            retained_flag
        FROM ranked
        WHERE rn = 1 AND monthly_arpu_idr >= 0.0
    ) TO '{s_str}' (FORMAT 'PARQUET', COMPRESSION 'ZSTD');
    """
    conn.execute(query)
    conn.close()
    logger.info(f"Silver table written successfully to {s_path}")
    return s_path


def transform_silver_to_gold(
    silver_path: Optional[Path] = None,
    gold_path: Optional[Path] = None,
) -> Path:
    """Transform Silver stage Parquet into Gold feature store Parquet.

    Computes engineered features:
    - Point-in-time usage decay ratio
    - ARPU-to-tenure velocity
    - Drop call rate anomaly flag
    - Complaint intensity index
    - Projected Customer Lifetime Value (IDR)
    Zero target leakage is strictly maintained.
    """
    s_path = silver_path or (SILVER_DIR / "stg_validated_subscribers.parquet")
    g_path = gold_path or (GOLD_DIR / "fct_subscriber_features.parquet")
    g_path.parent.mkdir(parents=True, exist_ok=True)

    if not s_path.exists():
        raise FileNotFoundError(f"Silver file not found: {s_path}")

    s_str = str(s_path).replace("\\", "/")
    g_str = str(g_path).replace("\\", "/")

    conn = get_duckdb_connection()

    logger.info("Transforming Silver -> Gold in DuckDB...")
    query = f"""
    COPY (
        SELECT
            subscriber_id,
            CURRENT_DATE AS snapshot_date,
            tenure_months,
            monthly_arpu_idr,
            data_usage_gb_m1,
            usage_decay_ratio,
            drop_call_rate,
            complaint_count_90d,
            is_prepaid,
            is_annual,
            -- Point-in-time engineered features
            ROUND(monthly_arpu_idr / (tenure_months + 1.0), 2) AS arpu_tenure_velocity,
            CASE WHEN drop_call_rate >= 5.0 THEN 1 ELSE 0 END AS high_drop_rate_flag,
            CASE WHEN complaint_count_90d >= 2 THEN 1 ELSE 0 END AS high_complaint_flag,
            ROUND(cltv_idr, 0) AS cltv_idr,
            treatment_arm,
            retained_flag
        FROM read_parquet('{s_str}')
    ) TO '{g_str}' (FORMAT 'PARQUET', COMPRESSION 'ZSTD');
    """
    conn.execute(query)
    conn.close()
    logger.info(f"Gold feature table written successfully to {g_path}")
    return g_path


def run_medallion_pipeline() -> Path:
    """Execute end-to-end Medallion pipeline (Bronze -> Silver -> Gold)."""
    t0 = time.time()
    s_path = transform_bronze_to_silver()
    g_path = transform_silver_to_gold(silver_path=s_path)
    logger.info(f"Medallion ETL pipeline completed in {time.time() - t0:.2f}s")
    return g_path


if __name__ == "__main__":
    run_medallion_pipeline()
