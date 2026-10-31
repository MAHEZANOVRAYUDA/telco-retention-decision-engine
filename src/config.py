"""GARUDA-NBA System Configuration & Resource Ceiling Parameters.

Strictly aligned with 8 GB RAM laptop limits:
- DuckDB memory limit: 600MB, 4 threads
- Peak RSS ceiling: < 1.5 GB (target < 400 MB)
- Prescriptive retention economics & intervention costs
"""

import os
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

for d in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, MODELS_DIR, NOTEBOOKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hardware & Memory Guardrails (Strict 8 GB RAM Laptop Profile)
# ---------------------------------------------------------------------------
DUCKDB_MEMORY_LIMIT = "600MB"
DUCKDB_THREADS = 4
PEAK_RAM_CEILING_MB = 1500.0
TARGET_RAM_MB = 400.0

# ---------------------------------------------------------------------------
# Intervention Space & Unit Economics (Indosat Ooredoo Hutchison / SEA Benchmark)
# ---------------------------------------------------------------------------
ACTION_COSTS: Dict[str, float] = {
    "no_action": 0.0,
    "data_booster_15gb": 12000.0,      # Marginal network data injection cost
    "combo_discount_20pct": 35000.0,   # Average 3-month contract retention subsidy
    "qos_priority_care": 18000.0,      # Field engineer / priority BTS slice routing
}

ACTION_NAMES = list(ACTION_COSTS.keys())

DEFAULT_CAMPAIGN_BUDGET_IDR = 18_000_000.0

# ---------------------------------------------------------------------------
# Strategic Quadrant & Causal Decisioning Thresholds
# ---------------------------------------------------------------------------
MIN_UPLIFT_THRESHOLD = 0.01          # Minimum positive lift required for Persuadables
SURE_THING_RETENTION_THRESHOLD = 0.85 # Loyal baseline retention (intervention wasteful)
LOST_CAUSE_RETENTION_THRESHOLD = 0.25 # Inevitable churners regardless of incentive
SLEEPING_DOG_UPLIFT_THRESHOLD = 0.0  # Strictly prohibited negative uplift boundary
CONFIDENCE_ALPHA_DEFAULT = 0.10      # 90% Lower Confidence Bound (z_0.90 = 1.28)
Z_SCORE_90_PCT = 1.28

# ---------------------------------------------------------------------------
# MLOps Drift Governance Thresholds
# ---------------------------------------------------------------------------
PSI_MODERATE_SHIFT = 0.10
PSI_RETRAIN_THRESHOLD = 0.25          # Population Stability Index triggering retrain

# ---------------------------------------------------------------------------
# Agentic & LLM Strategy
# ---------------------------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Public Dataset URLs (Option 1: Hugging Face Parquet + Mirrors)
# ---------------------------------------------------------------------------
HF_TELCO_PARQUET_URL = (
    "https://huggingface.co/datasets/aai510-group1/telco-customer-churn/resolve/main/data/train-00000-of-00001.parquet"
)
HF_TELCO_PARQUET_FALLBACK_URL = (
    "https://huggingface.co/datasets/Yosephcl/telco-customer-churn/resolve/main/data/train-00000-of-00001.parquet"
)

