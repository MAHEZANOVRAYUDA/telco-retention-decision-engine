# GARUDA-NBA Task Checklist

- [x] Initial Discovery & Environment Setup
  - [x] Read `PRD.md`, `DESIGN.md`, `garuda_nba_experiment.py`, `garuda_nba_stackelberg_retention_causal_uplift_decision_console.html`, `antigravity_playbook.md`
  - [x] Create `requirements-8gb.txt`
  - [x] Make `garuda_nba_experiment.py` cross-platform (Windows `psutil` / Linux `resource`)
  - [x] Run `garuda_nba_experiment.py` and verify `notebooks/01_garuda_nba_experiment.ipynb` export (Peak RAM: 92.2 MB, Time: 0.08s)
  - [x] Install required dependencies (`duckdb`, `polars`, `lightgbm`, `shap`, `mlflow`)

- [x] Tahap 1: Data Engineering Lakehouse (DuckDB Medallion + Quality Gates)
  - [x] Create `src/config.py` (Memory ceilings, threads, action costs, thresholds)
  - [x] Create `src/data_engineering/ingest.py` (Hugging Face Parquet fetcher + calibrated RCT synthetic fallback to Bronze Parquet Zstd)
  - [x] Create `src/data_engineering/transform_medallion.py` (DuckDB Bronze -> Silver -> Gold `fct_subscriber_features.parquet`)
  - [x] Create `src/data_engineering/quality_gates.py` (Schema validation, null checks, duplicate checks, boundary checks)
  - [x] Create `tests/test_quality_gates.py`
  - [x] Run `pytest tests/test_quality_gates.py` & verify peak RSS < 400 MB (Measured: 66.3 MB, Tests: 3/3 passed)

- [x] Tahap 2: Causal Uplift, TreeSHAP & Stackelberg Game-Theoretic Optimizer
  - [x] Create `src/modeling/predictive_risk.py` (Calibrated LightGBM baseline churn model + TreeSHAP)
  - [x] Create `src/modeling/causal_uplift.py` (T-Learner multi-treatment CATE + 90% LCB + 4-quadrant classification)
  - [x] Create `src/decisioning/eligibility.py` (Guardrail rules, technical & commercial eligibility matrix)
  - [x] Create `src/decisioning/stackelberg_optimizer.py` (SciPy HiGHS MILP Knapsack solver under budget constraint)
  - [x] Create `src/modeling/evaluation.py` (Qini curves, Uplift@30%, Net EV comparison vs Naive policy)
  - [x] Create `tests/test_stackelberg_optimizer.py`
  - [x] Run `pytest tests/test_stackelberg_optimizer.py` & verify peak RSS < 600 MB (Measured: 277.2 MB on full 20k cohort, Tests: 3/3 passed)
  - [x] Run `graphify` extraction & clustering (`graphify-out/graph.json`: 143 nodes, 219 edges, 14 communities)

- [x] Tahap 3: Model-to-Agent Consumption Pattern & FastAPI Service
  - [x] Create `src/agentic/schemas.py` (Pydantic v2 `FeatureDriver`, `DecisionPacket`, `AgentInterventionResponse`)
  - [x] Create `src/agentic/retention_agent.py` (Groq API caller + deterministic structured template fallback + hard guardrails)
  - [x] Create `src/api/app.py` (FastAPI with `POST /v1/decide`, `POST /v1/optimize-cohort`, `GET /v1/drift`, `GET /v1/health`)
  - [x] Create `tests/test_decision_contract.py`
  - [x] Run `pytest tests/test_decision_contract.py` (4/4 passed, total suite 10/10 passed)
  - [x] Update graphify knowledge graph (`graphify-out/graph.json`: 261 nodes, 404 edges, 22 communities)

- [ ] Tahap 4: MLOps Drift Monitoring, MLflow SQLite & Portfolio Polish
  - [ ] Create `src/monitoring/drift_detector.py` (PSI & KS-test drift detector with `RETRAIN_REQUIRED` signal)
  - [ ] Create `src/pipeline.py` & integrate MLflow SQLite tracking (`sqlite:///mlflow.db`)
  - [ ] Create `Makefile` (`setup`, `pipeline`, `test`, `serve`)
  - [ ] Write executive-grade `README.md` with verified benchmarks, architecture diagrams, and economics analysis

- [ ] Tahap 5: Dominant-White Swiss Editorial Console UI Integration
  - [ ] Create `src/api/templates/index.html` from `garuda_nba_stackelberg_retention_causal_uplift_decision_console.html`
  - [ ] Connect interactive controls (Budget Slider, LCB Alpha, TreeSHAP Inspector, PSI Drift Stress-Tester) to FastAPI endpoints
  - [ ] Verify Dominant-White aesthetics (#FFFFFF / #F8FAFC, 1px solid #E2E8F0, JetBrains Mono tabular numerals, zero AI-slop)
  - [ ] End-to-end integration test of all endpoints & UI
