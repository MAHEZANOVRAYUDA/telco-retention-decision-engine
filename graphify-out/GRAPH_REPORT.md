# Graph Report - garuda-nba  (2026-10-31)

## Corpus Check
- 35 files · ~31,787 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 318 nodes · 499 edges · 25 communities (16 shown, 8 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 18 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c16d30ee`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- config.py
- test_stackelberg_optimizer.py
- BaselineRiskModel
- StackelbergBudgetOptimizer
- MultiTreatmentTLearner
- pipeline.py
- app.py
- garuda_nba_experiment.py
- garuda_nba_experiment (1).py
- data_engineering/__init__.py
- decisioning/__init__.py
- src/__init__.py
- modeling/__init__.py
- System Architecture & Technical Design Specification: GARUDA-NBA
- Product Requirements Document (PRD): GARUDA-NBA
- 3. Phase-by-Phase Task Breakdown
- 2. Urutan Prompt Bertahap untuk Agen Antigravity (Step-by-Step Prompts)
- ingest_bronze
- agentic/__init__.py
- api/__init__.py
- todo.md
- GARUDA-NBA: Game-Theoretic Autonomous Retention, Uplift & Decisioning Architecture
- monitoring/__init__.py
- test_ui_and_endpoints.py

## God Nodes (most connected - your core abstractions)
1. `MultiTreatmentTLearner` - 17 edges
2. `BaselineRiskModel` - 16 edges
3. `RetentionAgent` - 15 edges
4. `run_full_pipeline()` - 15 edges
5. `DecisionPacket` - 14 edges
6. `StackelbergBudgetOptimizer` - 14 edges
7. `ingest_bronze()` - 11 edges
8. `validate_gold_features()` - 11 edges
9. `evaluate_eligibility_matrix()` - 11 edges
10. `AgentInterventionResponse` - 10 edges

## Surprising Connections (you probably didn't know these)
- `test_ingestion_and_medallion_pipeline()` --calls--> `ingest_bronze()`  [EXTRACTED]
  tests/test_quality_gates.py → src/data_engineering/ingest.py
- `test_quality_gate_detects_duplicates()` --uses--> `QualityGateViolation`  [INFERRED]
  tests/test_quality_gates.py → src/data_engineering/quality_gates.py
- `test_quality_gate_detects_negative_arpu()` --uses--> `QualityGateViolation`  [INFERRED]
  tests/test_quality_gates.py → src/data_engineering/quality_gates.py
- `test_eligibility_and_stackelberg_optimizer()` --calls--> `StackelbergBudgetOptimizer`  [EXTRACTED]
  tests/test_stackelberg_optimizer.py → src/decisioning/stackelberg_optimizer.py
- `test_eligibility_and_stackelberg_optimizer()` --calls--> `MultiTreatmentTLearner`  [EXTRACTED]
  tests/test_stackelberg_optimizer.py → src/modeling/causal_uplift.py

## Import Cycles
- None detected.

## Communities (25 total, 8 thin omitted)

### Community 0 - "config.py"
Cohesion: 0.11
Nodes (28): DuckDBPyConnection, Exception, GARUDA-NBA System Configuration & Resource Ceiling Parameters. Strictly aligned…, any, Path, QualityGateViolation, Data Quality Gates and Boundary Contract Assertions for GARUDA-NBA. Validates:…, Raised when a data contract quality gate fails. (+20 more)

### Community 1 - "test_stackelberg_optimizer.py"
Cohesion: 0.10
Nodes (21): evaluate_eligibility_matrix(), DataFrame, ndarray, Business, Technical & Behavioral Eligibility Matrix for Retention…, Evaluate eligibility mask (N x K) and audit reasons for all subscribers.…, Multi-Treatment Causal Uplift Engine (T-Learner) with Conformal/Bootstrap LCB.…, compute_qini_curve(), evaluate_policy_comparison() (+13 more)

### Community 2 - "BaselineRiskModel"
Cohesion: 0.21
Nodes (10): BaselineRiskModel, Any, DataFrame, ndarray, Predict calibrated probability of churn P(Churn=1 | X)., Compute ROC-AUC and Brier score against ground truth., Compute individual TreeSHAP attributions and extract Top-K drivers., Calibrated LightGBM baseline churn classifier with TreeSHAP explanations. (+2 more)

### Community 3 - "StackelbergBudgetOptimizer"
Cohesion: 0.17
Nodes (10): Any, DataFrame, ndarray, Stackelberg Game-Theoretic Budget Allocation & Prescriptive Decisioning. Solves…, Internal HiGHS solver using vectorized sparse matrix formulation., Fast Dual Lagrangian Benefit-to-Cost Efficiency Ranking., Solves Global Campaign Knapsack MILP via SciPy HiGHS., Compute payoff matrix pi_{i,k} = tau_{k,LCB}(x_i) * CLTV_i - Cost(a_k). For k=0… (+2 more)

### Community 4 - "MultiTreatmentTLearner"
Cohesion: 0.22
Nodes (10): MultiTreatmentTLearner, DataFrame, ndarray, Compute Point CATE tau_k(x_i), Standard Error, and Lower Confidence Bound.…, Classify each subscriber into one of four game-theoretic quadrants. - Sleeping…, T-Learner multi-treatment causal uplift estimator., Train separate calibrated response estimators for each treatment arm., Predict retention probability mu_k(x_i) = P(Retain=1 | X, T=a_k) for all arms. (+2 more)

### Community 5 - "pipeline.py"
Cohesion: 0.10
Nodes (24): calculate_psi(), DriftDetector, Any, DataFrame, ndarray, MLOps Drift Monitoring & Automated Retraining Governance. Implements: 1.…, Simulate a severe network & macroeconomic shock batch for drift testing., Compute Population Stability Index (PSI) between baseline and production… (+16 more)

### Community 6 - "app.py"
Cohesion: 0.06
Nodes (50): BaseModel, FastAPI, get, post, Autonomous Retention Copilot Agent. Consumes validated DecisionPackets,…, Generate high-touch copy via Groq LPU API with strict timeout., Enterprise retention copilot with deterministic guardrails and Groq LLM…, Process DecisionPacket through guardrails and generate intervention payload. (+42 more)

### Community 7 - "garuda_nba_experiment.py"
Cohesion: 0.36
Nodes (6): build_design_matrix(), compute_auc(), compute_psi(), fit_logistic_l2(), DataFrame, ndarray

### Community 8 - "garuda_nba_experiment (1).py"
Cohesion: 0.43
Nodes (6): build_design_matrix(), compute_auc(), compute_psi(), fit_logistic_l2(), DataFrame, ndarray

### Community 14 - "System Architecture & Technical Design Specification: GARUDA-NBA"
Cohesion: 0.12
Nodes (16): 1.1 Mengapa Arsitektur Ini Mengalahkan Stack Big Data Tradisional di Laptop 8 GB, 1.2 Alur Data & Keputusan (Data & Decision Flow), 1. Arsitektur Sistem End-to-End & Anggaran Memori 8 GB RAM, 2.1 Kerangka Potensial Hasil Rubin (Potential Outcomes Framework), 2.2 Permainan Stackelberg (Leader-Follower Retention Game), 2.3 Permainan Kooperatif: Nilai Shapley (TreeSHAP) untuk Key Drivers, 2. Formulasi Matematis: Causal Inference & Teori Permainan (Game Theory), 3.1 Kontrak Tabel Gold (`gold.fct_subscriber_features`) (+8 more)

### Community 15 - "Product Requirements Document (PRD): GARUDA-NBA"
Cohesion: 0.12
Nodes (16): 1.1 Why Traditional Churn Models Fail in Production (First-Principles Analysis), 1.2 Solusi yang Dibangun: `GARUDA-NBA`, 1. Executive Summary & First-Principles Problem Statement, 2. Pemetaan Nilai Jual untuk 3 Role Target (Multi-Role Portfolio Leverage), 3.1 Opsi Dataset Publik (Pilih Salah Satu atau Mode Hybrid), 3.2 Referensi Arsitektur & Pustaka Standar Industri, 3. Kurasi Dataset Publik & Referensi Arsitektur (100% Gratis & Terverifikasi), 4. Spesifikasi Fungsional (Functional Requirements) (+8 more)

### Community 16 - "3. Phase-by-Phase Task Breakdown"
Cohesion: 0.18
Nodes (10): 1. Executive Summary & Architecture Overview, 2. Component Dependency Graph, 3. Phase-by-Phase Task Breakdown, Hard Constraints Profile (100% Strictly Enforced), Implementation Plan: GARUDA-NBA (Game-Theoretic Autonomous Retention, Uplift & Decisioning Architecture), Tahap 1: Data Engineering Lakehouse (DuckDB Medallion + Quality Gates), Tahap 2: Causal Uplift, TreeSHAP & Stackelberg Game-Theoretic Optimizer, Tahap 3: Model-to-Agent Consumption Pattern & FastAPI Service (+2 more)

### Community 17 - "2. Urutan Prompt Bertahap untuk Agen Antigravity (Step-by-Step Prompts)"
Cohesion: 0.20
Nodes (9): 1. Spesifikasi `requirements-8gb.txt` (100% Gratis & Ramah RAM 8 GB), 2. Urutan Prompt Bertahap untuk Agen Antigravity (Step-by-Step Prompts), 3. Cara Menjual Proyek Ini di CV & Interview (3 Sudut Pandang Role), Antigravity Execution Playbook: Building `GARUDA-NBA` on an 8 GB RAM Laptop, Tahap 1: Data Engineering Lakehouse (DuckDB Medallion + Quality Gates), Tahap 2: Causal Uplift, TreeSHAP & Stackelberg Game-Theoretic Optimizer, Tahap 3: Model-to-Agent Consumption Pattern & FastAPI Service, Tahap 4: MLOps Drift Monitoring, MLflow SQLite & Polish Dokumentasi (+1 more)

### Community 18 - "ingest_bronze"
Cohesion: 0.19
Nodes (12): download_hf_dataset(), generate_calibrated_rct_telemetry(), ingest_bronze(), DataFrame, Path, Data Ingestion Module for GARUDA-NBA. Fetches the public Telco Customer Churn…, Attempt downloading raw Parquet dataset from Hugging Face., Ingest Bronze Parquet dataset with Zstandard compression. Tries public HF… (+4 more)

### Community 22 - "GARUDA-NBA: Game-Theoretic Autonomous Retention, Uplift & Decisioning Architecture"
Cohesion: 0.15
Nodes (12): 1. Executive Summary & First-Principles Problem Statement, 2. End-to-End System Architecture, 3. Verified Benchmark: GARUDA-NBA vs. Traditional Churn Policy, 4. Hardware Efficiency & Zero-Cloud Guarantee (8 GB RAM Laptop Profile), 5. Model-to-Agent Integration Contract (`DecisionPacket`), 6. MLOps Observability & Drift Monitoring, 7. Cara Menjual Proyek Ini di CV & Interview (3 Sudut Pandang Role), 8. Panduan Menjalankan Sistem (Quick Start) (+4 more)

### Community 24 - "test_ui_and_endpoints.py"
Cohesion: 0.20
Nodes (9): client(), fixture, Integration & Aesthetic Validation Tests for Dominant-White Swiss Editorial UI.…, GET / must return valid 200 HTML console., Strict inspection of Dominant-White Swiss Editorial aesthetic tokens., Test full pipeline interaction from UI perspectives., test_serve_simulator_html(), test_swiss_editorial_styling_contracts() (+1 more)

## Knowledge Gaps
- **47 isolated node(s):** `1. Executive Summary & First-Principles Problem Statement`, `2. End-to-End System Architecture`, `3. Verified Benchmark: GARUDA-NBA vs. Traditional Churn Policy`, `4. Hardware Efficiency & Zero-Cloud Guarantee (8 GB RAM Laptop Profile)`, `Deterministic Policy Fallback (100% Offline Resilience)` (+42 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 181 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BaselineRiskModel` connect `BaselineRiskModel` to `test_stackelberg_optimizer.py`, `pipeline.py`, `app.py`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `MultiTreatmentTLearner` connect `MultiTreatmentTLearner` to `test_stackelberg_optimizer.py`, `pipeline.py`, `app.py`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Why does `StackelbergBudgetOptimizer` connect `StackelbergBudgetOptimizer` to `test_stackelberg_optimizer.py`, `pipeline.py`, `app.py`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `MultiTreatmentTLearner` (e.g. with `decide_single_subscriber()` and `lifespan()`) actually correct?**
  _`MultiTreatmentTLearner` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `BaselineRiskModel` (e.g. with `decide_single_subscriber()` and `lifespan()`) actually correct?**
  _`BaselineRiskModel` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `RetentionAgent` (e.g. with `AgentInterventionResponse` and `DecisionPacket`) actually correct?**
  _`RetentionAgent` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `DecisionPacket` (e.g. with `RetentionAgent` and `decide_single_subscriber()`) actually correct?**
  _`DecisionPacket` has 2 INFERRED edges - model-reasoned connections that need verification._