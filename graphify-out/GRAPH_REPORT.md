# Graph Report - garuda-nba  (2026-10-31)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 143 nodes · 219 edges · 14 communities (9 shown, 4 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12

## God Nodes (most connected - your core abstractions)
1. `BaselineRiskModel` - 11 edges
2. `MultiTreatmentTLearner` - 11 edges
3. `StackelbergBudgetOptimizer` - 9 edges
4. `validate_gold_features()` - 9 edges
5. `test_eligibility_and_stackelberg_optimizer()` - 8 edges
6. `QualityGateViolation` - 7 edges
7. `transform_bronze_to_silver()` - 7 edges
8. `transform_silver_to_gold()` - 7 edges
9. `test_ingestion_and_medallion_pipeline()` - 7 edges
10. `ingest_bronze()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `test_quality_gate_detects_duplicates()` --uses--> `QualityGateViolation`  [INFERRED]
  tests/test_quality_gates.py → src/data_engineering/quality_gates.py
- `test_quality_gate_detects_negative_arpu()` --uses--> `QualityGateViolation`  [INFERRED]
  tests/test_quality_gates.py → src/data_engineering/quality_gates.py
- `test_baseline_risk_and_treeshap()` --calls--> `BaselineRiskModel`  [EXTRACTED]
  tests/test_stackelberg_optimizer.py → src/modeling/predictive_risk.py
- `test_eligibility_and_stackelberg_optimizer()` --calls--> `BaselineRiskModel`  [EXTRACTED]
  tests/test_stackelberg_optimizer.py → src/modeling/predictive_risk.py
- `test_eligibility_and_stackelberg_optimizer()` --calls--> `StackelbergBudgetOptimizer`  [EXTRACTED]
  tests/test_stackelberg_optimizer.py → src/decisioning/stackelberg_optimizer.py

## Import Cycles
- None detected.

## Communities (14 total, 4 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.11
Nodes (27): DuckDBPyConnection, Exception, any, Path, QualityGateViolation, Data Quality Gates and Boundary Contract Assertions for GARUDA-NBA. Validates:…, Raised when a data contract quality gate fails., Execute deterministic quality gates against the Gold feature table. Returns… (+19 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (17): GARUDA-NBA System Configuration & Resource Ceiling Parameters. Strictly aligned…, evaluate_eligibility_matrix(), DataFrame, ndarray, Business, Technical & Behavioral Eligibility Matrix for Retention…, Evaluate eligibility mask (N x K) and audit reasons for all subscribers.…, Stackelberg Game-Theoretic Budget Allocation & Prescriptive Decisioning. Solves…, Multi-Treatment Causal Uplift Engine (T-Learner) with Conformal/Bootstrap LCB.… (+9 more)

### Community 2 - "Community 2"
Cohesion: 0.21
Nodes (10): BaselineRiskModel, Any, DataFrame, ndarray, Predict calibrated probability of churn P(Churn=1 | X)., Compute ROC-AUC and Brier score against ground truth., Compute individual TreeSHAP attributions and extract Top-K drivers., Calibrated LightGBM baseline churn classifier with TreeSHAP explanations. (+2 more)

### Community 3 - "Community 3"
Cohesion: 0.20
Nodes (9): Any, DataFrame, ndarray, Internal HiGHS solver using vectorized sparse matrix formulation., Fast Dual Lagrangian Benefit-to-Cost Efficiency Ranking., Solves Global Campaign Knapsack MILP via SciPy HiGHS., Compute payoff matrix pi_{i,k} = tau_{k,LCB}(x_i) * CLTV_i - Cost(a_k). For k=0…, Solve Multi-Choice Knapsack MILP via SciPy HiGHS. (+1 more)

### Community 4 - "Community 4"
Cohesion: 0.26
Nodes (8): MultiTreatmentTLearner, DataFrame, ndarray, Compute Point CATE tau_k(x_i), Standard Error, and Lower Confidence Bound.…, Classify each subscriber into one of four game-theoretic quadrants. - Sleeping…, T-Learner multi-treatment causal uplift estimator., Train separate calibrated response estimators for each treatment arm., Predict retention probability mu_k(x_i) = P(Retain=1 | X, T=a_k) for all arms.

### Community 5 - "Community 5"
Cohesion: 0.19
Nodes (12): fixture, download_hf_dataset(), generate_calibrated_rct_telemetry(), ingest_bronze(), DataFrame, Path, Data Ingestion Module for GARUDA-NBA. Fetches the public Telco Customer Churn…, Attempt downloading raw Parquet dataset from Hugging Face. (+4 more)

### Community 6 - "Community 6"
Cohesion: 0.25
Nodes (8): compute_qini_curve(), evaluate_policy_comparison(), Any, DataFrame, ndarray, Comprehensive Evaluation Module: Qini Curve, AUUC, Uplift@K% & Policy…, Compute Qini Curve and Normalized Qini Coefficient. Qini(t) = Y_treated(t) -…, Run an economic ablation study comparing GARUDA-NBA vs Naive Churn Policy. -…

### Community 7 - "Community 7"
Cohesion: 0.36
Nodes (6): build_design_matrix(), compute_auc(), compute_psi(), fit_logistic_l2(), DataFrame, ndarray

### Community 8 - "Community 8"
Cohesion: 0.43
Nodes (6): build_design_matrix(), compute_auc(), compute_psi(), fit_logistic_l2(), DataFrame, ndarray

## Knowledge Gaps
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BaselineRiskModel` connect `Community 2` to `Community 1`?**
  _High betweenness centrality (0.158) - this node is a cross-community bridge._
- **Why does `StackelbergBudgetOptimizer` connect `Community 3` to `Community 1`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `MultiTreatmentTLearner` connect `Community 4` to `Community 1`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.11264367816091954 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.1225296442687747 - nodes in this community are weakly interconnected._