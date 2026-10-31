# System Architecture & Technical Design Specification: GARUDA-NBA
**Document Type:** Principal Engineering & Mathematical Design Specification (`DESIGN.md`)  
**Target Environment:** Python 3.11+ · Linux/macOS/Windows WSL2 · 8 GB RAM Strict Memory Ceiling  
**Architecture Style:** Modular Monolith with Clean Hexagonal Boundaries (DE Lakehouse $\to$ Causal/Game-Theoretic ML $\to$ Agentic Consumer $\to$ MLOps Monitor)

## 1. Arsitektur Sistem End-to-End & Anggaran Memori 8 GB RAM

### 1.1 Mengapa Arsitektur Ini Mengalahkan Stack Big Data Tradisional di Laptop 8 GB
Kandidat umumnya gagal mengeksekusi proyek *Data Engineering + AI* di laptop 8 GB karena meniru tumpukan berbasis JVM (Apache Spark + Kafka + Airflow + PostgreSQL di Docker) yang langsung menghabiskan 5,5 GB RAM hanya untuk *idle daemon*. Dari **First Principles**, throughput pemrosesan data diukur dari **efisiensi vektorisasi CPU (SIMD) dan kompresi kolumnar (Apache Parquet)**, bukan dari banyaknya container yang berjalan.

| Komponen Arsitektur | Pilihan Teknologi `GARUDA-NBA` | Alokasi RAM Maksimum | Alasan Teknis (First-Principles Justification) |
| :--- | :--- | :--- | :--- |
| **OLAP Storage & ETL Engine** | **DuckDB** + **Apache Parquet (Zstd)** + **dbt-duckdb** | **600 MB** (`SET memory_limit='600MB'`) | Eksekusi *vectorized columnar* langsung di atas file Parquet (*out-of-core execution* otomatis ke disk jika data melebihi RAM). |
| **Feature & Tabular Processing** | **Polars** / **Pandas (PyArrow / float32)** | **250 MB** | *Zero-copy memory sharing* melalui Apache Arrow; *downcasting* `float64` $\to$ `float32` memangkas konsumsi memori hingga 50%. |
| **Predictive & Causal Engine** | **LightGBM** + **SHAP (TreeExplainer)** | **300 MB** | *Histogram-based gradient boosting* mengubah fitur kontinu menjadi *uint8 bins* (256 bin), menghemat memori $8\times$ dibanding XGBoost *exact greedy*. |
| **Game-Theoretic Optimizer** | **SciPy HiGHS (`scipy.optimize.milp`)** | **150 MB** | Solver C++ *Mixed-Integer Linear Programming* tercepat di dunia *open-source*, menyelesaikan 50.000 variabel keputusan dalam < 0.8 detik. |
| **Serving, MLOps & Agentic API** | **FastAPI** + **Pydantic v2** + **MLflow (SQLite)** | **200 MB** | Tanpa *background server* berat; MLflow menggunakan backend `sqlite:///mlflow.db` lokal dan agen LLM dipanggil melalui HTTP REST ringan. |
| **Total Peak Memory** | **Seluruh Pipeline Berjalan Sekuensial/Modular** | **~1.50 GB RSS** | Menyisakan **> 6.5 GB RAM** untuk sistem operasi, browser, dan agen **Antigravity**. |

### 1.2 Alur Data & Keputusan (Data & Decision Flow)
```
[Raw Telco Usage, Billing & RCT Campaign Logs (CSV/Parquet)]
       │
       ▼  (Layer 1: Data Engineering - DuckDB + dbt Medallion)
┌─────────────────────────────────────────────────────────────────────────┐
│ BRONZE: raw_subscribers, raw_network_telemetry, raw_campaign_history    │
│ SILVER: stg_validated_subscribers (Schema & Null Gate, Outlier Capping) │
│ GOLD:   fct_subscriber_features (Point-in-Time Usage Decay, ARPU, CLTV) │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼  (Layer 2: Predictive, Causal Uplift & Shapley Engine)
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Calibrated Baseline Risk Model: P(Churn=1 | X, T=a0)                 │
│ 2. Multi-Treatment CATE Uplift Models (T-Learner / X-Learner):          │
│    τ_k(x_i) = P(Retain=1 | X=x_i, T=a_k) - P(Retain=1 | X=x_i, T=a_0)   │
│ 3. Conformal / Bootstrap Uncertainty: [τ_k,lower(x_i), τ_k,upper(x_i)]  │
│ 4. Exact TreeSHAP Attribution: Top-3 Key Drivers φ_j(x_i)               │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼  (Layer 3: Stackelberg Game-Theoretic Prescriptive Optimizer)
┌─────────────────────────────────────────────────────────────────────────┐
│ • Eligibility Matrix Filter (Technical & Commercial Guardrails)         │
│ • Risk-Adjusted Payoff: π_{i,k} = τ_{k,lower}(x_i) * CLTV_i - Cost(a_k) │
│ • Global Budget Knapsack MILP Solver (HiGHS) -> Optimal Action a*_i     │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼  (Layer 4: Model-to-Agent Consumption & Drift Governance)
┌─────────────────────────────────────────────────────────────────────────┐
│ • Strict Pydantic `DecisionPacket` JSON Contract                        │
│ • Retention Copilot Agent (Groq/Gemini + Deterministic Guardrail)       │
│ • PSI & KS-Stat Drift Monitor -> Auto-Retraining Trigger (`MLflow`)     │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Formulasi Matematis: Causal Inference & Teori Permainan (Game Theory)

### 2.1 Kerangka Potensial Hasil Rubin (Potential Outcomes Framework)
Untuk setiap pelanggan $i \in \{1, \dots, N\}$ dengan vektor fitur pra-intervensi $X_i \in \mathbb{R}^d$, misalkan $\mathcal{A} = \{a_0, a_1, a_2, a_3\}$ adalah himpunan diskrit aksi retensi di mana $a_0$ adalah kontrol (*tanpa intervensi*). Misalkan $Y_i(a_k) \in \{0, 1\}$ menyatakan status retensi pelanggan $i$ (1 = Bertahan / *Retained*, 0 = Berhenti / *Churned*) apabila diberikan aksi $a_k$.

Efek Perlakuan Rata-Rata Kondisional (*Conditional Average Treatment Effect* / **CATE**) dari aksi $a_k$ terhadap kontrol $a_0$ adalah:
$$\tau_k(x_i) = \mathbb{E}[Y_i(a_k) - Y_i(a_0) \mid X_i = x_i] = \mu_k(x_i) - \mu_0(x_i)$$
di mana $\mu_k(x_i) = P(Y_i = 1 \mid X_i = x_i, T_i = a_k)$.

Untuk mengestimasi ketidaktahuan epistemik (*uncertainty*) pada ukuran sampel terbatas, kita melatih ansambel *bootstrap* sebanyak $M$ model per lengan perlakuan dan mengambil **Lower Confidence Bound (LCB)** pada tingkat signifikansi $\alpha = 0.10$:
$$\hat{\tau}_{k,\text{LCB}}(x_i) = \bar{\tau}_k(x_i) - z_{1-\alpha} \cdot \text{SE}(\hat{\tau}_k(x_i))$$

### 2.2 Permainan Stackelberg (Leader-Follower Retention Game)
Interaksi antara operator telekomunikasi (Pemimpin / *Leader* $\mathcal{L}$) dan pelanggan (Pengikut / *Follower* $\mathcal{F}_i$) dimodelkan sebagai **Permainan Sekuensial Stackelberg**:

1. **Ruang Strategi Pemimpin ($\mathcal{L}$):** Memilih vektor alokasi biner $z_{i,k} \in \{0, 1\}$ untuk setiap pelanggan $i$ dan aksi $a_k \in \mathcal{A}$ dengan biaya intervensi langsung $c(a_k)$ di bawah kendala anggaran total $B$.
2. **Fungsi Utilitas & Respons Terbaik Pengikut ($\mathcal{F}_i$):** Pelanggan $i$ mengamati aksi $a_k$ dan memilih bertahan ($r_i = 1$) jika utilitas bersihnya melebihi utilitas pindah ke kompetitor:
   $$U_i(a_k, x_i) = u_{\text{base}}(x_i) + \Delta u_i(a_k \mid x_i) - p_i + c_{\text{switch}}(x_i) > u_{\text{competitor}}(x_i)$$
   Secara probabilistik, probabilitas respons terbaik $\mathbb{P}(r_i^*(a_k) = 1 \mid x_i)$ ditangkap oleh estimator kausal $\hat{\mu}_k(x_i)$.
3. **Matriks Empat Kuadran Perilaku Strategis:**
   - **Persuadables ($\hat{\tau}_{k,\text{LCB}}(x_i) > \theta_{\text{min}}$ dan $\pi_{i,k} > 0$):** Pelanggan yang mengubah keputusannya dari *Churn* menjadi *Retain* karena insentif $a_k$.
   - **Sure Things ($\hat{\mu}_0(x_i) \ge 0.85$ dan $\hat{\tau}_k(x_i) < \theta_{\text{min}}$):** Pelanggan loyal yang tetap bertahan tanpa insentif; strategi ekuilibrium pemimpin adalah $a_0$ (hemat anggaran).
   - **Lost Causes ($\hat{\mu}_k(x_i) < 0.25 \; \forall k$):** Pelanggan yang tetap berhenti meskipun diberi insentif; strategi ekuilibrium pemimpin adalah $a_0$.
   - **Sleeping Dogs ($\hat{\tau}_k(x_i) < 0$):** Intervensi memicu reaksi negatif (mengingatkan pelanggan untuk membatalkan langganan); intervensi dilarang keras ($E_{i,k} = 0$).
4. **Optimasi Ekuilibrium Subgame-Perfect (Global MILP Budget Allocation):**
   Pemimpin menyelesaikan masalah *Multi-Choice Knapsack* atas seluruh populasi $N$:
   $$\max_{z_{i,k} \in \{0,1\}} \sum_{i=1}^{N} \sum_{k=0}^{K} z_{i,k} \cdot \pi_{i,k} \quad \text{dengan} \quad \pi_{i,k} = \hat{\tau}_{k,\text{LCB}}(x_i) \cdot \text{CLTV}_i - \hat{\mu}_k(x_i) \cdot c(a_k)$$
   Dengan kendala:
   $$\sum_{k=0}^{K} z_{i,k} = 1 \quad \forall i \in \{1,\dots,N\}, \qquad \sum_{i=1}^{N} \sum_{k=1}^{K} z_{i,k} \cdot c(a_k) \le B, \qquad z_{i,k} \le E_{i,k}$$
   di mana $E_{i,k} \in \{0, 1\}$ adalah matriks kelayakan (*Eligibility Matrix*).

### 2.3 Permainan Kooperatif: Nilai Shapley (TreeSHAP) untuk Key Drivers
Untuk memenuhi syarat JD IOH (*"Develop model outputs beyond a single score, including key drivers"*), kontribusi setiap fitur $j \in F$ terhadap prediksi individu $f(x_i)$ dihitung menggunakan aksioma **Nilai Shapley** dari teori permainan kooperatif:
$$\phi_j(f, x_i) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_{S \cup \{j\}}(x_{i, S \cup \{j\}}) - f_S(x_{i, S}) \right]$$
Menggunakan algoritma **TreeSHAP** pada LightGBM, kompleksitas komputasi turun dari eksponensial $O(2^{|F|})$ menjadi polinomial $O(T L D^2)$, menghasilkan 3 pendorong utama (*Top-3 Key Drivers*) secara instan untuk setiap pelanggan.

## 3. Desain Skema Data Lakehouse (DuckDB + dbt)

### 3.1 Kontrak Tabel Gold (`gold.fct_subscriber_features`)
| Nama Kolom | Tipe Data DuckDB | Deskripsi & Rumus Rekayasa Fitur |
| :--- | :--- | :--- |
| `subscriber_id` | `VARCHAR PRIMARY KEY` | Identitas unik pelanggan (dienkripsi / *anonymized*). |
| `snapshot_date` | `DATE` | Tanggal *point-in-time* observasi fitur (mencegah *data leakage*). |
| `tenure_months` | `INTEGER` | Lama berlangganan dalam bulan. |
| `monthly_arpu_idr` | `FLOAT` | *Average Revenue Per User* bulanan dalam Rupiah. |
| `data_usage_gb_m1` | `FLOAT` | Penggunaan kuota data bulan berjalan (GB). |
| `usage_decay_ratio` | `FLOAT` | Rasio perubahan pemakaian data $\frac{\text{Usage}_{t}}{\text{Usage}_{t-1} + 0.1}$ (indikator *Action Phase* sebelum *churn*). |
| `drop_call_rate` | `FLOAT` | Persentase panggilan/koneksi terputus akibat gangguan BTS lokal. |
| `complaint_count_90d` | `INTEGER` | Jumlah tiket keluhan pelanggan dalam 90 hari terakhir. |
| `cltv_idr` | `FLOAT` | *Customer Lifetime Value* terproyeksi dalam Rupiah ($\text{ARPU} \times \text{Expected Margin} \times \text{Horizon}$). |
| `treatment_arm` | `VARCHAR` | Lengan intervensi historis (`no_action`, `data_booster_15gb`, `combo_discount_20pct`, `qos_priority_care`). |
| `retained_flag` | `TINYINT` | Target biner observasi ($1 = \text{Retained}$, $0 = \text{Churned}$). |

## 4. Kontrak Integrasi Model-ke-Agen (`DecisionPacket` Schema)

Agar output model dapat dikonsumsi secara aman oleh *Agentic Factory* tanpa halusinasi angka, sistem menerapkan kontrak JSON ketat menggunakan **Pydantic v2**:

```python
from typing import List, Literal
from pydantic import BaseModel, Field

class FeatureDriver(BaseModel):
    feature_name: str
    feature_value: float
    shap_contribution: float
    direction: Literal["increases_churn_risk", "reduces_churn_risk"]

class DecisionPacket(BaseModel):
    subscriber_id: str
    baseline_churn_prob: float = Field(..., ge=0.0, le=1.0)
    strategic_quadrant: Literal["Persuadable", "Sure_Thing", "Lost_Cause", "Sleeping_Dog"]
    recommended_action: Literal["no_action", "data_booster_15gb", "combo_discount_20pct", "qos_priority_care"]
    expected_uplift_pct: float
    uplift_confidence_lower_90: float
    uplift_confidence_upper_90: float
    eligibility_passed: bool
    eligibility_reason: str
     intervention_cost_idr: float
    expected_net_value_idr: float
    top_key_drivers: List[FeatureDriver]
```

## 5. Struktur Direktori Repositori Produksi (`garuda-nba`)

Struktur repositori ini dirancang bersih, modular, dan bebas dari *boilerplate* sampah (*anti-AI-slop*):

```text
garuda-nba/
├── README.md                          # Executive Summary, Architecture, & Verified Benchmarks
├── PRD.md                             # Product Requirements Document
├── DESIGN.md                          # Principal Engineering & Game-Theoretic Design Spec
├── Makefile                           # One-command automation (make setup, make pipeline, make test)
├── requirements-8gb.txt               # Pinned lightweight dependencies (DuckDB, LightGBM, SciPy, FastAPI)
├── data/
│   ├── bronze/                        # Raw ingested Parquet files (auto-downloaded via scripts/ingest.py)
│   ├── silver/                        # Validated & cleaned Parquet tables
│   └── gold/                          # Feature store & decision output Parquet tables
├── notebooks/
│   └── 01_garuda_nba_experiment.ipynb # End-to-End First-Principles Experiment & Ablation Study
├── src/
│   ├── __init__.py
│   ├── config.py                      # Memory limits, action costs, & threshold constants
│   ├── data_engineering/
│   │   ├── ingest.py                  # Public Telco dataset loader & DuckDB Bronze writer
│   │   ├── transform_medallion.py     # DuckDB SQL Bronze -> Silver -> Gold pipeline
│   │   └── quality_gates.py           # Deterministic schema, null, & boundary assertions
│   ├── modeling/
│   │   ├── predictive_risk.py         # Calibrated LightGBM churn classifier + TreeSHAP
│   │   ├── causal_uplift.py           # Multi-treatment T-Learner/X-Learner + Bootstrap LCB
│   │   └── evaluation.py              # ROC-AUC, Brier Score, Qini Coefficient, Uplift@K%
│   ├── decisioning/
│   │   ├── eligibility.py             # Business & technical eligibility matrix
│   │   └── stackelberg_optimizer.py   # SciPy HiGHS MILP global budget allocation solver
│   ├── agentic/
│   │   ├── schemas.py                 # Pydantic v2 DecisionPacket & AgentAction contracts
│   │   └── retention_agent.py         # LLM Agent consumer (Groq/Gemini + deterministic guardrail)
│   ├── monitoring/
│   │   └── drift_detector.py          # PSI & KS-test feature/uplift drift + retrain trigger
│   └── api/
│       └── app.py                     # FastAPI REST endpoints (/v1/decide, /v1/optimize-batch, /v1/drift)
└── tests/
    ├── test_quality_gates.py
    ├── test_stackelberg_optimizer.py
    └── test_decision_contract.py
```


## 6. Spesifikasi Terkunci (Locked Decisions: Dataset 1, Groq LPU Agent, & Clean White UI)

Berdasarkan kesepakatan final arsitektur, tiga keputusan berikut dikunci sebagai standar implementasi utama untuk agen **Antigravity**:

### 6.1 Dataset Utama Terkunci: Opsi 1 (Hybrid Hugging Face Telco Parquet + 64k RCT Uplift)
- **Sumber Primer Tabular & CLTV:** [aai510-group1/telco-customer-churn (Hugging Face)](https://huggingface.co/datasets/aai510-group1/telco-customer-churn) ([Parquet Branch](https://huggingface.co/datasets/Yosephcl/telco-customer-churn/tree/refs%2Fconvert%2Fparquet)) — diunduh langsung melalui HTTP Parquet menggunakan DuckDB `read_parquet()` atau `httpx` tanpa memerlukan kredensial login.
- **Kalibrasi Multi-Treatment RCT:** [Kevin Hillstrom 64,000-row MineThatData Uplift Benchmark](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html) untuk memvalidasi estimator *T-Learner / X-Learner* dan kurva Qini pada data eksperimen acak terkontrol (*Randomized Controlled Trial*).

### 6.2 Lapisan Agentic Terkunci: Groq API (`llama-3.3-70b-versatile`) + Deterministic Guardrail Fallback
- Apabila variabel lingkungan `GROQ_API_KEY` atau `GEMINI_API_KEY` tersedia, `src/agentic/retention_agent.py` memanggil endpoint REST ringan (`httpx.Client(timeout=4.0)`) dengan *system prompt* yang mengunci skema JSON output dan melarang perubahan angka pada `DecisionPacket`.
- Apabila *offline* atau tanpa API key, sistem secara transparan beralih ke **Deterministic Policy Engine** yang menghasilkan pesan personalisasi dan tiket teknis secara instan (`0 ms network overhead`, `100% reproducible` di laptop RAM 8 GB).

### 6.3 Sistem Desain Visual Terkunci: Dominant-White Swiss Editorial (`Anti-AI-Slop`)
Antarmuka web *Executive Decision Simulator* (`garuda_nba_simulator.html` / FastAPI + HTMX) wajib mematuhi aturan desain berikut:
- **Palet Warna Dominan Putih & Bersih:** Latar utama `#FFFFFF` (Pure White), permukaan kontrol sekunder `#F8FAFC` (Slate-50), garis batas tipis `1px solid #E2E8F0` (Hairline Slate-200), dan teks utama `#0F172A` (Slate-900).
- **Larangan Mutlak (Zero AI-Slop Policy):** Dilarang menggunakan latar gelap neon, gradasi ungu/magenta/cyan (`linear-gradient` mencolok), efek *glow/blur* berlebihan, ikon robot/otak generik, atau kalimat basa-basi pemasaran.
- **Tipografi Presisi:** Gunakan `Inter` untuk hierarki struktural dan `JetBrains Mono` (`font-variant-numeric: tabular-nums`) untuk seluruh angka Rupiah (IDR), persentase *CATE Uplift*, *Brier Score*, nilai *TreeSHAP* ($\phi_j$), dan skor *PSI Drift*.
