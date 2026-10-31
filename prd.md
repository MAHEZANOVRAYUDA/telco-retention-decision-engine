# Product Requirements Document (PRD): GARUDA-NBA
**Project Codename:** `garuda-nba` (Game-Theoretic Autonomous Retention, Uplift & Decisioning Architecture — Next-Best-Action Engine)  
**Author:** Maheza Novrayuda ([GitHub](https://github.com/MAHEZANOVRAYUDA) | [Portfolio](https://portofolio-maheza-novrayuda.vercel.app) | [LinkedIn](https://www.linkedin.com/in/mahezanovrayuda/))  
**Target Roles Covered:** AI/ML Engineer · Data Scientist · Data Engineer  
**Target Industry Benchmark:** Telecommunications & Digital Ecosystem (Indosat Ooredoo Hutchison / IOH, Telkomsel, XLSmart, GoTo Financial)  
**Hardware Constraint Profile:** Strict 8 GB RAM Local-First Execution (< 1.8 GB Peak RSS Memory) · 100% Free & Open-Source Stack

## 1. Executive Summary & First-Principles Problem Statement

### 1.1 Why Traditional Churn Models Fail in Production (First-Principles Analysis)
Sebagian besar proyek *Data Science* tingkat universitas berhenti pada pelatihan model klasifikasi biner $P(\text{Churn}=1 \mid X)$ dan mengasumsikan bahwa setiap pelanggan dengan skor risiko tinggi harus diberikan promo retensi. Dari sudut pandang **First Principles** dan **Unit Economics** operator telekomunikasi (seperti Indosat Ooredoo Hutchison / IOH), pendekatan skor tunggal (*single-score prediction*) memiliki tiga cacat fatal:

1. **Prediksi Risiko $\neq$ Sensitivitas Intervensi (Causal Uplift):** Pelanggan dengan risiko *churn* tinggi sering kali merupakan *Lost Causes* (sudah pindah ke kompetitor dan tidak akan kembali meski diberi diskon) atau *Sleeping Dogs* (pelanggan pascabayar/prabayar yang justru terpicu berhenti ketika dikirimi notifikasi kontrak). Sebaliknya, memberikan diskon 20% kepada *Sure Things* (pelanggan bernilai tinggi yang sebenarnya tetap bertahan tanpa promo) secara langsung menggerus margin pendapatan (*ARPU dilution*).
2. **Ketiadaan Alokasi Budget Terbatas (Zero-Sum Budget Constraint):** Tim *Campaign & Growth* memiliki anggaran intervensi bulanan terbatas ($B$). Memilih aksi optimal untuk ratusan ribu pelanggan bukan sekadar masalah klasifikasi, melainkan permainan strategis (**Stackelberg Leader-Follower Game**) dan optimasi kombinatorial (*Multi-Choice Knapsack Problem*).
3. **Output Model Tidak Siap Dikonsumsi Sistem Hilir (*Downstream Agentic Workflows*):** Skor mentah `0.83` tidak memberi tahu sistem otomatis *mengapa* pelanggan berisiko (*key drivers*), *seberapa yakin* model terhadap estimasi tersebut (*calibrated confidence interval*), *apakah* pelanggan memenuhi syarat teknis/bisnis (*eligibility*), *berapa* nilai ekonomi bersih yang diharapkan (*Expected Monetary Value*), dan *aksi spesifik apa* yang harus dieksekusi (*Recommended Next-Best-Action*).

### 1.2 Solusi yang Dibangun: `GARUDA-NBA`
`GARUDA-NBA` adalah platform keputusan otonom *end-to-end* yang mengintegrasikan:
- **Modern Data Engineering Lakehouse (DuckDB + Polars + dbt):** Mengolah data telemetri penggunaan jaringan, *billing*, dan keluhan pelanggan menggunakan arsitektur Medallion (*Bronze $\to$ Silver $\to$ Gold*) di atas Apache Parquet dengan pemakaian RAM di bawah 500 MB.
- **Predictive, Causal Uplift & Conformal Uncertainty Engine:** Mengestimasi risiko dasar (*Baseline Churn Risk*), dampak kausal intervensi per aksi (*Conditional Average Treatment Effect* / CATE), batas keyakinan terkalibrasi (*Conformal Prediction / Bootstrap Lower Confidence Bound*), serta atribusi fitur **TreeSHAP** (*Cooperative Game Theory*).
- **Stackelberg Game-Theoretic Budget Optimizer:** Menyelesaikan *Mixed-Integer Linear Programming* (`scipy.optimize.milp`) untuk mengalokasikan kuota data, diskon paket, atau prioritas *Quality of Service* (QoS) jaringan yang memaksimalkan *Net Incremental Retained Value* (IDR) di bawah batasan budget kampanye.
- **Model-to-Agent Consumption & Observability Layer:** Menyajikan output model terstruktur (`DecisionPacket` Pydantic v2) yang dikonsumsi secara aman oleh *Agentic Retention Workflow* (Groq LPU / Gemini Flash API dengan *deterministic fallback*) serta dilengkapi deteksi pergeseran distribusi data (*Population Stability Index* / PSI & Kolmogorov-Smirnov Drift Monitor).

## 2. Pemetaan Nilai Jual untuk 3 Role Target (Multi-Role Portfolio Leverage)

Proyek ini dirancang secara strategis agar satu repositori *flagship* dapat memvalidasi kompetensi inti untuk tiga posisi sekaligus:

| Posisi yang Dilamar | Komponen `GARUDA-NBA` yang Ditonjolkan di CV & Interview | Bukti Kompetensi Setara Industri |
| :--- | :--- | :--- |
| **Data Scientist** | • Causal Uplift Modeling (T-Learner / X-Learner) & evaluasi Qini / AUUC<br>• Formulasi *Stackelberg Game* & optimasi *Expected Value* (IDR)<br>• Atribusi Shapley (*TreeSHAP*) & *Uncertainty Quantification* | Membuktikan kemampuan mengubah masalah bisnis menjadi hipotesis kausal, *experimentation*, dan model preskriptif yang terhubung langsung ke ekonomi intervensi (*intervention economics*). |
| **AI / ML Engineer** | • Inferensi *low-latency* (< 25 ms p95) via FastAPI & ONNX/LightGBM<br>• Kontrak *Model-to-Agent* (`DecisionPacket` JSON Schema) + *Guardrails*<br>• MLflow Registry, *Drift Detection* (PSI/KS), & *Auto-Retraining Trigger* | Membuktikan kesiapan mengindustrialisasi model ke produksi, menjaga reliabilitas sistem, dan mengintegrasikan model prediktif ke *Agentic AI Factory*. |
| **Data Engineer** | • Pipeline Medallion (*Bronze-Silver-Gold*) berbasis **DuckDB** & **dbt-duckdb**<br>• *Point-in-Time Correct Feature Engineering* (mencegah *data leakage*)<br>• Validasi kualitas data otomatis (*Schema & Distribution Contracts*) | Membuktikan penguasaan SQL analitik skala besar, pemodelan *Data Warehouse / Lakehouse* modern, dan efisiensi memori ekstrem pada format kolumnar Parquet. |

## 3. Kurasi Dataset Publik & Referensi Arsitektur (100% Gratis & Terverifikasi)

Agar proyek ini memiliki kredibilitas tinggi dan dapat langsung diunduh serta dieksekusi oleh agen Antigravity tanpa hambatan *paywall*, berikut adalah daftar dataset publik dan repositori referensi standar industri:

### 3.1 Opsi Dataset Publik (Pilih Salah Satu atau Mode Hybrid)

| Kode Opsi | Nama Dataset & Sumber Resmi | Skala & Fitur Utama | Kelebihan untuk `GARUDA-NBA` |
| :--- | :--- | :--- | :--- |
| **Dataset 1 (Rekomendasi Utama - Hybrid)** | **[Telco Customer Churn Extended (Hugging Face Parquet/CSV)](https://huggingface.co/datasets/aai510-group1/telco-customer-churn)** + **[Kevin Hillstrom 64k Uplift Benchmark](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html)** ([Kaggle Mirror](https://www.kaggle.com/davinwijaya/customer-retention/code)) | • 7.043 pelanggan Telco lengkap dengan *CLTV*, *Churn Reason*, kontrak, dan layanan<br>• 64.000 baris eksperimen *multi-treatment* RCT untuk kalibrasi *uplift* | Dapat diakses langsung via `datasets` / HTTP Parquet tanpa login, sangat cepat di RAM 8 GB, dan memungkinkan penggabungan fitur domain Telco dengan distribusi *multi-treatment uplift* riil. |
| **Dataset 2 (Skala Ratusan Ribu - SEA/Prepaid)** | **[SEA/Indian Prepaid Telecom Churn (Kaggle)](https://www.kaggle.com/datasets/shivam131019/telecom-churn-dataset)** atau **[100K Synthetic Telco Churn (Kaggle)](https://www.kaggle.com/datasets/dhrubangtalukdar/telco-customer-churn-data)** | • ~100.000 baris data historis penggunaan (bulan ke-6, 7, 8, dan *churn* bulan ke-9)<br>• Mencerminkan pasar prabayar (*prepaid*) Asia Tenggara yang identik dengan IOH | Sangat cocok untuk memamerkan *windowing SQL* di DuckDB/dbt (perubahan penggunaan *Good Phase* vs *Action Phase* sebelum *churn*). |
| **Dataset 3 (Benchmark Akademik & Industri)** | **[Cell2Cell Duke Teradata Telecom Churn (71k rows)](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom)** atau **[Orange Belgium Churn Uplift Data (Mendeley)](https://data.mendeley.com/datasets/vrnzjwbnkd)** | • 71.047 baris × 58 atribut perilaku panggilan, perangkat, dan kualitas jaringan | Memiliki fitur teknis jaringan (*dropped calls, blocked calls, equipment age*) yang kaya untuk *SHAP driver analysis*. |

### 3.2 Referensi Arsitektur & Pustaka Standar Industri
1. **Causal Inference & Uplift Modeling:** [uber/causalml](https://github.com/uber/causalml) (dokumentasi meta-learner CATE) dan [maks-sh/scikit-uplift](https://github.com/maks-sh/scikit-uplift) (standar metrik *Qini Curve* dan *Area Under Uplift Curve* / AUUC).
2. **Modern In-Process Data Lakehouse:** [duckdb/dbt-duckdb](https://github.com/duckdb/dbt-duckdb) (*Modern Data Stack in a Box* di atas Parquet tanpa overhead JVM).
3. **ML & LLM Observability:** [evidentlyai/evidently](https://github.com/evidentlyai/evidently) (metodologi *Data Drift*, *Target Drift*, dan *Population Stability Index*).

## 4. Spesifikasi Fungsional (Functional Requirements)

### FR-01: Data Engineering & Point-in-Time Feature Pipeline
- Sistem harus mengonsumsi data mentah pelanggan, penggunaan jaringan (*data/voice usage*), riwayat tagihan (*ARPU*), dan log interaksi *treatment* ke lapisan **Bronze** (Parquet).
- Transformasi **Silver** wajib melakukan pembersihan tipe data, penanganan nilai kosong secara deterministik, dan validasi kontrak skema (nol duplikasi `subscriber_id`, `monthly_arpu_idr >= 0`, `tenure_months >= 0`).
- Lapisan **Gold (`fct_subscriber_features`)** wajib menghitung fitur turunan berbasis rasio dan tren (*usage decay ratio*, *ARPU-to-tenure velocity*, *complaint intensity index*, *network drop rate anomaly*) menggunakan DuckDB SQL tanpa membocorkan informasi masa depan (*zero target leakage*).

### FR-02: Multi-Output Predictive & Causal Uplift Modeling
- Sistem harus melatih model **Baseline Churn Risk** $P(Y_{\text{churn}}=1 \mid X, T=a_0)$ yang dikalibrasi dengan *Isotonic / Platt Scaling* (dievaluasi dengan **Brier Score** dan **ROC-AUC**).
- Sistem harus melatih **Multi-Treatment Causal Uplift Engine** untuk mengestimasi pengurangan risiko *churn* $\hat{\tau}_k(x_i) = \hat{P}(Y=0 \mid X=x_i, T=a_k) - \hat{P}(Y=0 \mid X=x_i, T=a_0)$ pada setiap opsi intervensi $a_k$:
  - $a_0$: *Holdout / No Intervention* (Biaya: Rp 0)
  - $a_1$: *Data Booster 15 GB (7 Hari)* (Biaya marginal: Rp 12.000)
  - $a_2$: *Retention Combo Discount 20% (3 Bulan)* (Biaya subsidi rata-rata: Rp 35.000)
  - $a_3$: *Proactive Network QoS Priority & Tech Care Ticket* (Biaya operasional: Rp 18.000)
- Setiap prediksi individu wajib menyertakan:
  - **Key Drivers:** Top-3 fitur pendorong risiko dan sensitivitas intervensi menggunakan nilai eksak **TreeSHAP** ($\phi_j$).
  - **Confidence & Uncertainty:** Rentang keyakinan 90% ($[\hat{\tau}_{k,\text{lower}}, \hat{\tau}_{k,\text{upper}}]$) melalui *Quantile/Bootstrap Ensemble* untuk menghindari keputusan spekulatif.
  - **Quadrant Classification:** Kategorisasi otomatis pelanggan ke dalam *Persuadables*, *Sure Things*, *Lost Causes*, atau *Sleeping Dogs*.

### FR-03: Game-Theoretic Prescriptive Optimizer (Next-Best-Action)
- Sistem harus mengevaluasi matriks *Eligibility Rules* (misalnya: aksi $a_3$ hanya *eligible* jika `network_drop_rate` tinggi atau ada keluhan teknis; aksi $a_2$ tidak boleh diberikan kepada pelanggan dengan $\hat{\tau}_2(x_i) \le 0$ untuk mencegah efek *Sleeping Dog*).
- Sistem harus menghitung **Risk-Adjusted Expected Value (EV)** dalam Rupiah:
  $$\text{EV}_{i,k} = \left(\hat{\tau}_{k,\text{lower}}(x_i) \times \text{CLTV}_i\right) - c(a_k)$$
- Sistem harus menyelesaikan alokasi anggaran kampanye global ($B$) menggunakan pemrograman linier bilangan bulat (*MILP*) sehingga total nilai retensi bersih seluruh kohort pelanggan maksimum tanpa melampaui anggaran $B$.

### FR-04: Safe Agentic Workflow Consumption (`Model-to-Agent Pattern`)
- Model ML tidak boleh dipanggil secara bebas tanpa skema oleh LLM. Seluruh hasil keputusan harus dikemas ke dalam kontrak ketat `DecisionPacket` (Pydantic v2).
- *Agentic Workflow* bertugas mengonsumsi `DecisionPacket`, memverifikasi *guardrails* kepatuhan (memastikan diskon tidak melebihi batas *eligibility* dan alasan sesuai *SHAP key drivers*), serta menghasilkan pesan komunikasi personalisasi multi-saluran (WhatsApp / Push Notification aplikasi MyIOH) dan tiket intervensi otomatis.

### FR-05: MLOps Drift Monitoring & Retraining Governance
- Sistem wajib menghitung **Population Stability Index (PSI)** pada fitur utama dan distribusi skor *uplift* antara data referensi (*training baseline*) dan batch produksi terbaru.
- Jika $\text{PSI} \ge 0.25$ atau terjadi penurunan *realized conversion rate* di bawah batas kontrol, sistem secara otomatis mengeluarkan sinyal `RETRAIN_REQUIRED` beserta laporan diagnostik fitur yang mengalami pergeseran.

## 5. Non-Functional Requirements (NFR) — Standar Laptop RAM 8 GB

| Parameter Sistem | Target Kuantitatif (SLA) | Strategi Rekayasa First-Principles |
| :--- | :--- | :--- |
| **Batas Memori Puncak (Peak RAM)** | **< 1.8 GB RSS** (Aman di laptop 8 GB RAM bersama OS & IDE) | Gunakan **DuckDB** (`memory_limit='1.2GB'`) + **Polars/PyArrow** + `float32`/`int32` *downcasting* + `gc.collect()`. Dilarang menggunakan JVM (Spark/Kafka) atau LLM lokal berukuran besar. |
| **Waktu Eksekusi Pipeline End-to-End** | **< 45 detik** untuk 50.000–100.000 pelanggan di CPU laptop | Gunakan algoritma berbasis histogram **LightGBM** (`max_bin=255`, `n_jobs=4`) dan solver vektorisasi `scipy.optimize.milp` (HiGHS). |
| **Latensi API Inferensi + NBA** | **< 35 ms (p95)** untuk *scoring* + *SHAP* + *NBA decision* | Pre-komputasi pohon *TreeSHAP* dan pemisahan jalur *real-time scoring* dari *asynchronous LLM message generation*. |
| **Biaya Lisensi & Cloud** | **Rp 0 (100% Gratis)** | Seluruh komponen berjalan di atas pustaka *open-source* lokal; integrasi LLM menggunakan *free tier* Groq API / Google Gemini Flash dengan *deterministic template fallback* jika *offline*. |

## 6. Metrik Keberhasilan (Analytical & Business Success Metrics)

1. **Model Quality Metrics:**
   - *Baseline Churn Classifier:* ROC-AUC $\ge 0.82$, Brier Score $\le 0.14$ (menjamin probabilitas terkalibrasi).
   - *Causal Uplift Model:* **Qini Coefficient** positif signifikan di atas *random targeting* dan **Uplift@30%** minimal $1.8\times$ dibandingkan strategi *Churn-Risk-Only Targeting*.
2. **Business & Economic Value Metrics:**
   - **Incremental Net Retained Value (IDR):** Kenaikan profit retensi bersih minimal **+25% hingga +40%** pada anggaran kampanye yang sama dibandingkan kebijakan tradisional (*menargetkan skor churn tertinggi semata*).
   - **Budget Waste Reduction:** Penurunan alokasi promo sia-sia pada segmen *Sure Things* dan *Sleeping Dogs* hingga **> 70%**.
