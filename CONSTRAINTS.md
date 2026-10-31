# Constraints (garuda-nba)

Last reviewed: 2026-10-31 by Engineering Architecture Lead

## 1. Floor (Always Enforced, Non-Negotiable)

- **No Suppression Shortcuts:** Dilarang menambahkan `# noqa`, `# type: ignore`, atau kompensasi palsu tanpa dokumentasi justifikasi teknis.
- **No Unimplemented Stubs:** Dilarang menyisakan placeholder kosong `raise NotImplementedError` atau fungsi dummy pada jalur produksi.
- **No Test Deletion/Skipping:** Semua unit dan integration test wajib lulus 100% (`pytest tests/`). Dilarang menghapus atau menandai `@pytest.mark.skip` untuk meloloskan build.
- **Zero Hardcoded Secrets:** Token dan kredensial (seperti `GROQ_API_KEY`) hanya dibaca dari environment variables via `src/config.py`. Tidak ada file `.env` yang di-commit ke source control.
- **Deterministic Offline Resilience:** Semua dependensi eksternal (termasuk LLM API) wajib memiliki deterministic fallback sehingga sistem 100% fungsional saat offline.

## 2. Hard Computational & Memory Constraints (Ramah Laptop 8 GB RAM)

| Dimensi | Batas Maksimum (Ceiling) | Target Operasional | Mekanisme Enforce |
| :--- | :--- | :--- | :--- |
| **Peak RAM RSS** | `< 1,500 MB` | `< 400 MB` | `psutil.Process().memory_info().rss` telemetry assertion |
| **DuckDB Memory** | `600 MB` | `300 MB` | `SET memory_limit='600MB'`, `SET threads=4` di `src/config.py` |
| **Numeric Types** | `float32`, `int32` | `float32` | Explicit downcasting pada DuckDB SQL & Polars dataframe |
| **Backend Compute** | C++/Rust binaries only | SciPy HiGHS / LightGBM | **Dilarang keras:** JVM (Spark, Kafka), distributed DB berat |
| **MILP Knapsack Latency** | `< 3.0 detik` (N=20,000) | `< 1.0 detik` | Pre-filtering candidate pool & sparse matrix COO formulation |

## 3. Data Integrity & Domain Quality Gates

| Dimensi | Aturan | Diperiksa Oleh | Eksekusi |
| :--- | :--- | :--- | :--- |
| **Subscriber Uniqueness** | 0 duplicate `subscriber_id` | `src/data_engineering/quality_gates.py` | Pipeline & Ingest |
| **Revenue Sanity** | 0 nilai negatif pada `monthly_arpu_idr` | `src/data_engineering/quality_gates.py` | Pipeline & Ingest |
| **Feature Distribution** | $0.0 \le \text{usage\_decay\_ratio} \le 1.0$, $0.0 \le \text{drop\_call\_rate} \le 1.0$ | DuckDB `LEAST(GREATEST(...))` clamping | Silver/Gold Transform |
| **Causal Guardrails** | Sleeping Dog ($\hat{\tau}_{k,\text{LCB}} \le 0$) & Ineligible DILARANG menerima promo | `src/agentic/retention_agent.py` | Agent & Decisioning API |
| **Model Drift Gate** | Otomatis trigger `RETRAIN_REQUIRED` jika $\text{PSI} \ge 0.25$ | `src/monitoring/drift_detector.py` | Periodic / POST `/v1/drift` |

## 4. Tone, Narrative & Positioning Standards

- **Executive & Objective Framing:** Repositori dan seluruh dokumentasi teknis (`README.md`, `design.md`, `prd.md`, komentar kode) wajib disusun dengan nada teknik netral, profesional, dan berbasis first-principles.
- **Prohibited Terminology:** Dilarang menggunakan narasi pencarian kerja, tips wawancara (interview), optimasi CV/ATS, atau klaim portofolio mahasiswa/anak-anakan. Repositori ini diposisikan murni sebagai arsitektur riset dan produksi mandiri (*self-hosted production architecture*) untuk memecahkan inefisiensi alokasi anggaran retensi telekomunikasi.

## 5. Visual Hierarchy & Frontend UI/UX Standards

- **Aesthetic Direction:** Swiss Editorial Dominant-White.
- **Color Tokens:**
  - Background: `#FFFFFF` (Main canvas) & `#F8FAFC` (Card surfaces).
  - Borders: 1px hairline solid `#E2E8F0`.
  - Text & Accents: Slate-900 `#0F172A` (Headings), Slate-600 `#475569` (Body), Emerald-600 `#059669` (Success/Value), Crimson-600 `#DC2626` (Risk/Sleeping Dog).
- **Zero AI-Slop Directive:** Dilarang menggunakan gradasi neon ungu-biru, drop shadow hitam tebal (`0 4px 6px #000`), hero section kosong tanpa hirarki, atau tombol tanpa interaksi mikro yang jelas.
- **Typography:** Inter untuk teks antarmuka dan JetBrains Mono untuk tabular numerals / metrik finansial.
- **Accessibility:** Kontras teks minimum 4.5:1 (WCAG 2.1 AA) dengan visual keyboard focus state (`ring-2 ring-slate-900`).

## 6. Verification & Test Suite Matrix

| Komponen | Cakupan Pengujian | Target Lulus |
| :--- | :--- | :--- |
| `tests/test_quality_gates.py` | Lakehouse Ingestion, Medallion Transform, Quality Gate Rejections | 100% Passed |
| `tests/test_stackelberg_optimizer.py` | Calibrated LightGBM, TreeSHAP drivers, T-Learner CATE, SciPy HiGHS MILP Knapsack | 100% Passed |
| `tests/test_decision_contract.py` | Pydantic v2 schemas, Agent hard guardrails, Deterministic message synthesis | 100% Passed |
| `tests/test_drift_detector.py` | Kolmogorov-Smirnov test, PSI calculation, MLflow SQLite telemetry tracking | 100% Passed |
| `tests/test_ui_and_endpoints.py` | Swiss Editorial HTML serving, CSS hairline borders, REST API contract validation | 100% Passed |

