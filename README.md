# AuditLoop

**A reconciliation agent that never lets an LLM have the final say on a financial match.**

Multi-source reconciliation agent for Razorpay settlements, bank statements,
and internal ledgers. A deterministic matching engine runs first; an LLM is
only invoked to explain and propose resolutions for unresolved exceptions —
it can never commit a match directly, every proposal is re-verified
deterministically before it counts. A strict 3-way reconciliation invariant 
(Settlement + Bank + Ledger must all be present) is enforced by the deterministic gate. 
Every decision, matched or not, is logged to a tamper-evident cryptographic audit trail. 
Accuracy is measured against a known ground-truth batch, not demoed on cherry-picked examples.

All data (settlements, bank statement, and ledger) used in this repository's default benchmark is fully synthetic for reproducibility and data privacy. No live API keys are required to execute the evaluation suite.

## Project Structure

```
auditloop/
├── api/
│   └── app.py                    # FastAPI REST API (with /audit/verify & sync threadpool execution)
├── data/
│   ├── fetch_settlements.py     # Real Razorpay test-mode API pull
│   ├── generate_data.py         # Synthetic bank + ledger data, linked to live pull
│   ├── ground_truth.json        # Answer key for measuring precision/recall
│   └── sample_batch/            # Committed example batch (runs with zero API keys)
├── engine/
│   ├── matcher.py               # Vectorized exact + fuzzy matching (Stage 1 O(N+M+L), Stage 2 O(NM+NL))
│   └── exceptions.py            # Stage 3 — dispatch unresolved records to the LLM
├── llm/
│   ├── client.py                 # Groq API wrapper with function calling & retry backoff
│   ├── schemas.py                 # Pydantic response models (Chain-of-Thought & strict validation)
│   ├── prompts.py                 # Few-shot scoped prompts with CoT directions
│   └── privacy.py                 # PII redaction layer (scrubs emails, phones, customer names)
├── audit/
│   ├── store.py                   # Cryptographically chained audit log (SHA-256 block hashing)
│   └── models.py
├── metrics/
│   └── evaluate.py                # Precision / recall / match-rate vs ground truth
├── dashboard/
│   └── app.py                     # Streamlit reviewer dashboard with audit chain inspector
├── tests/
│   ├── test_matcher.py
│   ├── test_llm_schema_validation.py
│   ├── test_end_to_end_metrics.py
│   ├── test_audit_integrity.py
│   ├── test_pii_sanitizer.py
│   ├── test_adversarial_security.py   # Prompt injection & extreme numeric fuzzing
│   ├── test_human_in_the_loop.py      # Maker-Checker manual resolution & hash chain
│   ├── test_performance_scaling.py
│   └── test_api.py
├── docs/
│   ├── architecture.png
│   ├── SCRIPT.md                      # 5-minute winning presentation & live demo script
│   └── DESIGN_DECISIONS.md            # Why deterministic-first, thresholds, tradeoffs
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Tech Stack

Python · FastAPI · pandas · RapidFuzz · Groq API (GPT-OSS 120B function calling +
Pydantic schemas) · Razorpay API · SQLite (SHA-256 Chained) · Streamlit · pytest · Docker

## API Endpoints (Prototype)

AuditLoop exposes a REST API for programmatic access and institutional compliance:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with component status |
| `/reconcile` | POST | Run full reconciliation pipeline (threadpool executed) |
| `/metrics` | GET | Get latest metrics report against ground truth |
| `/audit/verify` | GET | Cryptographically verify SHA-256 audit log integrity |
| `/audit/resolve` | POST | Maker-Checker: Sign & append human controller resolution |
| `/audit/history` | GET | Inspect complete chronological lifecycle of a record pair |
| `/audit/recent` | GET | Inspect recent audit log entries |
| `/audit/disagreements` | GET | View all LLM vs deterministic conflict records |
| `/audit/summary` | GET | Get throughput and cryptographic health stats |

### Example API Usage

```bash
# Health check
curl -H "X-API-Key: dev-secret-key" http://localhost:8000/health

# Cryptographically verify the audit chain
curl -H "X-API-Key: dev-secret-key" http://localhost:8000/audit/verify

# Maker-Checker: Human Controller Manual Resolution
curl -X POST http://localhost:8000/audit/resolve \
  -H "X-API-Key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "record_ids": "sett_001-TXN_001",
    "decision": "human_approved_match",
    "reviewer_id": "CONTROLLER_001",
    "notes": "Verified gateway MDR fee discrepancy with signed merchant invoice."
  }'

# Run reconciliation with custom parameters
curl -X POST http://localhost:8000/reconcile \
  -H "X-API-Key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"records": 20, "seed": 42, "messiness": 0.25, "demo_disagreement": true}'

# Get latest metrics
curl -H "X-API-Key: dev-secret-key" http://localhost:8000/metrics

# View recent audit entries
curl -H "X-API-Key: dev-secret-key" http://localhost:8000/audit/recent?limit=10
```

### Running the API Server

```bash
# Via Docker (recommended)
docker-compose --profile api up

# Or as part of all-in-one deployment
docker-compose --profile all up
# API available at http://localhost:8000

# Local development
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

## Quick Start

> **Record Count Note:** Default Quick Start and evaluation commands use **20 synthetic settlement records**. Orphan bank/ledger events are counted separately and do **not** inflate the transaction evaluation population. The bundled CSVs under `data/` are a **reproducible synthetic snapshot** (`source=synthetic`). They are not live Razorpay production data. The ingestion layer can consume Razorpay test-mode settlement data when `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` are configured.

### Option 1: Docker (Recommended - Zero Setup)

```bash
cd auditloop
docker-compose --profile all up
```

This will:
1. Run the reconciliation pipeline with sample data (no API keys needed)
2. Launch the Streamlit dashboard at http://localhost:8501

### Option 2: Local Python Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data (uses sample batch if no API keys)
python data/generate_data.py --records 20 --seed 42

# Run the full pipeline
python run_pipeline.py --demo-disagreement --records 20

# View metrics
cat metrics_report.json

# Launch dashboard
streamlit run dashboard/app.py --server.port 8501
```

### Option 3: With Real Razorpay API Data

```bash
# 1. Get test-mode credentials from https://dashboard.razorpay.com/
#    (Enable Test Mode toggle, go to Settings -> API Keys)

# 2. Create .env file
cp .env.example .env
# Edit .env with your RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET

# 3. Fetch real settlements
python data/fetch_settlements.py --year 2026 --month 9 --day 1

# 4. Generate linked synthetic data
python data/generate_data.py --records 20 --seed 42 --settlements data/settlements_live.csv

# 5. Run pipeline
python run_pipeline.py --demo-disagreement --records 20
```

## Configuration

| Environment Variable | Description | Required |
|---------------------|-------------|----------|
| `RAZORPAY_KEY_ID` | Razorpay test-mode API key ID | No (falls back to sample batch) |
| `RAZORPAY_KEY_SECRET` | Razorpay test-mode API secret | No (falls back to sample batch) |
| `GROQ_API_KEY` | Groq API key for GPT-OSS 120B exception reasoning | No (LLM disabled if missing) |
| `GROQ_MODEL` | Groq model identifier (default: `openai/gpt-oss-120b`, confirmed active 2026-09-04) | No |
| `API_SECRET_KEY` | API key for FastAPI REST authentication | No in demo (**default: `dev-secret-key`**). **Strictly required in production** (`ENV=production` or `DEMO_MODE=false`); server startup halts if unset or default. |
| `ENV` | Environment mode (`development` or `production`) | No (default: `development`) |
| `DEMO_MODE` | Demo mode toggle (`true` or `false`) | No (default: `true`) |
| `AUDIT_DB_PATH` | Path to SQLite audit trail database | No (default: `audit_trail.db` / `/app/runtime/audit_trail.db`) |
| `RESULTS_PATH` | Path to reconciliation results JSON | No (default: `results.json` / `/app/runtime/results.json`) |
| `METRICS_PATH` | Path to metrics evaluation JSON | No (default: `metrics_report.json` / `/app/runtime/metrics_report.json`) |

## Honest System Limitations & Security Invariants

As a genuinely submission-ready financial system, AuditLoop explicitly declares its invariants and limitations:

1. **3-Way Reconciliation Invariant:** A match *requires* all three legs (Settlement + Bank + Ledger). If a bank transaction is missing, the deterministic gate will explicitly reject any LLM proposal to match, producing an `llm_deterministic_disagreement`. The system fails closed.
2. **PII Sanitization:** The `llm/privacy.py` layer proactively redacts PAN numbers, IFSC codes, bank account numbers, UPI VPAs, phone numbers, and emails before the payload ever reaches the LLM. 
3. **Tamper-Evident Audit Log:** The SQLite audit log uses a SHA-256 hash chain (previous block hash + data = current hash) protected by immediate transactions and thread locks. It is *tamper-evident* (cryptographically verifiable via `/audit/verify`), not strictly "immutable" (since an attacker with root DB access could recalculate the chain, though this requires rewriting history).
4. **API Honesty:** The `/health` endpoint truthfully reports LLM status as `configured` (if the API key is present) rather than `connected`, since the system does not ping the LLM to verify upstream connectivity on every health check.
5. **Partial Refunds:** True multi-record partial refund aggregation (e.g., mapping one Ledger order row to multiple Settlements/Refunds) is not supported in the v1 matching engine. Such cases are correctly and safely routed to the exception queue for human review rather than being silently mis-matched.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_matcher.py -v
pytest tests/test_llm_schema_validation.py -v
pytest tests/test_end_to_end_metrics.py -v
```

## Key Metrics Explained

All headline metrics (precision, recall, F1) are computed on **input settlements** as the evaluation unit. Orphan bank rows, orphan ledger rows, and duplicate-suspect events are reported separately. Demo-injected disagreements (`forced_demo_case`) are excluded from TP/FP/FN.

The dashboard shows `total_input_transactions`, `evaluated_transactions`, and `ground_truth_coverage` so denominators are never implied.

Numbers below are **illustrative of a past local run**, not hardcoded product claims. Re-run `python run_pipeline.py --no-llm --records 20 --seed 42` and read `metrics_report.json`.

| Metric | What It Means | How it is computed |
|--------|---------------|---------------------|
| **Input transactions** | Settlement rows in the original input set | `len(settlements)` |
| **Evaluated transactions** | Input settlements that have a ground-truth label | input − unverified |
| **Precision** | TP / (TP + FP) over evaluated settlements | from `metrics_report.json` |
| **Recall** | TP / (TP + FN) over evaluated settlements | from `metrics_report.json` |
| **Orphan bank / ledger** | Unmatched bank or ledger events | excluded from the transaction denominator |
| **Demo-injected count** | Forced disagreement rows | excluded from organic TP/FP/FN |

> **Note on `reviewer_id` attribution:** The `reviewer_id` field in `/audit/resolve` is an **attribution display label** (e.g. employee ID) supplied in the request body for human review assignment and audit-trail record keeping. It is **not** a cryptographically verified identity — this prototype uses a single shared API key for authentication. In a production deployment, `reviewer_id` should be bound to an authenticated principal (e.g. OAuth2 sub claim). See `audit/store.py::resolve_exception` docstring.


## Regenerating Metrics

`metrics_report.json`, `metrics/metrics_report.json`, and
`runtime/runs/<run-id>/results.json` are
**always written by the pipeline** — never hand-edited. Per-run results live
under `runtime/runs/<run-id>/`; bundled `data/` fixtures are never overwritten.
To reproduce the exact numbers from scratch (same seed → same output):

```bash
# Step 1: Rebuild ground truth and synthetic data (same seed as pipeline)
# N=20 is the default synthetic batch size used by the CLI.
# Using a larger N without regenerating ground truth lowers coverage.
python data/build_ground_truth.py --records 20 --seed 42

# Step 2: Run the full reconciliation pipeline
#   (with LLM if GROQ_API_KEY is set, without otherwise)
python run_pipeline.py --seed 42 --records 20

# Step 3: Inspect metrics  (also written to metrics/metrics_report.json)
cat metrics_report.json

# Step 4: Optionally re-evaluate standalone (strict mode is the default;
# the evaluator consumes the structured PipelineResult natively)
python -m metrics.evaluate --results runtime/runs/<run-id>/results.json
```

After a fresh run, `ground_truth_coverage` should be **1.0** because
`build_ground_truth.py` generates a GT entry for every record in the batch.
`ground_truth_coverage` measures the **fraction of the processed batch** that
has a verified label, not merely utilization of the GT file.
If you change `--records` or `--seed`, always re-run Step 1 first so the
ground-truth file stays in sync.

## Understanding Disagreements

When you see `llm_deterministic_disagreement` cases in the dashboard:

1. The LLM proposed a match with action="match"
2. Our deterministic re-check rejected it (amount/date mismatch too large)
3. **This is intentional** - we fail closed, never open
4. These cases are flagged for human review

This is the core differentiator: *The LLM can propose, but never commit.*

To guarantee at least one LLM-vs-deterministic disagreement is visible in every demo run (these are rare in a small batch), you can pass `demo_disagreement=true` to `/reconcile` (or `--demo-disagreement` to `run_pipeline.py`). This injects one fully-labeled synthetic case (`forced_demo_case: true` in its audit record) — it does not affect organic exception processing metrics. Set it to `false` to see only organically-discovered disagreements.

## Reproducibility

Because `generate_data.py` writes a local CSV snapshot, the pipeline is deterministic for a given seed and snapshot. `fetch_settlements.py` only runs when Razorpay test-mode keys are present.

```bash
# Same seed + same snapshot = same results (proves determinism)
python run_pipeline.py --seed 42 --records 20
python run_pipeline.py --seed 42 --records 20  # Identical output

# Different seed = different synthetic bank/ledger data but valid results
python run_pipeline.py --seed 123 --records 20
python run_pipeline.py --seed 456 --records 20
```

## Synthetic Data Disclosure

Because no public sandboxes exist that supply multi-party reconciliation data (a Razorpay test settlement does not natively correspond to a mock bank statement and ERP ledger), AuditLoop ships a reproducible **synthetic** generator (`data/generate_data.py`).

The repository includes a reproducible synthetic benchmark dataset. The ingestion layer can consume Razorpay test-mode settlement data when configured. Bundled `data/settlements_live.csv` rows in this snapshot are tagged `source=synthetic`.

## Data Provenance

Every row in `data/settlements_live.csv`, `data/bank_statement.csv`, and
`data/internal_ledger.csv` carries a `source` column:

| Value | Meaning |
|-------|---------|
| `synthetic` | Generated by `data/generate_data.py` (this is the bundled snapshot) |
| `razorpay_test` | Fetched from Razorpay test-mode Settlement Recon API when keys are configured |
| `benchmark_fixture` | Hand-authored adversarial case |

**This bundled snapshot is synthetic.** It is not live Razorpay production data and it is not a verified Razorpay test-mode dump unless you fetch one yourself.

**How to verify per-row provenance from the data files:**

```bash
# Count rows by source in the settlements file
python -c "import pandas as pd; print(pd.read_csv('data/settlements_live.csv')['source'].value_counts())"

# Same for the latest per-run results.json (structured PipelineResult:
# transaction_results + orphan/duplicate/exception event streams)
python -c "import json, glob, os; p=sorted(glob.glob('runtime/runs/*/results.json'), key=os.path.getmtime)[-1]; d=json.load(open(p)); print(p); print({r.get('source','missing') for r in d['transaction_results']})"

# Same in the SQLite audit trail
python -c "
import sqlite3; conn = sqlite3.connect('audit_trail.db')
print(dict(conn.execute('SELECT source, COUNT(*) FROM audit_log GROUP BY source').fetchall()))
"
```

The `source` field is propagated end-to-end: generator → matcher → exception dispatcher → `results.json` → SQLite audit log. Any row with `source=None` or missing is a bug — `test_provenance.py` asserts this cannot happen.

> **Important:** Even when Razorpay API credentials are available, the bank statement and internal ledger counterparts are always **synthetic** (generated to match the settlement amounts/dates). No real ICICI/HDFC bank APIs are used — no public sandbox for those exists. Rows from the Razorpay API pull are tagged `razorpay_test`; the corresponding bank and ledger rows are tagged `synthetic`.

## Performance Benchmark


The deterministic matching engine is optimized for high-throughput scaling without LLM latency:
* **Stage 1 (Exact Hash Join):** O(N+M+L) complexity. On a 500-record batch, Stage 1 completes in < 0.5 seconds (asserted by `tests/test_performance_scaling.py`).
* **Stage 2 (Fuzzy Fee-Aware Match):** O(NM+NL) complexity. On the unresolved tail of a 500-record batch, Stage 2 completes in < 1.0 seconds.

## License

MIT License - Built for Razorpay AI Buildathon 2026
