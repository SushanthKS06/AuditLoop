# AuditLoop

**A reconciliation agent that never lets an LLM have the final say on a financial match.**

Multi-source reconciliation agent for Razorpay settlements, bank statements,
and internal ledgers. A deterministic matching engine runs first; an LLM is
only invoked to explain and propose resolutions for unresolved exceptions —
it can never commit a match directly, every proposal is re-verified
deterministically before it counts. Every decision, matched or not, is
logged to an audit trail. Accuracy is measured against a known ground-truth
batch, not demoed on cherry-picked examples.

Settlements data is pulled live from Razorpay's test-mode Settlement Recon
API; bank statement and ledger data are synthetic (clearly tagged where
used) since no equivalent sandbox exists for those.

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

AuditLoop exposes a prototype REST API for programmatic access and institutional compliance:

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
  -d '{"records": 20, "seed": 42, "messiness": 0.25, "force_disagreement": true}'

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

> **Record Count Note:** All Quick Start workflows and evaluation benchmarks canonically process **20 settlement records** (derived from the 20 live Razorpay test-mode settlements in `data/settlements_live.csv` and matched 1-to-1 in `data/ground_truth.json`). In multi-source generation, this produces 24 reconciliation events across bank, ledger, and gateway legs.

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
python run_pipeline.py --force-disagreement --records 20

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
python run_pipeline.py --force-disagreement --records 20
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

All headline metrics (precision, recall, F1) are computed in **strict coverage
mode**: only records that have a matching entry in `data/ground_truth.json` are
scored. Records without a ground-truth entry are excluded and counted as
`unverified_count`. The dashboard always displays `ground_truth_coverage` next
to the metrics so this number is never silently assumed.

*Note: The ground truth and test data share the same generator. The pipeline intentionally uses a high `messiness_ratio` parameter (0.40) to inject genuine ambiguities and edge cases into the batch, making the resulting precision and recall scores meaningful rather than trivially perfect.*

| Metric | What It Means | Target / Verified Benchmark |
|--------|---------------|-----------------------------|
| **Match Rate** | % of records successfully reconciled | ~66.7% (16/24 on messiness=0.40; >80% on clean batches) |
| **Precision** | Of verified matches, how many are correct against ground truth | 93.8% (15/16 true positive matches) |
| **Recall** | Of all true matches in ground truth, how many did we find | 88.2% (15/17 true matches recovered) |
| **Disagreement Rate** | LLM vs deterministic conflicts | 0.0% (clean run) / 4.2% (forced demo case) |
| **Unresolved Rate** | Exceptions needing human review | 16.7% (4 of 24 records routed to human controller) |
| **Ground-Truth Coverage** | Fraction of the processed batch that has a verified ground-truth label (not coverage of the GT file) | 1.0 (100.0% — 24 of 24 records verified against 20-entry GT file) |

> **Note on `reviewer_id` attribution:** The `reviewer_id` field in `/audit/resolve` is an **attribution display label** (e.g. employee ID) supplied in the request body for human review assignment and audit-trail record keeping. It is **not** a cryptographically verified identity — this prototype uses a single shared API key for authentication. In a production deployment, `reviewer_id` should be bound to an authenticated principal (e.g. OAuth2 sub claim). See `audit/store.py::resolve_exception` docstring.


## Regenerating Metrics

`metrics_report.json`, `metrics/metrics_report.json`, and `results.json` are
**always written by the pipeline** — never hand-edited. To reproduce the exact
numbers from scratch (same seed → same output):

```bash
# Step 1: Rebuild ground truth and synthetic data (same seed as pipeline)
# N=20 matches the 20 real settlements in data/settlements_live.csv.
# Using a larger N generates unlabeled synthetic-only records and lowers
# ground_truth_coverage below 1.0 — always use the same N in both commands.
python data/build_ground_truth.py --records 20 --seed 42

# Step 2: Run the full reconciliation pipeline
#   (with LLM if GROQ_API_KEY is set, without otherwise)
python run_pipeline.py --seed 42 --records 20

# Step 3: Inspect metrics  (also written to metrics/metrics_report.json)
cat metrics_report.json

# Step 4: Optionally re-evaluate standalone (strict mode is the default)
python -m metrics.evaluate --results results.json
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

To guarantee at least one LLM-vs-deterministic disagreement is visible in every demo run (these are rare in a small batch), you can pass `force_disagreement=true` to `/reconcile` (or `--force-disagreement` to `run_pipeline.py`). This injects one fully-labeled synthetic case (`forced_demo_case: true` in its audit record) — it does not affect real exception processing. Set it to `false` to see only organically-discovered disagreements.

## Reproducibility

Because `fetch_settlements.py` caches the live Razorpay API pull to a local CSV snapshot (`data/settlements_live.csv`), the entire pipeline is fully deterministic and reproducible as long as you use the same snapshot and seed.

```bash
# Same seed + same snapshot = same results (proves determinism)
python run_pipeline.py --seed 42 --records 20
python run_pipeline.py --seed 42 --records 20  # Identical output

# Different seed = different synthetic bank/ledger data but valid results
python run_pipeline.py --seed 123 --records 20
python run_pipeline.py --seed 456 --records 20
```

## Synthetic Data Disclosure

Because no public sandboxes exist that supply multi-party reconciliation data (i.e. where a Razorpay API test settlement natively corresponds to a mock ICICI bank statement and a mock internal ERP ledger), AuditLoop leverages a robust synthetic data generator (`data/generate_data.py`). 

The generator explicitly links to live Razorpay Test-Mode Settlement data (if API keys are provided) and generates the corresponding Bank and Ledger counterparts. This ensures the matching engine is tested on realistic data volumes and anomalies (fees, taxes, fuzzy dates) while preserving the integrity of the evaluation.

## Data Provenance

Every row in `data/settlements_live.csv`, `data/bank_statement.csv`, and
`data/internal_ledger.csv` carries a `source` column with one of two values:

| Value | Meaning |
|-------|---------|
| `razorpay_test` | Fetched from Razorpay's test-mode Settlement Recon API (`data/fetch_settlements.py`) |
| `synthetic` | Fully synthetic record generated by `data/generate_data.py` |

**How to verify per-row provenance from the data files:**

```bash
# Count rows by source in the settlements file
python -c "import pandas as pd; print(pd.read_csv('data/settlements_live.csv')['source'].value_counts())"

# Same for results.json
python -c "import json; d=json.load(open('results.json')); print({r.get('source','missing') for r in d})"

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
