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
│   ├── matcher.py               # Vectorized exact + fuzzy matching (Stage 1 & 2 O(N+M))
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

Python · FastAPI · pandas · RapidFuzz · Groq API (LLaMA 3.3 70B function calling +
Pydantic schemas) · Razorpay API · SQLite (SHA-256 Chained) · Streamlit · pytest · Docker

## API Endpoints (Production Ready)

AuditLoop exposes a production REST API for programmatic access and institutional compliance:

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
curl http://localhost:8000/health

# Cryptographically verify the audit chain
curl http://localhost:8000/audit/verify

# Maker-Checker: Human Controller Manual Resolution
curl -X POST http://localhost:8000/audit/resolve \
  -H "Content-Type: application/json" \
  -d '{
    "record_ids": "sett_001-TXN_001",
    "decision": "human_approved_match",
    "reviewer_id": "CONTROLLER_001",
    "notes": "Verified gateway MDR fee discrepancy with signed merchant invoice."
  }'

# Run reconciliation with custom parameters
curl -X POST http://localhost:8000/reconcile \
  -H "Content-Type: application/json" \
  -d '{"records": 50, "seed": 42, "messiness": 0.25, "force_disagreement": true}'

# Get latest metrics
curl http://localhost:8000/metrics

# View recent audit entries
curl http://localhost:8000/audit/recent?limit=10
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
python data/generate_data.py --records 80 --seed 42

# Run the full pipeline
python run_pipeline.py --force-disagreement --records 50

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
python data/generate_data.py --records 80 --seed 42 --settlements data/settlements_live.csv

# 5. Run pipeline
python run_pipeline.py --force-disagreement
```

## Configuration

| Environment Variable | Description | Required |
|---------------------|-------------|----------|
| `RAZORPAY_KEY_ID` | Razorpay test-mode API key ID | No (falls back to sample batch) |
| `RAZORPAY_KEY_SECRET` | Razorpay test-mode API secret | No (falls back to sample batch) |
| `GROQ_API_KEY` | Groq API key for LLaMA 3.3 70B exception reasoning | No (LLM disabled if missing) |
| `GROQ_MODEL` | Groq model identifier (default: `llama-3.3-70b-versatile`) | No |

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

| Metric | What It Means | Target |
|--------|---------------|--------|
| **Match Rate** | % of records successfully reconciled | >80% |
| **Precision** | Of all matches, how many are correct (vs ground truth) | >90% |
| **Recall** | Of all true matches, how many did we find | >75% |
| **Disagreement Rate** | LLM vs deterministic conflicts | <10% |
| **Unresolved Rate** | Exceptions needing human review | <5% |

## Understanding Disagreements

When you see `llm_deterministic_disagreement` cases in the dashboard:

1. The LLM proposed a match with action="match"
2. Our deterministic re-check rejected it (amount/date mismatch too large)
3. **This is intentional** - we fail closed, never open
4. These cases are flagged for human review

This is the core differentiator: *The LLM can propose, but never commit.*

To guarantee at least one LLM-vs-deterministic disagreement is visible in every demo run (these are rare in a small batch), you can pass `force_disagreement=true` to `/reconcile` (or `--force-disagreement` to `run_pipeline.py`). This injects one fully-labeled synthetic case (`forced_demo_case: true` in its audit record) — it does not affect real exception processing. Set it to `false` to see only organically-discovered disagreements.

## Reproducibility

```bash
# Same seed = same results (proves determinism)
python run_pipeline.py --seed 42 --records 50
python run_pipeline.py --seed 42 --records 50  # Identical output

# Different seed = different but valid results (proves not hardcoded)
python run_pipeline.py --seed 123 --records 50
python run_pipeline.py --seed 456 --records 50
```

## License

MIT License - Built for Razorpay AI Buildathon 2026
