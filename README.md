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
│   └── app.py                    # FastAPI REST API endpoints
├── data/
│   ├── fetch_settlements.py     # Real Razorpay test-mode API pull
│   ├── generate_data.py         # Synthetic bank + ledger data, linked to live pull
│   ├── ground_truth.json        # Answer key for measuring precision/recall
│   └── sample_batch/            # Committed example batch (runs with zero API keys)
├── engine/
│   ├── matcher.py               # Deterministic exact + fuzzy matching (Stage 1 & 2)
│   └── exceptions.py            # Stage 3 — dispatch unresolved records to the LLM
├── llm/
│   ├── client.py                 # Claude API wrapper
│   ├── schemas.py                 # Pydantic response models (strict validation)
│   └── prompts.py                 # Scoped system + tool prompts
├── audit/
│   ├── store.py                   # SQLite audit log (append-only)
│   └── models.py
├── metrics/
│   └── evaluate.py                # Precision / recall / match-rate vs ground truth
├── dashboard/
│   └── app.py                     # Streamlit reviewer dashboard
├── tests/
│   ├── test_matcher.py
│   ├── test_llm_schema_validation.py
│   └── test_end_to_end_metrics.py
├── docs/
│   ├── architecture.png
│   └── DESIGN_DECISIONS.md        # Why deterministic-first, thresholds, tradeoffs
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Tech Stack

Python · FastAPI · pandas · RapidFuzz · Anthropic Claude (tool calling +
Pydantic schemas) · Razorpay API · SQLite · Streamlit · pytest · Docker

## API Endpoints (New - Production Ready)

AuditLoop now exposes a REST API for programmatic access:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with version info |
| `/reconcile` | POST | Run full reconciliation pipeline |
| `/metrics` | GET | Get latest metrics report |
| `/audit/recent` | GET | Inspect recent audit log entries |

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

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
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | No (LLM disabled if missing) |

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
