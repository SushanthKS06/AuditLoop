# Changelog

All notable changes to AuditLoop are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-01

### Added
- **Initial release for Razorpay AI Buildathon 2026 — Track 04**
- Deterministic matching engine with two-stage pipeline:
  - Stage 1: Exact match on normalized amount/date/reference fields
  - Stage 2: Fuzzy match with configurable confidence threshold using RapidFuzz
- LLM exception handling layer (Anthropic Claude API):
  - `explain_exception` tool with enum-constrained root cause classification
  - `propose_resolution` tool with strict Pydantic schema validation
  - **Critical safety**: Deterministic re-verification of all LLM-proposed matches
- Audit trail system (SQLite, append-only):
  - Every decision logged with record_ids, stage_reached, confidence, llm_reasoning
  - Immutable log with current-status view
- Metrics harness comparing results against ground_truth.json:
  - Precision, recall, false-positive rate, match rate
  - Reproducible across different random seeds
- Streamlit dashboard for reviewer inspection:
  - Summary metrics header
  - Exception drill-down with field-level diffs
  - Dedicated "Disagreements" tab for llm_deterministic_disagreement cases
- Data generation pipeline:
  - Real Razorpay test-mode Settlement Recon API integration
  - Synthetic bank statement and ledger generation with tunable messiness
  - Hybrid fallback when live API returns insufficient records
  - Ground truth answer key generation
- **FastAPI REST API** (NEW):
  - `POST /reconcile` - Run full pipeline with custom parameters
  - `GET /metrics` - Get latest metrics report
  - `GET /audit/recent` - Inspect recent audit entries
  - `GET /health` - Health check endpoint
- Docker packaging for one-command deployment
- Comprehensive test suite:
  - `test_matcher.py` - Deterministic matching unit tests
  - `test_llm_schema_validation.py` - Pydantic schema validation tests
  - `test_end_to_end_metrics.py` - Full pipeline reproducibility tests
- CHANGELOG.md for version tracking

### Design Decisions
- **Deterministic-first architecture**: LLM never has unilateral authority over financial matches
- **Fail-closed behavior**: Ambiguous cases become visible exceptions, never silent matches
- **Bounded LLM context**: Only exceptions are sent to LLM, not entire dataset
- **No agent framework**: Explicit orchestration for inspectability and defendability
- **Honest data sourcing**: Clear tagging of real vs synthetic data; no fabricated "live" claims

### Technical Stack
- Python 3.11+
- FastAPI, pandas, RapidFuzz
- Anthropic Claude SDK (tool calling + Pydantic schemas)
- Razorpay API (test-mode Settlement Recon)
- SQLite (audit log)
- Faker (synthetic data)
- Streamlit (dashboard)
- pytest (testing)
- Docker + docker-compose
- Uvicorn (ASGI server for API)

---

## [Unreleased]

### Planned Improvements
- Export architecture diagram as PNG in addition to Excalidraw source
- Expand sample_batch to 8-10 records for richer zero-key demo
- Add video script (SCRIPT.md) for 5-minute pitch
- Document live metrics from seed=42 run in README

### Future Extensions (Post-Buildathon)
- Real bank statement API integrations (Plaid, Yodlee, account aggregators)
- ERP connectors (SAP, Oracle, Tally)
- Configurable confidence thresholds via dashboard UI
- Batch export of audit logs for compliance reporting
- Multi-currency normalization support

---

## Version History Notes

This changelog reflects the disciplined, scope-ruthless approach required for a 5-7 day buildathon submission. Every feature included was chosen because it directly supports one of the four judging axes:

1. **Problem Taste** — Clear thesis, honest data story
2. **Build Quality** — Clean architecture, comprehensive tests
3. **AI Judgment** — Bounded LLM scope, schema enforcement
4. **Failure Recovery** — Visible disagreements, fail-closed design

Features that were explicitly **not** included (and why):
- No LangGraph/CrewAI/AutoGen — would require justification under pressure
- No full database abstraction — SQLite direct access is sufficient for MVP
- No user authentication — single-user reviewer dashboard is adequate
- No real-time streaming — batch processing matches the use case

The restraint shown in scope decisions is itself a demonstration of engineering judgment.
