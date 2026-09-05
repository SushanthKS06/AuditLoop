"""
FastAPI service exposing AuditLoop reconciliation as a prototype REST API.

Provides endpoints for programmatic reconciliation runs, metrics evaluation,
and real-time inspection of the append-only audit trail.
"""

from fastapi import FastAPI, HTTPException, Query, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
import logging
import os
import sys
import json
import secrets
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Add parent directory to path for clean package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_pipeline import ReconciliationPipeline
from audit.store import AuditStore
from audit.models import AuditEntry, AuditSummaryStats, HumanResolutionRequest, HumanResolutionResponse
from metrics.evaluate import MetricsEvaluator

from contextlib import asynccontextmanager

def validate_security_configuration():
    """
    Strict environment validation:
    In production or when DEMO_MODE is disabled, reject missing or default dev secret keys.
    """
    env = os.getenv("ENV", "development").strip().lower()
    demo_mode = os.getenv("DEMO_MODE", "true").strip().lower()
    is_production = env == "production" or demo_mode in ("false", "0", "no")
    secret_key = os.getenv("API_SECRET_KEY")

    if is_production:
        if not secret_key or secret_key.strip() == "dev-secret-key":
            raise RuntimeError(
                "CRITICAL SECURITY CONFIGURATION ERROR: Production deployment (or DEMO_MODE=false) "
                "requires API_SECRET_KEY to be set and not use default credentials ('dev-secret-key')."
            )

# Run security validation at module load
validate_security_configuration()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Verify security configuration on server startup."""
    validate_security_configuration()
    yield

app = FastAPI(
    title="AuditLoop Reconciliation API",
    description="Prototype REST API for multi-source financial reconciliation with deterministic-first matching, fee awareness, and LLM-assisted exception handling.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Warn loudly if running with the insecure demo default in development mode
if not os.getenv("API_SECRET_KEY"):
    logger.warning(
        "API_SECRET_KEY not set — using insecure default key 'dev-secret-key'. "
        "Set API_SECRET_KEY in production (see README Configuration section)."
    )

async def get_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.getenv("API_SECRET_KEY", "dev-secret-key")
    if not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=403,
            detail="Could not validate API KEY"
        )
    return api_key


class ReconcileRequest(BaseModel):
    """Request model for triggering reconciliation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "records": 50,
                "seed": 42,
                "messiness": 0.25,
                "demo_disagreement": True,
                "use_llm": True
            }
        }
    )
    records: int = Field(50, ge=1, le=1000, description="Number of records in the reconciliation batch")
    seed: int = Field(42, description="Random seed for deterministic reproducibility")
    messiness: float = Field(0.25, ge=0.0, le=1.0, description="Injected anomaly/messiness ratio")
    demo_disagreement: bool = Field(False, description="Inject one demo LLM-vs-deterministic disagreement (excluded from organic metrics)")
    demo_disagreement: bool = Field(False, description="Deprecated alias for demo_disagreement")
    use_llm: bool = Field(True, description="Enable LLM for Stage 3 exception explanation and resolution proposals")
    settlements_path: str = Field("data/settlements_live.csv", description="Path to Razorpay settlements CSV")
    bank_path: str = Field("data/bank_statement.csv", description="Path to bank statement CSV")
    ledger_path: str = Field("data/internal_ledger.csv", description="Path to internal ledger CSV")


class ReconcileResponse(BaseModel):
    """Response model for reconciliation execution."""
    status: str
    timestamp: str
    matches_count: int
    exceptions_count: int
    disagreements_count: int
    audit_log_count: int
    metrics: Dict[str, Any]
    audit_summary: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    timestamp: str
    deterministic_engine: str = "active"
    audit_store: str = "sqlite_wal"
    llm_status: str = "not_configured"


@app.get("/health", response_model=HealthResponse, dependencies=[Depends(get_api_key)])
async def health_check():
    """
    Health check endpoint.
    Returns service health, version, and component status.
    Decoupled from LLM availability: returns status='healthy' (HTTP 200) as long as
    the core deterministic matching engine and SQLite audit store are operational,
    with llm_status indicating LLM connectivity/configuration status.
    """
    llm_key = os.getenv("GROQ_API_KEY")
    groq_available = True
    try:
        import groq as _groq  # noqa: F401
    except ImportError:
        groq_available = False

    if not llm_key or len(llm_key.strip()) <= 5:
        llm_status = "not_configured"
    elif not groq_available:
        llm_status = "degraded"
    else:
        # Key present; no live provider ping (health must stay cheap).
        llm_status = "configured"

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
        deterministic_engine="active",
        audit_store="sqlite_wal",
        llm_status=llm_status
    )


@app.post("/reconcile", response_model=ReconcileResponse, dependencies=[Depends(get_api_key)])
def run_reconciliation(request: ReconcileRequest):
    """
    Run the full end-to-end reconciliation pipeline.
    
    Executed in FastAPI worker threadpool to prevent CPU-bound event loop starvation.
    
    Pipeline Steps:
    1. Ingestion / Data generation with realistic messiness
    2. Stage 1: Exact matching on normalized UTR / Order ID / Payment ID (O(N+M+L))
    3. Stage 2: Fuzzy matching with MDR fee-deduction awareness
    4. Stage 3: LLM exception explanation & proposal generation with CoT & PII masking
    5. Deterministic re-verification of all LLM match proposals (fail-closed)
    6. Cryptographically chained audit trail persistence (SHA-256)
    7. Mathematical metrics computation against ground truth
    """
    try:
        pipeline = ReconciliationPipeline(
            use_llm=request.use_llm,
            demo_disagreement_demo=bool(request.demo_disagreement or request.demo_disagreement)
        )
        
        results = pipeline.run(
            settlements_path=request.settlements_path,
            bank_path=request.bank_path,
            ledger_path=request.ledger_path,
            generate_if_missing=True,
            num_records=request.records,
            seed=request.seed,
            messiness_ratio=request.messiness  # propagate API param — never silently discard
        )
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        audit_store = AuditStore()
        audit_count = audit_store.count()
        
        return ReconcileResponse(
            status="completed",
            timestamp=datetime.now(timezone.utc).isoformat(),
            matches_count=results.get("matches_count", 0),
            exceptions_count=results.get("exceptions_count", 0),
            disagreements_count=results.get("metrics", {}).get("disagreement_count", 0),
            audit_log_count=audit_count,
            metrics=results.get("metrics", {}),
            audit_summary=results.get("audit_summary", {})
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error_type": "bad_input", "message": str(e)}
        )
    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail={"error_type": "audit_failure", "message": "Persistence error"}
        )
    except Exception as e:
        logger.exception("Reconciliation execution failed")
        raise HTTPException(
            status_code=500,
            detail={"error_type": "internal_server_failure", "message": "Reconciliation execution failed"}
        )


@app.get("/audit/verify", response_model=Dict[str, Any], dependencies=[Depends(get_api_key)])
def verify_audit_integrity():
    """
    Cryptographically verify the entire append-only audit trail.
    
    Recalculates SHA-256 block hashes from genesis to head and proves
    that zero records have been altered, deleted, or injected.
    """
    store = AuditStore()
    return store.verify_integrity()


@app.get("/metrics", response_model=Dict[str, Any], dependencies=[Depends(get_api_key)])
async def get_latest_metrics():
    """
    Get latest reconciliation metrics from the most recent run.
    Reads metrics_report.json and returns precision, recall, F1, and error rates.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    paths_to_check = []
    env_metrics = os.getenv("METRICS_PATH")
    if env_metrics:
        paths_to_check.append(env_metrics)
    paths_to_check.extend([
        os.path.join(base_dir, "metrics", "metrics_report.json"),
        os.path.join(base_dir, "metrics_report.json")
    ])
    
    for p in paths_to_check:
        if os.path.exists(p):
            with open(p, 'r') as f:
                return json.load(f)
    
    raise HTTPException(
        status_code=404,
        detail="No metrics report found. Run /reconcile or run_pipeline.py first."
    )


@app.get("/audit/recent", response_model=List[Dict[str, Any]], dependencies=[Depends(get_api_key)])
async def get_recent_audit_entries(
    limit: int = Query(20, ge=1, le=500, description="Maximum entries to return"),
    status: Optional[str] = Query(None, description="Filter by final status (e.g. matched, llm_deterministic_disagreement)")
):
    """
    Get recent entries from the tamper-evident SQLite audit log.
    Supports optional status filtering and pagination limit.
    """
    store = AuditStore()
    
    if status:
        entries = store.get_by_status(status)
        return entries[-limit:] if entries else []
    
    return store.get_recent(limit=limit)


@app.get("/audit/disagreements", response_model=List[Dict[str, Any]], dependencies=[Depends(get_api_key)])
async def get_disagreements():
    """
    Get all cases where the LLM proposal conflicted with deterministic re-verification.
    Proves the fail-closed Failure Recovery guarantee.
    """
    store = AuditStore()
    return store.get_disagreements()


@app.get("/audit/exceptions", response_model=List[Dict[str, Any]], dependencies=[Depends(get_api_key)])
async def get_unresolved_exceptions():
    """
    Get all unresolved exceptions requiring human reviewer inspection.
    """
    store = AuditStore()
    return store.get_exceptions()


@app.get("/audit/summary", response_model=Dict[str, Any], dependencies=[Depends(get_api_key)])
async def get_audit_summary():
    """
    Get high-level throughput and confidence summary statistics from the audit log.
    """
    store = AuditStore()
    return store.get_summary_stats()


@app.post("/audit/resolve", response_model=HumanResolutionResponse, dependencies=[Depends(get_api_key)])
def resolve_audit_exception(request: HumanResolutionRequest):
    """
    Human-in-the-Loop Maker-Checker endpoint.
    Allows an authorized financial controller to manually resolve a flagged exception or disagreement.
    
    NOTE ON ATTRIBUTION:
    reviewer_id is an attribution display label supplied in the request body for audit trail
    logging and human review assignment, NOT an authenticated cryptographic identity.
    
    The decision is appended to the SHA-256 chained audit trail with cryptographic tamper-evidence.
    """
    store = AuditStore()
    try:
        result = store.resolve_exception(
            record_ids=request.record_ids,
            reviewer_id=request.reviewer_id,
            decision=request.decision,
            notes=request.notes
        )
        return HumanResolutionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.get("/audit/history", response_model=List[Dict[str, Any]], dependencies=[Depends(get_api_key)])
async def get_record_audit_history(
    record_ids: str = Query(..., description="Target record_ids pair to inspect full history for")
):
    """
    Inspect the complete chronological decision history of a specific record pair across all stages.
    """
    store = AuditStore()
    return store.get_record_history(record_ids)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

