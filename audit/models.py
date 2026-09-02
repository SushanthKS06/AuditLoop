"""
Audit Trail Models

Pydantic models representing immutable audit log entries, decision records,
and summary metrics for the reconciliation lifecycle.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class AuditEntry(BaseModel):
    """
    Representation of an immutable decision logged to the audit trail with cryptographic hash chaining.
    """
    model_config = ConfigDict(from_attributes=True)
    
    id: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    record_ids: str = Field(..., description="Hyphen-delimited identifier pair, e.g. sett_001-TXN_001")
    stage: str = Field(..., description="Stage where decision was made: stage1_exact, stage2_fuzzy, stage3_llm")
    rule_fired: Optional[str] = Field(None, description="Deterministic rule name or LLM root cause")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    decision: str = Field(..., description="Outcome: matched, low_confidence, unresolved_exception, etc.")
    match_type: Optional[str] = Field(None, description="exact_utr, exact_order, fuzzy_bank, fuzzy_ledger, llm_verified")
    llm_reasoning: Optional[str] = Field(None, description="Explanation text if LLM was invoked")
    final_status: Optional[str] = Field(None, description="Final resolved state in pipeline")
    previous_hash: Optional[str] = Field(None, description="SHA-256 hash of previous audit record")
    record_hash: Optional[str] = Field(None, description="SHA-256 hash of this record (previous_hash + payload)")


class AuditSummaryStats(BaseModel):
    """
    Aggregated health, throughput, and cryptographic integrity statistics from the audit log.
    """
    total_records: int = 0
    matched: int = 0
    exceptions: int = 0
    disagreements: int = 0
    match_rate: float = 0.0
    average_confidence: float = 0.0
    integrity_verified: bool = True
    tampered_records_count: int = 0


class HumanResolutionRequest(BaseModel):
    """
    Model for human controller manual resolution / Maker-Checker sign-off.
    """
    record_ids: str = Field(..., description="Target record_ids pair to resolve")
    decision: Literal["human_approved_match", "human_rejected_duplicate", "human_written_off"] = Field(
        ..., description="Manual review decision"
    )
    reviewer_id: str = Field(..., min_length=3, description="Employee/Controller ID performing the review")
    notes: str = Field(..., min_length=5, description="Auditable rationale for the override")


class HumanResolutionResponse(BaseModel):
    """
    Response model confirming cryptographic persistence of human decision.
    """
    status: str = "success"
    audit_entry_id: int
    record_ids: str
    decision: str
    reviewer_id: str
    record_hash: str
    integrity_verified: bool



