"""
Explicit state machine definitions for AuditLoop Reconciliation.

Centralizing these states prevents arbitrary strings and ensures
the evaluation/metrics engines have a stable API to measure.
"""

from enum import Enum

class ReconciliationState(str, Enum):
    # Deterministic Matches (Stage 1 & 2)
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    
    # Legacy compatibility state (pipeline expects 'matched' for exact/fuzzy)
    MATCHED = "matched"
    
    # LLM-Assisted States (Stage 3)
    MATCHED_LLM_VERIFIED = "matched_llm_verified"
    LLM_DETERMINISTIC_DISAGREEMENT = "llm_deterministic_disagreement"
    FLAGGED_FOR_REVIEW = "flagged_for_review"
    REJECTED_DUPLICATE = "rejected_duplicate"
    EXPLAINED_NO_RESOLUTION = "explained_no_resolution"
    
    # Unresolved / Exceptions
    UNRESOLVED_EXCEPTION = "unresolved_exception"
    LOW_CONFIDENCE = "low_confidence"
    
    # Errors
    LLM_ERROR = "llm_error"
    LLM_PARSE_ERROR = "llm_parse_error"
    LLM_UNAVAILABLE = "llm_unavailable"
    
    # Exception Types (Orphans, Data Issues)
    UNMATCHED_BANK = "unmatched_bank"
    UNMATCHED_LEDGER = "unmatched_ledger"
    UNMATCHED_SETTLEMENT = "unmatched_settlement"
    
    @classmethod
    def is_match(cls, state: str) -> bool:
        return state in {
            cls.MATCHED,
            cls.EXACT_MATCH,
            cls.FUZZY_MATCH,
            cls.MATCHED_LLM_VERIFIED,
        }
