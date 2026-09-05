"""
Explicit state machine for AuditLoop reconciliation.

Meaning, entry conditions, evaluation behavior, and audit behavior
are documented on each member. There are no redundant synonyms.

Critical invariant:
    LLM proposes MATCH + deterministic rejection
    → LLM_DETERMINISTIC_DISAGREEMENT or INCOMPLETE_COUNTERPARTS
    Never silently collapse into generic review without the exact reason.
"""

from enum import Enum


class ReconciliationState(str, Enum):
    # ── Deterministic matches (Stage 1 & 2) ──────────────────────────────
    EXACT_MATCH = "exact_match"
    # Entry: all three legs present; IDs and amounts agree within Stage-1 thresholds.
    # Evaluation: counted as a match. Audit: stage1_exact.

    FUZZY_MATCH = "fuzzy_match"
    # Entry: all three legs present; Stage-2 score ≥ confidence_threshold on both legs.
    # Evaluation: counted as a match. Audit: stage2_fuzzy.

    MATCHED = "matched"
    # Compatibility alias used by the adversarial benchmark runner for Stage 1/2 hits.
    # Same evaluation behavior as EXACT_MATCH / FUZZY_MATCH.

    # ── LLM-assisted (Stage 3) ────────────────────────────────────────────
    LLM_PROPOSED = "llm_proposed"
    # Intermediate: LLM returned a valid proposal; not a terminal pipeline status.

    MATCHED_LLM_VERIFIED = "matched_llm_verified"
    # Entry: LLM action=match AND ReconciliationContext has all three legs
    # AND deterministic verifier confirms amounts/currency/dates.
    # Evaluation: counted as a match. Audit: stage3_llm.

    LLM_DETERMINISTIC_DISAGREEMENT = "llm_deterministic_disagreement"
    # Entry: LLM action=match AND all legs present AND verifier rejects
    # (amount, currency, date, or identifier conflict).
    # Evaluation: not a match; counted in disagreement_count. Audit: preserved verbatim.

    INCOMPLETE_COUNTERPARTS = "incomplete_counterparts"
    # Entry: LLM action=match (or Stage 1/2 candidate) but bank and/or ledger missing.
    # Evaluation: not a match. Distinct from disagreement so the reason is explicit.

    LOW_CONFIDENCE = "low_confidence"
    # Entry: Stage-2 score below auto-match threshold, or LLM proposal confidence < 0.5.
    # Evaluation: not a match. Audit: stage2_fuzzy or stage3_llm.

    DUPLICATE_SUSPECT = "duplicate_suspect"
    REJECTED_DUPLICATE = "rejected_duplicate"
    # Entry: LLM action=reject_duplicate, or identifier conflict during merge.
    # Evaluation: not a match; counted in duplicate_suspects.

    ORPHAN_RECORD = "orphan_record"
    UNMATCHED_BANK = "unmatched_bank"
    UNMATCHED_LEDGER = "unmatched_ledger"
    UNMATCHED_SETTLEMENT = "unmatched_settlement"
    # Orphan / unmatched legs. Evaluation: bank/ledger orphans are NOT transaction units.

    FLAGGED_FOR_REVIEW = "flagged_for_review"
    HUMAN_REVIEW = "human_review"
    # LLM or policy routed to a human. Evaluation: not a match; counted in review_count.

    EXPLAINED_NO_RESOLUTION = "explained_no_resolution"
    UNRESOLVED_EXCEPTION = "unresolved_exception"
    # Explained but cannot resolve (e.g. no counterpart), or generic unresolved tail.

    REJECTED = "rejected"
    # Deterministic rejection without an LLM proposal (policy veto).

    # ── Errors ────────────────────────────────────────────────────────────
    LLM_ERROR = "llm_error"
    LLM_PARSE_ERROR = "llm_parse_error"
    LLM_UNAVAILABLE = "llm_unavailable"
    LLM_PROVIDER_FAILURE = "llm_provider_failure"
    ERROR = "error"
    # Provider/schema/internal failures. Not counted as reconciliation inaccuracy
    # in the business-logic benchmark (which uses MockLLM).

    @classmethod
    def is_match(cls, state: str) -> bool:
        return state in {
            cls.MATCHED.value,
            cls.EXACT_MATCH.value,
            cls.FUZZY_MATCH.value,
            cls.MATCHED_LLM_VERIFIED.value,
        }

    @classmethod
    def is_orphan_event(cls, type_or_status: str) -> bool:
        return type_or_status in {
            cls.UNMATCHED_BANK.value,
            cls.UNMATCHED_LEDGER.value,
            cls.ORPHAN_RECORD.value,
        }

    @classmethod
    def is_review(cls, state: str) -> bool:
        return state in {
            cls.FLAGGED_FOR_REVIEW.value,
            cls.HUMAN_REVIEW.value,
            cls.LOW_CONFIDENCE.value,
            cls.INCOMPLETE_COUNTERPARTS.value,
            cls.UNRESOLVED_EXCEPTION.value,
            cls.EXPLAINED_NO_RESOLUTION.value,
        }

    @classmethod
    def is_rejected(cls, state: str) -> bool:
        return state in {
            cls.REJECTED.value,
            cls.REJECTED_DUPLICATE.value,
            cls.DUPLICATE_SUSPECT.value,
            cls.LLM_DETERMINISTIC_DISAGREEMENT.value,
        }
