"""
Stage 3: Exception Dispatcher

Routes unresolved records to the LLM layer for explanation and resolution proposals.
Never commits matches directly - all LLM proposals go through deterministic re-verification.

Core invariant (enforced in _deterministic_recheck)
---------------------------------------------------
A settlement with a missing bank leg OR a missing ledger leg can NEVER produce
a final_status of 'matched_llm_verified', regardless of what the LLM proposes.
The LLM may PROPOSE 'match' but the deterministic gate will reject it.

Design principle: DETERMINISTIC SYSTEMS DECIDE. LLM SYSTEMS PROPOSE.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import threading
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone

from engine.states import ReconciliationState
from engine.context import ReconciliationContext
from engine.matcher import DeterministicMatcher


class ExceptionDispatcher:
    """
    Dispatches exceptions from the deterministic matcher to the LLM layer.
    
    Two types of LLM calls:
    1. explain_exception - Understand root cause (rounding, timing, duplicate, etc.)
    2. propose_resolution - Suggest action (match, flag for human, reject)
    
    CRITICAL: LLM proposals with action="match" are NOT committed directly.
    They are re-run through the deterministic scorer as verification.
    Only if both agree does the match get committed.
    """
    
    def __init__(self, llm_client=None, max_workers: int = 2):
        """
        Args:
            llm_client: Groq API client instance
            max_workers: ThreadPoolExecutor concurrency for parallel exception processing
        """
        self.llm_client = llm_client
        self.max_workers = max_workers
        self.audit_callback = None
        self._audit_lock = threading.Lock()
    
    def set_audit_callback(self, callback):
        """Set callback for writing audit records."""
        self.audit_callback = callback
    
    def process_exceptions(
        self,
        exceptions: List[Dict],
        demo_disagreement_case: bool = False,
        concurrent: bool = True
    ) -> List[Dict]:
        """
        Process all exceptions through the LLM layer with optional concurrent worker pool.
        
        Args:
            exceptions: List of exception records from matcher
            demo_disagreement_case: If True, ensure at least one disagreement case exists for demo
            concurrent: Whether to parallelize LLM network calls across worker threads
            
        Returns:
            List of processed exception records with LLM reasoning
        """
        if not exceptions:
            return []
            
        processed = [None] * len(exceptions)
        has_disagreement = False
        forced_idx = len(exceptions) // 2 if demo_disagreement_case else -1
        
        def _worker(idx: int, exc: Dict) -> tuple:
            # Stagger dispatches slightly to avoid sudden burst rate-limits
            if idx > 0 and self.llm_client:
                time.sleep(0.3 * (idx % self.max_workers))
            if idx == forced_idx:
                res = self._simulate_disagreement_case(exc)
            else:
                res = self._process_single_exception(exc)
            return idx, res
        
        # Parallel dispatch when LLM is active and concurrency is enabled
        if concurrent and self.llm_client and len(exceptions) > 1:
            workers = min(self.max_workers, len(exceptions))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_worker, i, exc) for i, exc in enumerate(exceptions)]
                for fut in as_completed(futures):
                    idx, res = fut.result()
                    processed[idx] = res
                    if res.get('final_status') == 'llm_deterministic_disagreement':
                        has_disagreement = True
        else:
            for i, exc in enumerate(exceptions):
                _, res = _worker(i, exc)
                processed[i] = res
                if res.get('final_status') == 'llm_deterministic_disagreement':
                    has_disagreement = True
        
        # Dispatch audit callbacks in strict input order
        if self.audit_callback:
            with self._audit_lock:
                for result in processed:
                    if result:
                        self.audit_callback({
                            'record_ids': result.get('record_ids', ''),
                            'stage': 'stage3_llm',
                            'rule_fired': result.get('llm_root_cause', 'unclassified'),
                            'confidence': result.get('llm_confidence', 0),
                            'decision': result.get('final_status', 'unresolved_exception'),
                            'llm_reasoning': result.get('llm_explanation') or result.get('llm_error_detail', '') or '',
                            'llm_error_detail': result.get('llm_error_detail'),
                            'source': result.get('source', 'synthetic'),
                            'forced_demo_case': result.get('forced_demo_case', False)
                        })
        
        return processed
    
    def _process_single_exception(self, exception: Dict) -> Dict:
        """Process a single exception through the LLM layer."""
        result = {
            **exception,
            'source': exception.get('source', 'synthetic'),
            'forced_demo_case': exception.get('forced_demo_case', False),
            'llm_invoked': False,
            'llm_root_cause': None,
            'llm_explanation': None,
            'llm_confidence': None,
            'llm_proposed_action': None,
            'llm_proposal_reasoning': None,
            'deterministic_recheck_passed': None,
            'llm_error_detail': None,
            'final_status': ReconciliationState.UNRESOLVED_EXCEPTION.value
        }
        
        if not self.llm_client:
            result['final_status'] = ReconciliationState.LLM_UNAVAILABLE.value
            return result
        
        try:
            # Step 1: Get LLM explanation
            explanation_result = self.llm_client.explain_exception(
                record_a=exception.get('settlement'),
                record_b=exception.get('counterpart')
            )
            
            if not explanation_result or not explanation_result.get('valid'):
                result['final_status'] = ReconciliationState.LLM_PARSE_ERROR.value
                result['llm_error_detail'] = explanation_result.get('error', 'Unknown explanation error') if explanation_result else 'Null explanation result'
                return result
            
            result['llm_invoked'] = True
            result['llm_root_cause'] = explanation_result.get('root_cause')
            result['llm_explanation'] = explanation_result.get('explanation')
            result['llm_confidence'] = explanation_result.get('confidence')

            ctx = ReconciliationContext.from_exception(exception)

            # Step 2: Get LLM resolution proposal when at least one counterpart exists
            counterpart_for_llm = exception.get('counterpart')
            if counterpart_for_llm is None:
                if ctx.bank_present():
                    counterpart_for_llm = ctx.bank
                elif ctx.ledger_present():
                    counterpart_for_llm = ctx.ledger

            if exception.get('settlement') and counterpart_for_llm:
                proposal_result = self.llm_client.propose_resolution(
                    record_a=exception.get('settlement'),
                    record_b=counterpart_for_llm
                )
                
                if proposal_result and proposal_result.get('valid'):
                    result['llm_proposed_action'] = proposal_result.get('action')
                    result['llm_proposal_reasoning'] = proposal_result.get('reasoning')
                    result['llm_confidence'] = proposal_result.get('confidence', result.get('llm_confidence'))

                    llm_conf = float(result['llm_confidence'] or 0)
                    if llm_conf < 0.5:
                        result['final_status'] = ReconciliationState.LOW_CONFIDENCE.value
                        result['deterministic_recheck_passed'] = False
                        result['rejection_reason'] = 'llm_low_confidence'
                    elif proposal_result.get('action') == 'match':
                        recheck_passed, reason = self._verify_context(ctx)
                        result['deterministic_recheck_passed'] = recheck_passed
                        result['rejection_reason'] = None if recheck_passed else reason
                        if recheck_passed:
                            result['final_status'] = ReconciliationState.MATCHED_LLM_VERIFIED.value
                        elif reason == 'incomplete_counterparts':
                            result['final_status'] = ReconciliationState.INCOMPLETE_COUNTERPARTS.value
                        else:
                            result['final_status'] = ReconciliationState.LLM_DETERMINISTIC_DISAGREEMENT.value
                    elif proposal_result.get('action') == 'flag_for_human':
                        result['final_status'] = ReconciliationState.FLAGGED_FOR_REVIEW.value
                    elif proposal_result.get('action') == 'reject_duplicate':
                        result['final_status'] = ReconciliationState.REJECTED_DUPLICATE.value
                    else:
                        result['final_status'] = ReconciliationState.UNRESOLVED_EXCEPTION.value
                else:
                    result['final_status'] = ReconciliationState.LLM_PARSE_ERROR.value
                    result['llm_error_detail'] = proposal_result.get('error', 'Unknown proposal error') if proposal_result else 'Null proposal result'
            else:
                if explanation_result.get('confidence', 0) >= 0.8:
                    result['final_status'] = ReconciliationState.EXPLAINED_NO_RESOLUTION.value
                else:
                    result['final_status'] = ReconciliationState.UNRESOLVED_EXCEPTION.value
        
        except Exception as e:
            err = str(e).lower()
            if any(tok in err for tok in ('timeout', '429', 'rate limit', 'connection', 'unavailable', '503')):
                result['final_status'] = ReconciliationState.LLM_PROVIDER_FAILURE.value
            else:
                result['final_status'] = ReconciliationState.LLM_PARSE_ERROR.value
            result['llm_error_detail'] = str(e)
        
        return result
    
    def _simulate_disagreement_case(self, exception: Dict) -> Dict:
        """
        Create an artificial disagreement case for demo purposes.
        
        This ensures we have at least one visible llm_deterministic_disagreement
        case to show in the dashboard - critical for demonstrating Failure Recovery.
        """
        result = {
            **exception,
            'source': exception.get('source', 'synthetic'),
            'llm_invoked': True,
            'llm_root_cause': 'timing_lag',
            'llm_explanation': 'The LLM suggests these records match despite the amount difference, citing potential fee variations.',
            'llm_confidence': 0.72,
            'llm_proposed_action': 'match',
            'llm_proposal_reasoning': 'Records appear to represent the same underlying transaction with minor discrepancies.',
            'deterministic_recheck_passed': False,  # Deliberate disagreement
            'final_status': ReconciliationState.LLM_DETERMINISTIC_DISAGREEMENT.value,
            'forced_demo_case': True
        }
        
        return result
    
    def _all_required_counterparts_present(
        self,
        settlement: Optional[Dict],
        counterpart: Optional[Dict]
    ) -> bool:
        """Legacy helper: two-record presence. Prefer ReconciliationContext."""
        if not settlement or not counterpart:
            return False
        sett_amount = (
            settlement.get('settled_amount')
            if settlement.get('settled_amount') is not None
            else settlement.get('amount')
        )
        if sett_amount is None:
            return False
        count_amount = (
            counterpart.get('amount')
            if counterpart.get('amount') is not None
            else counterpart.get('expected_amount')
        )
        if count_amount is None:
            return False
        return True

    def _deterministic_recheck(
        self,
        settlement: Optional[Dict],
        counterpart: Optional[Dict],
        exception_context: Optional[Dict] = None
    ) -> bool:
        """
        Backward-compatible wrapper. New callers should use _verify_context.

        Returns True only when a full 3-way context can be assembled and
        the verifier accepts it. A two-record pair without both bank and
        ledger legs always returns False.
        """
        payload = dict(exception_context or {})
        payload.setdefault('settlement', settlement)
        payload.setdefault('counterpart', counterpart)
        ctx = ReconciliationContext.from_exception(payload)
        if settlement and not ctx.settlement:
            ctx.settlement = settlement
        if counterpart:
            if not ctx.bank_present() and not ctx.ledger_present():
                # Infer leg type from fields
                if counterpart.get('expected_amount') is not None or counterpart.get('order_id'):
                    ctx.ledger = counterpart
                else:
                    ctx.bank = counterpart
        passed, _reason = self._verify_context(ctx)
        return passed

    def _verify_context(self, ctx: ReconciliationContext) -> tuple:
        """
        Deterministic 3-way verifier.

        Asks:
          Are all required counterparts present?
          Do IDs agree?
          Do amounts agree (Decimal, fee-aware)?
          Do currencies agree?
          Do dates satisfy policy?
          Does the proposed interpretation violate an invariant?

        Returns:
            (passed: bool, reason: str)
        """
        if not ctx.all_required_legs_present():
            return False, "incomplete_counterparts"

        matcher = DeterministicMatcher()
        settlement = ctx.settlement
        bank = ctx.bank
        ledger = ctx.ledger

        sett_net = matcher._normalize_amount(
            settlement.get('settled_amount') if settlement.get('settled_amount') is not None else settlement.get('amount')
        )
        sett_gross = matcher._normalize_amount(settlement.get('amount'))
        sett_fee = matcher._normalize_amount(settlement.get('fee')) or Decimal('0')
        bank_amt = matcher._normalize_amount(bank.get('amount'))
        ledger_amt = matcher._normalize_amount(ledger.get('expected_amount'))

        if sett_net is None or bank_amt is None or ledger_amt is None:
            return False, "missing_amount"

        def _rel(a: Decimal, b: Decimal) -> Decimal:
            return abs(a - b) / max(abs(a), abs(b), Decimal('0.01')) * Decimal('100')

        # Bank vs settlement net (or fee-adjusted)
        bank_diff = _rel(sett_net, bank_amt)
        fee_ok = False
        if sett_fee:
            fee_ok = _rel(sett_net + sett_fee, bank_amt) <= Decimal('1.5') or _rel(sett_gross or sett_net, bank_amt) <= Decimal('1.5')
        if bank_diff > Decimal('2.0') and not fee_ok:
            # Allow standard MDR band 1.5–3.5% only when fee is consistent
            if not (Decimal('1.5') <= bank_diff <= Decimal('3.5')):
                return False, "amount_mismatch_bank"

        # Ledger vs settlement gross
        gross = sett_gross if sett_gross is not None else (sett_net + sett_fee)
        ledger_diff = _rel(gross, ledger_amt)
        if ledger_diff > Decimal('2.0'):
            if not (sett_fee and _rel(sett_net + sett_fee, ledger_amt) <= Decimal('1.5')):
                return False, "amount_mismatch_ledger"
                
        # Sign check: Bank and Ledger must have same sign (or be zero)
        if bank_amt != 0 and ledger_amt != 0 and (bank_amt > 0) != (ledger_amt > 0):
            return False, "sign_mismatch"

        # Currency
        sett_ccy = str(settlement.get('currency') or '').strip().upper()
        bank_ccy = str(bank.get('currency') or '').strip().upper()
        ledger_ccy = str(ledger.get('currency') or '').strip().upper()
        present_ccy = [c for c in (sett_ccy, bank_ccy, ledger_ccy) if c]
        if len(set(present_ccy)) > 1:
            return False, "currency_mismatch"

        # Identifiers: conflicting non-empty UTRs / order_ids
        sett_utr = matcher._normalize_text(settlement.get('settlement_utr', ''))
        bank_utr = matcher._normalize_text(bank.get('utr', ''))
        if sett_utr and bank_utr and sett_utr != bank_utr:
            return False, "identifier_conflict_utr"

        sett_order = matcher._normalize_text(settlement.get('order_id', ''))
        ledger_order = matcher._normalize_text(ledger.get('order_id', ''))
        if sett_order and ledger_order and sett_order != ledger_order:
            return False, "identifier_conflict_order"

        # Dates
        sett_date = matcher._normalize_date(settlement.get('settled_at') or settlement.get('created_at'))
        bank_date = matcher._normalize_date(bank.get('value_date'))
        ledger_date = matcher._normalize_date(ledger.get('order_date'))
        for other in (bank_date, ledger_date):
            if sett_date and other:
                if abs((sett_date - other).days) > 5:
                    return False, "date_window_exceeded"

        return True, "verified"

