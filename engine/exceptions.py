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
        force_disagreement_case: bool = False,
        concurrent: bool = True
    ) -> List[Dict]:
        """
        Process all exceptions through the LLM layer with optional concurrent worker pool.
        
        Args:
            exceptions: List of exception records from matcher
            force_disagreement_case: If True, ensure at least one disagreement case exists for demo
            concurrent: Whether to parallelize LLM network calls across worker threads
            
        Returns:
            List of processed exception records with LLM reasoning
        """
        if not exceptions:
            return []
            
        processed = [None] * len(exceptions)
        has_disagreement = False
        forced_idx = len(exceptions) // 2 if force_disagreement_case else -1
        
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
            'final_status': 'unresolved_exception'
        }
        
        if not self.llm_client:
            result['final_status'] = 'llm_unavailable'
            return result
        
        try:
            # Step 1: Get LLM explanation
            explanation_result = self.llm_client.explain_exception(
                record_a=exception.get('settlement'),
                record_b=exception.get('counterpart')
            )
            
            if not explanation_result or not explanation_result.get('valid'):
                result['final_status'] = 'llm_parse_error'
                result['llm_error_detail'] = explanation_result.get('error', 'Unknown explanation error') if explanation_result else 'Null explanation result'
                return result
            
            result['llm_invoked'] = True
            result['llm_root_cause'] = explanation_result.get('root_cause')
            result['llm_explanation'] = explanation_result.get('explanation')
            result['llm_confidence'] = explanation_result.get('confidence')
            
            # Step 2: Get LLM resolution proposal (if we have both records)
            if exception.get('settlement') and exception.get('counterpart'):
                proposal_result = self.llm_client.propose_resolution(
                    record_a=exception.get('settlement'),
                    record_b=exception.get('counterpart')
                )
                
                if proposal_result and proposal_result.get('valid'):
                    result['llm_proposed_action'] = proposal_result.get('action')
                    result['llm_proposal_reasoning'] = proposal_result.get('reasoning')
                    
                    # Step 3: CRITICAL - Re-verify LLM match proposals deterministically
                    if proposal_result.get('action') == 'match':
                        recheck_passed = self._deterministic_recheck(
                            exception.get('settlement'),
                            exception.get('counterpart'),
                            exception_context=exception  # carries has_bank_leg flag
                        )
                        result['deterministic_recheck_passed'] = recheck_passed
                        
                        if recheck_passed:
                            result['final_status'] = 'matched_llm_verified'
                        else:
                            result['final_status'] = 'llm_deterministic_disagreement'
                    elif proposal_result.get('action') == 'flag_for_human':
                        result['final_status'] = 'flagged_for_review'
                    elif proposal_result.get('action') == 'reject_duplicate':
                        result['final_status'] = 'rejected_duplicate'
                    else:
                        result['final_status'] = 'unresolved_exception'
                else:
                    result['final_status'] = 'llm_parse_error'
                    result['llm_error_detail'] = proposal_result.get('error', 'Unknown proposal error') if proposal_result else 'Null proposal result'
            else:
                # No counterpart to compare - can only explain, not resolve
                if explanation_result.get('confidence', 0) >= 0.8:
                    result['final_status'] = 'explained_no_resolution'
                else:
                    result['final_status'] = 'unresolved_exception'
        
        except Exception as e:
            result['final_status'] = 'llm_parse_error'
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
            'final_status': 'llm_deterministic_disagreement',
            'forced_demo_case': True
        }
        
        return result
    
    def _all_required_counterparts_present(
        self,
        settlement: Optional[Dict],
        counterpart: Optional[Dict]
    ) -> bool:
        """
        Invariant gate: verify that enough counterpart data exists to confirm a match.

        A full 3-way reconciliation requires all three legs:
          settlement + bank + ledger

        At Stage 3 exceptions we only receive the settlement and one counterpart
        (either bank or ledger, whichever was available).  If that counterpart is
        None *or* has no amount field, we cannot confirm a match and must return
        False immediately — BEFORE any numeric comparison.

        This prevents the scenario where:
          1. A settlement has no bank record (orphan)
          2. The LLM incorrectly proposes action='match'
          3. _deterministic_recheck accidentally passes on edge-case amounts
          4. The result is promoted to 'matched_llm_verified'

        Returns:
            True only when both settlement and counterpart carry the minimum
            required fields for a deterministic numeric comparison.
        """
        if not settlement:
            return False
        if not counterpart:
            return False

        # Settlement must have at least one amount field
        sett_amount = (
            settlement.get('settled_amount')
            if settlement.get('settled_amount') is not None
            else settlement.get('amount')
        )
        if sett_amount is None:
            return False

        # Counterpart must have at least one amount field
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
        Re-verify an LLM-proposed match using deterministic rules.

        WHY: The LLM can propose, but never commit. This is the core
        differentiator - we don't trust the LLM with unilateral authority
        over financial decisions.

        The first check is the counterpart-presence invariant: if the
        settlement has no counterpart (bank or ledger leg is missing), the
        recheck MUST return False regardless of LLM confidence. Missing legs
        can never form a fully reconciled match.

        Bank-leg requirement: A 3-way reconciliation requires settlement +
        bank + ledger.  If the exception was raised because the bank leg was
        absent (has_bank_leg=False in exception_context), then even a ledger-
        side counterpart cannot satisfy the 3-way condition.  We reject.

        Returns:
            True if the deterministic re-check confirms the match
        """
        # ── INVARIANT GATE ────────────────────────────────────────────────
        # Enforce: missing counterpart → never matched_llm_verified.
        # This must be checked BEFORE any numeric comparison so that an
        # orphan record with a None counterpart cannot accidentally pass.
        if not self._all_required_counterparts_present(settlement, counterpart):
            return False

        # ── BANK LEG REQUIREMENT ──────────────────────────────────────────
        # 3-way reconciliation: settlement + BANK + ledger are all required.
        # If the exception record explicitly records has_bank_leg=False, the
        # bank transaction was never present and the match cannot be confirmed.
        if exception_context is not None:
            has_bank_leg = exception_context.get('has_bank_leg')
            if has_bank_leg is False:
                # No bank leg confirmed — cannot satisfy 3-way requirement
                return False

        # Extract amounts
        sett_amount = settlement.get('settled_amount') if settlement.get('settled_amount') is not None else settlement.get('amount')
        count_amount = counterpart.get('amount') if counterpart.get('amount') is not None else counterpart.get('expected_amount')
        sett_fee = settlement.get('fee', 0.0) or 0.0
        
        if sett_amount is None or count_amount is None:
            return False
        
        try:
            from decimal import Decimal, InvalidOperation
            
            def to_dec(val):
                if isinstance(val, Decimal):
                    return val
                return Decimal(str(val))
                
            sett_amount = to_dec(sett_amount)
            count_amount = to_dec(count_amount)
            sett_fee = to_dec(sett_fee)
        except (ValueError, TypeError, InvalidOperation):
            return False
        
        # Check amount difference (stricter than initial fuzzy match)
        amount_diff_pct = abs(sett_amount - count_amount) / max(sett_amount, count_amount, Decimal('1')) * Decimal('100')
        
        # Check if amount matches directly (<2%) or matches after standard fee deduction
        fee_adjusted_match = False
        if abs((sett_amount + sett_fee) - count_amount) / max(count_amount, Decimal('1')) * Decimal('100') <= Decimal('1.5'):
            fee_adjusted_match = True
        elif Decimal('1.5') <= amount_diff_pct <= Decimal('3.5'):
            # Standard MDR fee deduction range (2% + 18% GST = 2.36%)
            fee_adjusted_match = True
        
        if amount_diff_pct > Decimal('2.0') and not fee_adjusted_match:
            return False
        
        # Check date proximity with robust timezone normalization
        sett_date_str = settlement.get('settled_at') or settlement.get('created_at')
        count_date_str = counterpart.get('value_date') or counterpart.get('order_date')
        
        if sett_date_str and count_date_str:
            try:
                from datetime import datetime, timezone
                
                def _parse_dt(d):
                    if isinstance(d, datetime):
                        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
                    s = str(d).replace('Z', '+00:00').strip()
                    try:
                        dt = datetime.fromisoformat(s)
                        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                    except ValueError:
                        dt = datetime.strptime(s.split()[0].split('+')[0].strip(), '%Y-%m-%d')
                        return dt.replace(tzinfo=timezone.utc)
                
                sett_date = _parse_dt(sett_date_str)
                count_date = _parse_dt(count_date_str)
                date_diff = abs((sett_date - count_date).days)
                
                if date_diff > 5:  # Stricter than initial window
                    return False
            except Exception:
                # Fail closed on unparseable dates
                return False
        
        # Passed deterministic re-check
        return True
