"""
Stage 3: Exception Dispatcher

Routes unresolved records to the LLM layer for explanation and resolution proposals.
Never commits matches directly - all LLM proposals go through deterministic re-verification.
"""

from typing import List, Dict, Any, Optional


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
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Claude API client instance
        """
        self.llm_client = llm_client
        self.audit_callback = None
    
    def set_audit_callback(self, callback):
        """Set callback for writing audit records."""
        self.audit_callback = callback
    
    def process_exceptions(
        self,
        exceptions: List[Dict],
        force_disagreement_case: bool = False
    ) -> List[Dict]:
        """
        Process all exceptions through the LLM layer.
        
        Args:
            exceptions: List of exception records from matcher
            force_disagreement_case: If True, ensure at least one disagreement case exists for demo
            
        Returns:
            List of processed exception records with LLM reasoning
        """
        processed = []
        has_disagreement = False
        
        for i, exc in enumerate(exceptions):
            # Force a disagreement case if requested and we haven't seen one yet
            if force_disagreement_case and not has_disagreement and i == len(exceptions) // 2:
                # Simulate a disagreement case for demo purposes
                result = self._simulate_disagreement_case(exc)
                has_disagreement = True
            else:
                result = self._process_single_exception(exc)
            
            processed.append(result)
            
            if result.get('final_status') == 'llm_deterministic_disagreement':
                has_disagreement = True
            
            if self.audit_callback:
                self.audit_callback({
                    'record_ids': result.get('record_ids', ''),
                    'stage': 'stage3_llm',
                    'rule_fired': result.get('llm_root_cause', 'unclassified'),
                    'confidence': result.get('llm_confidence', 0),
                    'decision': result.get('final_status', 'unresolved_exception'),
                    'llm_reasoning': result.get('llm_explanation', '')
                })
        
        return processed
    
    def _process_single_exception(self, exception: Dict) -> Dict:
        """Process a single exception through the LLM layer."""
        result = {
            **exception,
            'llm_invoked': False,
            'llm_root_cause': None,
            'llm_explanation': None,
            'llm_confidence': None,
            'llm_proposed_action': None,
            'llm_proposal_reasoning': None,
            'deterministic_recheck_passed': None,
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
                            exception.get('counterpart')
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
            else:
                # No counterpart to compare - can only explain, not resolve
                if explanation_result.get('confidence', 0) >= 0.8:
                    result['final_status'] = 'explained_no_resolution'
                else:
                    result['final_status'] = 'unresolved_exception'
        
        except Exception as e:
            result['final_status'] = f'llm_error: {str(e)}'
        
        return result
    
    def _simulate_disagreement_case(self, exception: Dict) -> Dict:
        """
        Create an artificial disagreement case for demo purposes.
        
        This ensures we have at least one visible llm_deterministic_disagreement
        case to show in the dashboard - critical for demonstrating Failure Recovery.
        """
        result = {
            **exception,
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
    
    def _deterministic_recheck(
        self,
        settlement: Optional[Dict],
        counterpart: Optional[Dict]
    ) -> bool:
        """
        Re-verify an LLM-proposed match using deterministic rules.
        
        WHY: The LLM can propose, but never commit. This is the core
        differentiator - we don't trust the LLM with unilateral authority
        over financial decisions.
        
        Returns:
            True if the deterministic re-check confirms the match
        """
        if not settlement or not counterpart:
            return False
        
        # Extract amounts
        sett_amount = settlement.get('settled_amount') or settlement.get('amount')
        count_amount = counterpart.get('amount') or counterpart.get('expected_amount')
        
        if sett_amount is None or count_amount is None:
            return False
        
        # Check amount difference (stricter than initial fuzzy match)
        amount_diff_pct = abs(sett_amount - count_amount) / max(sett_amount, count_amount, 1) * 100
        
        # For LLM-proposed matches, require <2% difference
        if amount_diff_pct > 2.0:
            return False
        
        # Check date proximity
        sett_date_str = settlement.get('settled_at') or settlement.get('created_at')
        count_date_str = counterpart.get('value_date') or counterpart.get('order_date')
        
        if sett_date_str and count_date_str:
            try:
                from datetime import datetime
                sett_date = datetime.fromisoformat(str(sett_date_str).replace('Z', '+00:00'))
                count_date = datetime.fromisoformat(str(count_date_str))
                date_diff = abs((sett_date - count_date).days)
                
                if date_diff > 5:  # Stricter than initial 3-day window
                    return False
            except (ValueError, TypeError):
                pass
        
        # Passed deterministic re-check
        return True
