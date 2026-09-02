"""
Regression and reliability tests for Exception Dispatcher and Exception Record IDs.

Verifies:
1. All exception types (low_confidence, unmatched_settlement, unmatched_bank, unmatched_ledger) have non-empty record_ids.
2. Parallel processing of >= 5 exceptions via ThreadPoolExecutor completes reliably without llm_parse_error.
3. Errors from LLM failure are captured cleanly in llm_error_detail instead of being swallowed.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock

from engine.matcher import DeterministicMatcher
from engine.exceptions import ExceptionDispatcher


class TestExceptionRecordIDs:
    """Test that all exception categories preserve traceability through record_ids."""
    
    def test_all_exception_types_have_non_empty_record_ids(self):
        matcher = DeterministicMatcher()
        
        # 1. Low confidence
        low_conf = pd.DataFrame([{
            'settlement': {'entity_id': 'sett_101', 'amount': 1000.0},
            'bank': {'txn_id': 'TXN_202', 'amount': 950.0},
            'confidence': 0.72,
            'rule_fired': 'weak_candidate'
        }])
        
        # 2. Unmatched settlement
        unmatched_sett = pd.DataFrame([{
            'entity_id': 'sett_303',
            'amount': 2500.0,
            'payment_id': 'pay_303'
        }])
        
        # 3. Unmatched bank
        unmatched_bank = pd.DataFrame([{
            'txn_id': 'TXN_404',
            'amount': 1400.0,
            'utr': 'UTR_404'
        }])
        
        # 4. Unmatched ledger
        unmatched_ledger = pd.DataFrame([{
            'order_id': 'ORD_505',
            'expected_amount': 3200.0,
            'payment_id': 'PAY_505'
        }])
        
        exceptions = matcher.get_exceptions(
            low_confidence=low_conf,
            unmatched_settlements=unmatched_sett,
            unmatched_bank=unmatched_bank,
            unmatched_ledger=unmatched_ledger
        )
        
        assert len(exceptions) == 4
        for exc in exceptions:
            assert 'record_ids' in exc, f"Missing record_ids in exception type: {exc.get('type')}"
            assert isinstance(exc['record_ids'], str)
            assert len(exc['record_ids'].strip()) > 0, f"Empty record_ids in exception type: {exc.get('type')}"
            
        # Specific assertions
        assert exceptions[0]['record_ids'] == "sett_101-TXN_202"
        assert exceptions[1]['record_ids'] == "sett_303"
        assert exceptions[2]['record_ids'] == "TXN_404"
        assert exceptions[3]['record_ids'] == "ORD_505"


class TestExceptionDispatcherReliability:
    """Test concurrent processing and error retention in ExceptionDispatcher."""
    
    def test_concurrent_dispatch_healthy_client(self):
        """Test concurrent processing of >= 5 exceptions with a healthy mock client."""
        mock_llm = MagicMock()
        mock_llm.explain_exception.return_value = {
            "valid": True,
            "thought_process": "Amount matches after fee deduction.",
            "root_cause": "rounding",
            "explanation": "2% MDR fee accounts for the difference.",
            "confidence": 0.94
        }
        mock_llm.propose_resolution.return_value = {
            "valid": True,
            "thought_process": "Verify date and amount delta within limits.",
            "action": "match",
            "confidence": 0.92,
            "reasoning": "Standard MDR fee and date alignment."
        }
        
        dispatcher = ExceptionDispatcher(llm_client=mock_llm, max_workers=2)
        
        test_exceptions = [
            {
                'type': 'low_confidence',
                'record_ids': f'sett_{i:03d}-TXN_{i:03d}',
                'settlement': {'entity_id': f'sett_{i:03d}', 'amount': 1000.0, 'settled_amount': 976.40, 'fee': 23.60},
                'counterpart': {'txn_id': f'TXN_{i:03d}', 'amount': 1000.0},
                'confidence': 0.75,
                'rule_fired': 'weak_candidate'
            }
            for i in range(6)
        ]
        
        results = dispatcher.process_exceptions(test_exceptions, concurrent=True)
        
        assert len(results) == 6
        for res in results:
            assert res['final_status'] != 'llm_parse_error'
            assert res['llm_root_cause'] == 'rounding'
            assert res['llm_confidence'] == 0.94
            assert res['llm_explanation'] is not None
            assert res['record_ids'].startswith('sett_')
            assert res['final_status'] == 'matched_llm_verified'

    def test_error_detail_captured_on_invalid_response(self):
        """Test that validation/network failures populate llm_error_detail rather than swallowing."""
        mock_llm = MagicMock()
        mock_llm.explain_exception.return_value = {
            "valid": False,
            "error": "Rate limit 429: Too Many Requests on Groq RPM"
        }
        
        audit_records = []
        dispatcher = ExceptionDispatcher(llm_client=mock_llm)
        dispatcher.set_audit_callback(lambda r: audit_records.append(r))
        
        test_exception = {
            'type': 'unmatched_bank',
            'record_ids': 'TXN_999',
            'settlement': None,
            'counterpart': {'txn_id': 'TXN_999', 'amount': 5000.0},
            'confidence': 0.0,
            'rule_fired': 'no_candidate_found'
        }
        
        results = dispatcher.process_exceptions([test_exception], concurrent=False)
        
        assert len(results) == 1
        res = results[0]
        assert res['final_status'] == 'llm_parse_error'
        assert res['llm_error_detail'] == "Rate limit 429: Too Many Requests on Groq RPM"
        
        assert len(audit_records) == 1
        assert audit_records[0]['record_ids'] == 'TXN_999'
        assert "429" in audit_records[0]['llm_reasoning']
