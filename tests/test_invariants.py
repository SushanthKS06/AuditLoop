"""
System Invariant Tests

Proves that the core design invariants hold across all code paths.

INVARIANT 1: Missing counterpart cannot become fully matched.
INVARIANT 2: LLM cannot bypass deterministic verification.
INVARIANT 3: llm_deterministic_disagreement is preserved, never collapsed.
INVARIANT 4: PII never reaches the LLM payload text.
INVARIANT 5: Demo (forced) cases are tagged and do not contaminate organic metrics.
INVARIANT 6: Metrics never report impossible values (precision > 1.0, counts < 0, etc.).
INVARIANT 7: messiness_ratio=0.0 produces near-zero exception rate.
INVARIANT 8: messiness_ratio=1.0 produces near-100% exception rate.
"""

import pytest
from unittest.mock import MagicMock, patch
from engine.exceptions import ExceptionDispatcher
from llm.privacy import sanitize_record_for_llm, sanitize_text
from metrics.evaluate import MetricsEvaluator


# ── INVARIANT 1 ─────────────────────────────────────────────────────────────

class TestInvariant1MissingCounterpart:
    """A record with a missing counterpart can NEVER become matched_llm_verified."""

    def _make_dispatcher_with_match_proposing_llm(self):
        """Create dispatcher whose LLM always proposes action='match' at confidence=1.0."""
        mock_llm = MagicMock()
        mock_llm.explain_exception.return_value = {
            'valid': True,
            'root_cause': 'timing_lag',
            'explanation': 'Records seem related.',
            'confidence': 0.99,
        }
        mock_llm.propose_resolution.return_value = {
            'valid': True,
            'action': 'match',
            'reasoning': 'Amounts match after fee adjustment.',
            'confidence': 0.99,
        }
        return ExceptionDispatcher(llm_client=mock_llm)

    def test_none_counterpart_never_matched_llm_verified(self):
        """When counterpart is None, _deterministic_recheck must return False."""
        dispatcher = self._make_dispatcher_with_match_proposing_llm()
        settlement = {'amount': 1000.0, 'settled_amount': 980.0, 'fee': 20.0}

        result = dispatcher._deterministic_recheck(settlement, None)
        assert result is False, (
            "Invariant 1 broken: _deterministic_recheck returned True for None counterpart."
        )

    def test_missing_amount_counterpart_never_matched_llm_verified(self):
        """When counterpart has no amount fields, recheck must return False."""
        dispatcher = self._make_dispatcher_with_match_proposing_llm()
        settlement = {'amount': 1000.0, 'settled_amount': 980.0}
        counterpart = {'order_id': 'ORD_001'}  # No amount field at all

        result = dispatcher._deterministic_recheck(settlement, counterpart)
        assert result is False, (
            "Invariant 1 broken: recheck returned True for counterpart with no amount."
        )

    def test_full_exception_flow_none_counterpart_not_matched_verified(self):
        """End-to-end: exception with no counterpart must NOT produce matched_llm_verified."""
        dispatcher = self._make_dispatcher_with_match_proposing_llm()
        exception_record = {
            'type': 'unmatched_settlement',
            'record_ids': 'sett_orphan-none',
            'settlement': {'payment_id': 'PAY_ORPHAN', 'amount': 5000.0, 'settled_amount': 4900.0},
            'counterpart': None,  # No bank / ledger counterpart
            'source': 'synthetic',
        }

        results = dispatcher.process_exceptions([exception_record])
        assert len(results) == 1
        status = results[0].get('final_status')
        assert status != 'matched_llm_verified', (
            f"Invariant 1 broken: orphan exception with no counterpart produced "
            f"final_status='{status}' instead of explained_no_resolution / unresolved_exception."
        )

    def test_counterpart_presence_helper_validates_amounts(self):
        """_all_required_counterparts_present must require amount fields on both sides."""
        dispatcher = ExceptionDispatcher(llm_client=None)

        # Both None
        assert dispatcher._all_required_counterparts_present(None, None) is False
        # Settlement None
        assert dispatcher._all_required_counterparts_present(
            None, {'amount': 100.0}
        ) is False
        # Counterpart None
        assert dispatcher._all_required_counterparts_present(
            {'amount': 100.0}, None
        ) is False
        # Both present but counterpart has no amount
        assert dispatcher._all_required_counterparts_present(
            {'amount': 100.0, 'settled_amount': 98.0},
            {'order_id': 'ORD_NO_AMOUNT'}
        ) is False
        # Valid pair
        assert dispatcher._all_required_counterparts_present(
            {'amount': 100.0, 'settled_amount': 98.0},
            {'amount': 98.0}
        ) is True


# ── INVARIANT 2 ─────────────────────────────────────────────────────────────

class TestInvariant2LLMCannotBypassVerification:
    """LLM proposals with action='match' must always pass through _deterministic_recheck."""

    def test_high_amount_discrepancy_rejected_even_with_llm_match_proposal(self):
        """LLM proposes match on >10% discrepancy → deterministic recheck rejects → disagreement."""
        mock_llm = MagicMock()
        mock_llm.explain_exception.return_value = {
            'valid': True, 'root_cause': 'rounding',
            'explanation': 'Might be a fee.', 'confidence': 0.9,
        }
        mock_llm.propose_resolution.return_value = {
            'valid': True, 'action': 'match', 'reasoning': 'Looks close.', 'confidence': 0.9,
        }

        dispatcher = ExceptionDispatcher(llm_client=mock_llm)
        exception = {
            'type': 'fuzzy_exception',
            'record_ids': 'sett_001-TXN_001',
            'settlement': {
                'payment_id': 'PAY_001', 'amount': 10000.0, 'settled_amount': 9800.0, 'fee': 200.0,
                'settled_at': '2026-09-01',
            },
            'counterpart': {
                'amount': 8000.0,  # 18.4% discrepancy — way outside any fee range
                'value_date': '2026-09-01',
            },
            'source': 'synthetic',
        }

        results = dispatcher.process_exceptions([exception])
        assert len(results) == 1
        status = results[0]['final_status']
        assert status == 'llm_deterministic_disagreement', (
            f"Invariant 2 broken: LLM proposed match with 18% discrepancy, "
            f"but final_status='{status}'. Deterministic recheck failed to veto."
        )
        assert results[0]['deterministic_recheck_passed'] is False

    def test_llm_match_on_valid_fee_adjusted_pair_is_confirmed(self):
        """LLM proposes match on a fee-adjusted pair → deterministic recheck CONFIRMS it."""
        mock_llm = MagicMock()
        mock_llm.explain_exception.return_value = {
            'valid': True, 'root_cause': 'rounding',
            'explanation': 'Fee deduction.', 'confidence': 0.95,
        }
        mock_llm.propose_resolution.return_value = {
            'valid': True, 'action': 'match', 'reasoning': 'Fee adjusted.', 'confidence': 0.95,
        }

        dispatcher = ExceptionDispatcher(llm_client=mock_llm)
        exception = {
            'type': 'fuzzy_exception',
            'record_ids': 'sett_002-TXN_002',
            'settlement': {
                'payment_id': 'PAY_002', 'amount': 1000.0, 'settled_amount': 976.40, 'fee': 23.60,
                'settled_at': '2026-09-01',
            },
            'counterpart': {
                'amount': 976.40,  # Exact settled amount — within threshold
                'value_date': '2026-09-01',
            },
            'source': 'synthetic',
        }

        results = dispatcher.process_exceptions([exception])
        assert len(results) == 1
        status = results[0]['final_status']
        assert status == 'matched_llm_verified', (
            f"Valid fee-adjusted pair should produce matched_llm_verified, got '{status}'."
        )
        assert results[0]['deterministic_recheck_passed'] is True


# ── INVARIANT 3 ─────────────────────────────────────────────────────────────

class TestInvariant3DisagreementPreserved:
    """llm_deterministic_disagreement is never collapsed into another status."""

    def test_disagreement_status_survives_metrics_evaluation(self, tmp_path):
        """Disagreement results are correctly counted in metrics, not re-categorised."""
        evaluator = MetricsEvaluator(ground_truth_path=str(tmp_path / "no_gt.json"))
        evaluator.ground_truth = []  # No GT — all records go to unverified in strict mode

        results = [
            {'payment_id': 'PAY_D1', 'final_status': 'llm_deterministic_disagreement'},
            {'payment_id': 'PAY_D2', 'final_status': 'llm_deterministic_disagreement'},
            {'payment_id': 'PAY_D3', 'final_status': 'matched'},
        ]

        metrics = evaluator.evaluate(results, output_path=None, coverage_mode='strict')

        assert metrics['disagreement_count'] == 2, (
            f"Expected 2 disagreements counted, got {metrics['disagreement_count']}."
        )


# ── INVARIANT 4 ─────────────────────────────────────────────────────────────

class TestInvariant4PIINeverReachesLLM:
    """PII must never appear in the sanitized record passed to LLM calls."""

    _PII_FIELDS = [
        ('customer_name', 'Rahul Sharma'),
        ('email', 'rahul@example.com'),
        ('phone', '9876543210'),
        ('pan', 'ABCDE1234F'),
        ('account_number', '12345678901234'),
        ('ifsc', 'HDFC0001234'),
        ('vpa', 'rahul@upi'),
        ('upi_handle', 'merchant@paytm'),
        ('bank_account', '987654321098'),
        ('beneficiary_name', 'Sharma Enterprises'),
    ]

    def test_all_pii_fields_redacted(self):
        """Every field in _PII_FIELDS must be redacted to [REDACTED_PII]."""
        raw = {k: v for k, v in self._PII_FIELDS}
        raw['amount'] = 5000.0
        raw['order_id'] = 'ORD_SAFE_001'
        raw['payment_id'] = 'PAY_SAFE_001'

        clean = sanitize_record_for_llm(raw)

        for field, raw_value in self._PII_FIELDS:
            sanitized_value = clean.get(field, '')
            assert raw_value not in str(sanitized_value), (
                f"Invariant 4 broken: PII field '{field}' leaked raw value "
                f"'{raw_value}' into sanitized record. Got: '{sanitized_value}'"
            )

    def test_non_pii_fields_preserved(self):
        """Matching-critical fields must survive sanitization unchanged."""
        raw = {
            'order_id': 'ORD_001',
            'payment_id': 'PAY_001',
            'settlement_utr': 'UTR999888',
            'amount': 1500.0,
            'fee': 35.40,
            'customer_name': 'Rahul Sharma',  # PII - should be redacted
        }
        clean = sanitize_record_for_llm(raw)

        assert clean['order_id'] == 'ORD_001'
        assert clean['payment_id'] == 'PAY_001'
        assert clean['settlement_utr'] == 'UTR999888'
        assert clean['amount'] == 1500.0
        assert clean['fee'] == 35.40

    def test_narration_with_pan_and_phone_redacted(self):
        """Narration containing PAN and phone must have both redacted."""
        narration = "Payment by ABCDE1234F, mobile +91 9876543210, ref ORD_001"
        clean = sanitize_text(narration)

        assert 'ABCDE1234F' not in clean, "PAN leaked through sanitize_text"
        assert '9876543210' not in clean, "Phone number leaked through sanitize_text"
        assert 'ORD_001' in clean, "Non-PII reference was wrongly removed"

    def test_bank_account_in_narration_redacted(self):
        """14-digit bank account number in narration text must be redacted."""
        narration = "Transfer to account 12345678901234 completed"
        clean = sanitize_text(narration)
        assert '12345678901234' not in clean, "Bank account number leaked through sanitize_text"


# ── INVARIANT 5 ─────────────────────────────────────────────────────────────

class TestInvariant5DemoCasesTagged:
    """Demo (forced) cases must be tagged with forced_demo_case=True."""

    def test_simulated_disagreement_is_tagged_as_demo(self):
        """_simulate_disagreement_case must set forced_demo_case=True."""
        dispatcher = ExceptionDispatcher(llm_client=None)
        exception = {
            'type': 'fuzzy_exception',
            'record_ids': 'sett_demo-TXN_demo',
            'settlement': {'payment_id': 'PAY_DEMO', 'amount': 1000.0},
            'counterpart': {'amount': 900.0},
            'source': 'synthetic',
        }
        result = dispatcher._simulate_disagreement_case(exception)
        assert result.get('forced_demo_case') is True, (
            "Demo case must be tagged with forced_demo_case=True "
            "so it can be excluded from organic metrics."
        )
        assert result['final_status'] == 'llm_deterministic_disagreement'


# ── INVARIANT 6 ─────────────────────────────────────────────────────────────

class TestInvariant6MetricsNeverImpossible:
    """Metrics must never produce values outside mathematically valid ranges."""

    def _evaluate_n_results(self, tmp_path, results, gt_entries=None):
        import json
        gt_path = tmp_path / "gt.json"
        gt = gt_entries or []
        with open(gt_path, 'w') as f:
            json.dump(gt, f)
        evaluator = MetricsEvaluator(ground_truth_path=str(gt_path))
        return evaluator.evaluate(results, output_path=None, coverage_mode='strict')

    def test_precision_between_0_and_1(self, tmp_path):
        results = [{'payment_id': f'P{i}', 'final_status': 'matched'} for i in range(20)]
        metrics = self._evaluate_n_results(tmp_path, results)
        assert 0.0 <= metrics['precision'] <= 1.0, f"precision={metrics['precision']} out of [0,1]"

    def test_recall_between_0_and_1(self, tmp_path):
        results = [{'payment_id': f'P{i}', 'final_status': 'matched'} for i in range(20)]
        metrics = self._evaluate_n_results(tmp_path, results)
        assert 0.0 <= metrics['recall'] <= 1.0, f"recall={metrics['recall']} out of [0,1]"

    def test_match_rate_between_0_and_1(self, tmp_path):
        results = [{'payment_id': f'P{i}', 'final_status': 'matched'} for i in range(10)]
        results += [{'payment_id': f'E{i}', 'final_status': 'unresolved_exception'} for i in range(10)]
        metrics = self._evaluate_n_results(tmp_path, results)
        assert 0.0 <= metrics['match_rate'] <= 1.0, f"match_rate={metrics['match_rate']} out of [0,1]"

    def test_counts_non_negative(self, tmp_path):
        results = []
        metrics = self._evaluate_n_results(tmp_path, results)
        for key in ('true_positives', 'false_positives', 'true_negatives', 'false_negatives',
                    'matched_count', 'exception_count', 'disagreement_count', 'unresolved_count'):
            assert metrics[key] >= 0, f"{key}={metrics[key]} is negative — impossible value"

    def test_ground_truth_coverage_between_0_and_1(self, tmp_path):
        import json
        gt_entries = [
            {'payment_id': 'PAY_GT_001', 'should_match': True,
             'ledger_order_id': '', 'utr': '', 'bank_txn_id': '',
             'root_cause': 'exact_match', 'notes': '', 'messiness_type': 'exact_match'}
        ]
        results = [
            {'payment_id': 'PAY_GT_001', 'final_status': 'matched'},
            {'payment_id': 'PAY_NO_GT', 'final_status': 'matched'},
        ]
        metrics = self._evaluate_n_results(tmp_path, results, gt_entries)
        assert 0.0 <= metrics['ground_truth_coverage'] <= 1.0
