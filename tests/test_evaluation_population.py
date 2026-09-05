"""Transaction evaluation population must not be inflated by orphan events."""

from engine.states import ReconciliationState
from metrics.evaluate import MetricsEvaluator


class TestEvaluationPopulation:
    def test_20_settlements_plus_2_orphans(self, tmp_path):
        gt = [
            {
                'payment_id': f'PAY_{i:03d}',
                'ledger_order_id': f'ORD_{i:03d}',
                'utr': f'UTR{i:06d}',
                'bank_txn_id': f'TXN_{i:06d}',
                'should_match': True,
                'root_cause': 'exact_match',
                'notes': '',
                'messiness_type': 'exact_match',
            }
            for i in range(20)
        ]
        gt_path = tmp_path / 'ground_truth.json'
        gt_path.write_text(__import__('json').dumps(gt), encoding='utf-8')
        evaluator = MetricsEvaluator(ground_truth_path=str(gt_path))

        results = [
            {
                'payment_id': f'PAY_{i:03d}',
                'entity_id': f'sett_{i:04d}',
                'type': 'settlement',
                'final_status': ReconciliationState.EXACT_MATCH.value,
            }
            for i in range(20)
        ]
        results.append({
            'type': ReconciliationState.UNMATCHED_BANK.value,
            'txn_id': 'TXN_ORPHAN_BANK',
            'final_status': ReconciliationState.UNMATCHED_BANK.value,
        })
        results.append({
            'type': ReconciliationState.UNMATCHED_LEDGER.value,
            'order_id': 'ORD_ORPHAN_LEDGER',
            'final_status': ReconciliationState.UNMATCHED_LEDGER.value,
        })

        input_ids = [f'sett_{i:04d}' for i in range(20)]
        metrics = evaluator.evaluate(
            results,
            output_path=None,
            input_transaction_ids=input_ids,
            input_transaction_count=20,
        )

        assert metrics['total_input_transactions'] == 20
        assert metrics['orphan_bank_records'] == 1
        assert metrics['orphan_ledger_records'] == 1
        assert metrics['true_positives'] == 20
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['impossible_state'] is None

    def test_demo_injection_does_not_change_organic_tp(self, tmp_path):
        gt = [
            {'payment_id': 'PAY_1', 'should_match': True},
            {'payment_id': 'PAY_2', 'should_match': True},
        ]
        gt_path = tmp_path / 'gt.json'
        gt_path.write_text(__import__('json').dumps(gt), encoding='utf-8')
        evaluator = MetricsEvaluator(ground_truth_path=str(gt_path))
        organic = [
            {'payment_id': 'PAY_1', 'type': 'settlement', 'final_status': ReconciliationState.EXACT_MATCH.value},
            {'payment_id': 'PAY_2', 'type': 'settlement', 'final_status': ReconciliationState.EXACT_MATCH.value},
        ]
        with_demo = organic + [{
            'payment_id': 'PAY_DEMO',
            'type': 'settlement',
            'final_status': ReconciliationState.LLM_DETERMINISTIC_DISAGREEMENT.value,
            'forced_demo_case': True,
        }]
        m1 = evaluator.evaluate(organic, output_path=None, input_transaction_count=2)
        m2 = evaluator.evaluate(with_demo, output_path=None, input_transaction_count=2)
        assert m1['true_positives'] == m2['true_positives'] == 2
        assert m1['precision'] == m2['precision']
        assert m2['demo_injected_count'] == 1
        assert m1['demo_injected_count'] == 0
