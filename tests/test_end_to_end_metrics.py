"""
End-to-end metrics test.

Verifies that metrics can be regenerated correctly on a fixed seed.
This proves the pipeline is deterministic and not hardcoded.

Also verifies that unlabeled records are excluded from precision/recall/F1
and counted in ``unverified_count``.
"""

import pytest
import os
import json
import tempfile
from pathlib import Path

from data.generate_data import SyntheticDataGenerator
from metrics.evaluate import MetricsEvaluator
from engine.states import ReconciliationState


class TestEndToEndMetrics:
    """Test that metrics are reproducible with fixed seeds."""

    def test_reproducible_generation(self, tmp_path):
        """Test that same seed produces same ground truth."""
        gen1 = SyntheticDataGenerator(seed=42, messiness_ratio=0.25)
        gen1.generate(num_records=20, settlements_df=None, output_dir=str(tmp_path / "run1"))

        gen2 = SyntheticDataGenerator(seed=42, messiness_ratio=0.25)
        gen2.generate(num_records=20, settlements_df=None, output_dir=str(tmp_path / "run2"))

        with open(tmp_path / "run1" / "ground_truth.json") as f:
            gt1 = json.load(f)

        with open(tmp_path / "run2" / "ground_truth.json") as f:
            gt2 = json.load(f)

        assert len(gt1) == len(gt2)
        for i in range(len(gt1)):
            assert gt1[i]['messiness_type'] == gt2[i]['messiness_type']
            assert gt1[i]['should_match'] == gt2[i]['should_match']

    def test_different_seeds_produce_different_data(self, tmp_path):
        """Test that different seeds produce different ground truth."""
        gen1 = SyntheticDataGenerator(seed=42, messiness_ratio=0.25)
        gen1.generate(num_records=20, settlements_df=None, output_dir=str(tmp_path / "seed42"))

        gen2 = SyntheticDataGenerator(seed=123, messiness_ratio=0.25)
        gen2.generate(num_records=20, settlements_df=None, output_dir=str(tmp_path / "seed123"))

        with open(tmp_path / "seed42" / "ground_truth.json") as f:
            gt1 = json.load(f)

        with open(tmp_path / "seed123" / "ground_truth.json") as f:
            gt2 = json.load(f)

        differences = 0
        for i in range(min(len(gt1), len(gt2))):
            if gt1[i]['messiness_type'] != gt2[i]['messiness_type']:
                differences += 1

        assert differences > 0

    def test_metrics_evaluation_basic(self, tmp_path):
        """Test basic metrics evaluation."""
        gen = SyntheticDataGenerator(seed=42, messiness_ratio=0.25)
        gen.generate(num_records=20, settlements_df=None, output_dir=str(tmp_path))

        ground_truth = gen.ground_truth

        mock_results = []
        for gt in ground_truth:
            mock_results.append({
                'payment_id': gt['payment_id'],
                'type': 'settlement',
                'final_status': ReconciliationState.EXACT_MATCH.value if gt['should_match'] else ReconciliationState.UNRESOLVED_EXCEPTION.value
            })

        evaluator = MetricsEvaluator(ground_truth_path=str(tmp_path / "ground_truth.json"))
        metrics = evaluator.evaluate(mock_results, output_path=None)

        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'match_rate' in metrics
        assert metrics['total_input_transactions'] == 20

        assert metrics['precision'] >= 0.8
        assert metrics['recall'] >= 0.8


class TestMetricsEvaluation:
    """
    Acceptance tests for Metrics Evaluator
    """

    def _build_evaluator_with_gt(self, tmp_path, gt_entries):
        """Write a ground_truth.json and return a configured MetricsEvaluator."""
        gt_path = tmp_path / "ground_truth.json"
        with open(gt_path, 'w') as f:
            json.dump(gt_entries, f)
        return MetricsEvaluator(ground_truth_path=str(gt_path))

    def test_unlabeled_matched_records_not_tp(self, tmp_path):
        """
        Unlabeled records must NOT be counted as True Positives.
        """
        gt_entries = [
            {
                'payment_id': f'PAY_{i:03d}',
                'ledger_order_id': f'ORD_{i:03d}',
                'utr': f'UTR{i:06d}',
                'bank_txn_id': f'TXN_{i:06d}',
                'should_match': True,
                'root_cause': 'exact_match',
                'notes': 'Clean match',
                'messiness_type': 'exact_match',
            }
            for i in range(10)
        ]
        evaluator = self._build_evaluator_with_gt(tmp_path, gt_entries)

        verified_results = [
            {'payment_id': f'PAY_{i:03d}', 'type': 'settlement', 'final_status': ReconciliationState.EXACT_MATCH.value}
            for i in range(10)
        ]
        unverified_matched = [
            {'payment_id': f'PAY_UNLABELED_{i:03d}', 'type': 'settlement', 'final_status': ReconciliationState.EXACT_MATCH.value}
            for i in range(5)
        ]

        all_results = verified_results + unverified_matched

        metrics = evaluator.evaluate(all_results, output_path=None)

        assert metrics['true_positives'] == 10
        assert metrics['unverified_count'] == 5
        assert metrics['false_positives'] == 0
        
        assert metrics['ground_truth_coverage'] == 0.6667

    def test_unlabeled_non_matched_records_not_tn(self, tmp_path):
        """
        Unlabeled records flagged as exceptions must not become true negatives.
        """
        gt_entries = [
            {
                'payment_id': f'PAY_{i:03d}',
                'ledger_order_id': f'ORD_{i:03d}',
                'utr': f'UTR{i:06d}',
                'bank_txn_id': f'TXN_{i:06d}',
                'should_match': False,
                'root_cause': 'no_counterpart',
                'notes': 'Orphaned',
                'messiness_type': 'orphan_bank',
            }
            for i in range(6)
        ]
        evaluator = self._build_evaluator_with_gt(tmp_path, gt_entries)

        verified_results = [
            {'payment_id': f'PAY_{i:03d}', 'type': 'settlement', 'final_status': ReconciliationState.UNRESOLVED_EXCEPTION.value}
            for i in range(6)
        ]
        unverified_exceptions = [
            {'payment_id': f'PAY_UNK_{i:03d}', 'type': 'settlement', 'final_status': ReconciliationState.UNRESOLVED_EXCEPTION.value}
            for i in range(4)
        ]

        all_results = verified_results + unverified_exceptions
        metrics = evaluator.evaluate(all_results, output_path=None)

        assert metrics['true_negatives'] == 6
        assert metrics['unverified_count'] == 4

    def test_full_coverage_gives_coverage_1(self, tmp_path):
        """
        When every result record has a ground-truth entry,
        ground_truth_coverage must equal 1.0.
        """
        n = 15
        gt_entries = [
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
            for i in range(n)
        ]
        evaluator = self._build_evaluator_with_gt(tmp_path, gt_entries)
        results = [
            {'payment_id': f'PAY_{i:03d}', 'type': 'settlement', 'final_status': ReconciliationState.EXACT_MATCH.value}
            for i in range(n)
        ]
        metrics = evaluator.evaluate(results, output_path=None)

        assert metrics['ground_truth_coverage'] == 1.0
        assert metrics['unverified_count'] == 0

    def test_duplicate_gt_mapping_does_not_inflate_coverage(self, tmp_path):
        """
        Duplicate mappings should be false positives.
        """
        gt_entries = [
            {
                'payment_id': 'PAY_001',
                'ledger_order_id': 'ORD_001',
                'utr': 'UTR000001',
                'bank_txn_id': 'TXN_000001',
                'should_match': True,
                'root_cause': 'exact_match',
                'notes': '',
                'messiness_type': 'exact_match',
            },
            {
                'payment_id': 'PAY_002',
                'ledger_order_id': 'ORD_002',
                'utr': 'UTR000002',
                'bank_txn_id': 'TXN_000002',
                'should_match': True,
                'root_cause': 'exact_match',
                'notes': '',
                'messiness_type': 'exact_match',
            }
        ]
        evaluator = self._build_evaluator_with_gt(tmp_path, gt_entries)
        
        results = [
            {'payment_id': 'PAY_001', 'type': 'settlement', 'final_status': ReconciliationState.EXACT_MATCH.value},
            {'payment_id': 'PAY_001', 'type': 'settlement', 'final_status': ReconciliationState.EXACT_MATCH.value}
        ]
        
        metrics = evaluator.evaluate(results, output_path=None)
        
        assert metrics['ground_truth_coverage'] == 1.0
        assert metrics['duplicate_ground_truth_assignments'] == 1
        assert metrics['false_positives'] == 1

    def test_partial_batch_coverage_reports_correctly(self):
        """
        Verify that ground_truth_coverage correctly reports the fraction of the
        processed batch that has labels.
        """
        evaluator = MetricsEvaluator()
        
        def mock_load():
            return [
                {'payment_id': 'PAY_1', 'should_match': True},
                {'payment_id': 'PAY_2', 'should_match': True},
            ]
        evaluator._load_ground_truth = mock_load
        
        results = [
            {'payment_id': 'PAY_1', 'type': 'settlement', 'final_status': ReconciliationState.EXACT_MATCH.value},
            {'payment_id': 'PAY_2', 'type': 'settlement', 'final_status': ReconciliationState.EXACT_MATCH.value},
            {'payment_id': 'PAY_3', 'type': 'settlement', 'final_status': ReconciliationState.EXACT_MATCH.value},
            {'payment_id': 'PAY_4', 'type': 'settlement', 'final_status': ReconciliationState.UNRESOLVED_EXCEPTION.value},
            {'payment_id': 'PAY_5', 'type': 'settlement', 'final_status': ReconciliationState.LLM_ERROR.value},
        ]
        
        metrics = evaluator.evaluate(results, output_path=None)
        
        assert metrics['total_input_transactions'] == 5
        assert metrics['unverified_count'] == 3
        assert metrics['ground_truth_coverage'] == 0.4
