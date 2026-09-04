"""
End-to-end metrics test.

Verifies that metrics can be regenerated correctly on a fixed seed.
This proves the pipeline is deterministic and not hardcoded.

Also contains TestStrictCoverageMode which proves that in "strict" mode
(the default), records that have no ground-truth entry are NEVER silently
promoted to true positives — they are excluded from precision/recall/F1
and counted in ``unverified_count``.
"""

import pytest
import os
import json
import tempfile
from pathlib import Path

from data.generate_data import SyntheticDataGenerator
from metrics.evaluate import MetricsEvaluator


class TestEndToEndMetrics:
    """Test that metrics are reproducible with fixed seeds."""

    def test_reproducible_generation(self, tmp_path):
        """Test that same seed produces same ground truth."""
        # Generate twice with same seed
        gen1 = SyntheticDataGenerator(seed=42, messiness_ratio=0.25)
        gen1.generate(num_records=20, settlements_df=None, output_dir=str(tmp_path / "run1"))

        gen2 = SyntheticDataGenerator(seed=42, messiness_ratio=0.25)
        gen2.generate(num_records=20, settlements_df=None, output_dir=str(tmp_path / "run2"))

        # Load ground truth from both runs
        with open(tmp_path / "run1" / "ground_truth.json") as f:
            gt1 = json.load(f)

        with open(tmp_path / "run2" / "ground_truth.json") as f:
            gt2 = json.load(f)

        # Should be identical
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

        # Should have differences
        differences = 0
        for i in range(min(len(gt1), len(gt2))):
            if gt1[i]['messiness_type'] != gt2[i]['messiness_type']:
                differences += 1

        # Expect some differences (not 100% guaranteed but very likely)
        assert differences > 0

    def test_metrics_evaluation_basic(self, tmp_path):
        """Test basic metrics evaluation."""
        # Create synthetic data
        gen = SyntheticDataGenerator(seed=42, messiness_ratio=0.25)
        gen.generate(num_records=20, settlements_df=None, output_dir=str(tmp_path))

        # Create mock results
        ground_truth = gen.ground_truth

        # Simulate results where we matched everything we should have
        mock_results = []
        for gt in ground_truth:
            mock_results.append({
                'payment_id': gt['payment_id'],
                'final_status': 'matched' if gt['should_match'] else 'unresolved_exception'
            })

        # Evaluate
        evaluator = MetricsEvaluator(ground_truth_path=str(tmp_path / "ground_truth.json"))
        metrics = evaluator.evaluate(mock_results, output_path=None)

        # Check that metrics are computed
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'match_rate' in metrics
        assert metrics['total_records'] == 20

        # With perfect matching, precision/recall should be high
        assert metrics['precision'] >= 0.8
        assert metrics['recall'] >= 0.8

    def test_ground_truth_covers_all_messiness_types(self, tmp_path):
        """Test that ground truth includes all injected messiness types."""
        gen = SyntheticDataGenerator(seed=42, messiness_ratio=0.35)  # Higher messiness
        test_dir = tmp_path / "gt_test"
        gen.generate(num_records=100, settlements_df=None, output_dir=str(test_dir))

        messiness_types = set(gt['messiness_type'] for gt in gen.ground_truth)

        # Should have at least exact matches and some messiness
        assert 'exact_match' in messiness_types
        assert len(messiness_types) > 1  # At least one messy type


class TestStrictCoverageMode:
    """
    Acceptance tests for Bug 1 fix.

    Proves that in "strict" mode, engine-matched records with NO ground-truth
    entry are NEVER silently promoted to true positives.
    """

    def _build_evaluator_with_gt(self, tmp_path, gt_entries):
        """Write a ground_truth.json and return a configured MetricsEvaluator."""
        gt_path = tmp_path / "ground_truth.json"
        with open(gt_path, 'w') as f:
            json.dump(gt_entries, f)
        return MetricsEvaluator(ground_truth_path=str(gt_path))

    def test_unlabeled_matched_records_not_tp_in_strict_mode(self, tmp_path):
        """
        Core acceptance test for Bug 1.

        Scenario
        --------
        10 records have ground-truth entries (all should_match=True).
        5 extra records have NO ground-truth entry but the engine matched them.

        In "strict" mode:
        - TP must equal exactly 10 (the verified records that were correctly matched).
        - The 5 unlabeled-but-matched records must NOT inflate TP.
        - unverified_count must equal 5.
        - ground_truth_coverage must equal 10/15 ≈ 0.6667.
        """
        # Build 10 ground-truth entries
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

        # 10 results with GT (all matched correctly → should be TP)
        verified_results = [
            {'payment_id': f'PAY_{i:03d}', 'final_status': 'matched'}
            for i in range(10)
        ]
        # 5 results WITHOUT any GT entry, but the engine matched them
        unverified_matched = [
            {'payment_id': f'PAY_UNLABELED_{i:03d}', 'final_status': 'matched'}
            for i in range(5)
        ]

        all_results = verified_results + unverified_matched

        metrics = evaluator.evaluate(all_results, output_path=None, coverage_mode="strict")

        # The 5 unlabeled records must NOT become true positives
        assert metrics['true_positives'] == 10, (
            f"Expected TP=10 (only verified records), got TP={metrics['true_positives']}. "
            "Bug 1 not fixed: unlabeled engine-matched records are silently graded as TP."
        )
        assert metrics['unverified_count'] == 5, (
            f"Expected unverified_count=5, got {metrics['unverified_count']}"
        )
        assert metrics['false_positives'] == 0
        
        # Coverage metric asserts
        # We had 15 records, 10 had GT, 5 didn't. So coverage = 10/15 = 0.6667
        assert metrics['ground_truth_coverage'] == 0.6667, (
            f"Expected coverage == 0.6667, got {metrics['ground_truth_coverage']}"
        )
        assert metrics['coverage_mode'] == 'strict'

    def test_unlabeled_non_matched_records_not_tn_in_strict_mode(self, tmp_path):
        """
        Unlabeled records that the engine correctly flagged as exceptions must
        also not silently become true negatives in strict mode.
        """
        gt_entries = [
            {
                'payment_id': f'PAY_{i:03d}',
                'ledger_order_id': f'ORD_{i:03d}',
                'utr': f'UTR{i:06d}',
                'bank_txn_id': f'TXN_{i:06d}',
                'should_match': False,   # These are known non-matches
                'root_cause': 'no_counterpart',
                'notes': 'Orphaned',
                'messiness_type': 'orphan_bank',
            }
            for i in range(6)
        ]
        evaluator = self._build_evaluator_with_gt(tmp_path, gt_entries)

        # 6 verified results — engine correctly flagged them as exceptions → TN
        verified_results = [
            {'payment_id': f'PAY_{i:03d}', 'final_status': 'unresolved_exception'}
            for i in range(6)
        ]
        # 4 unlabeled results that the engine also flagged as exceptions
        unverified_exceptions = [
            {'payment_id': f'PAY_UNK_{i:03d}', 'final_status': 'unresolved_exception'}
            for i in range(4)
        ]

        all_results = verified_results + unverified_exceptions
        metrics = evaluator.evaluate(all_results, output_path=None, coverage_mode="strict")

        assert metrics['true_negatives'] == 6, (
            f"Expected TN=6 (only verified), got TN={metrics['true_negatives']}"
        )
        assert metrics['unverified_count'] == 4

    def test_coverage_fields_always_present(self, tmp_path):
        """
        ground_truth_coverage, unverified_count, and coverage_mode must be
        present in every evaluate() call regardless of mode.
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
            }
        ]
        evaluator = self._build_evaluator_with_gt(tmp_path, gt_entries)
        results = [{'payment_id': 'PAY_001', 'final_status': 'matched'}]

        for mode in ('strict', 'assumed'):
            metrics = evaluator.evaluate(results, output_path=None, coverage_mode=mode)
            assert 'ground_truth_coverage' in metrics, (
                f"ground_truth_coverage missing in {mode} mode"
            )
            assert 'unverified_count' in metrics, (
                f"unverified_count missing in {mode} mode"
            )
            assert 'coverage_mode' in metrics, (
                f"coverage_mode missing in {mode} mode"
            )
            assert metrics['coverage_mode'] == mode

    def test_assumed_mode_includes_unlabeled_as_assumed(self, tmp_path):
        """
        Backward-compat: "assumed" mode still includes unlabeled records in
        TP/TN counts but exposes assumed_true_positives / assumed_true_negatives
        so callers can identify them.
        """
        gt_entries = [
            {
                'payment_id': 'PAY_000',
                'ledger_order_id': 'ORD_000',
                'utr': 'UTR000000',
                'bank_txn_id': 'TXN_000000',
                'should_match': True,
                'root_cause': 'exact_match',
                'notes': '',
                'messiness_type': 'exact_match',
            }
        ]
        evaluator = self._build_evaluator_with_gt(tmp_path, gt_entries)

        # 1 verified + 3 unlabeled (matched, no counterpart → assumed should_match=True)
        results = [
            {'payment_id': 'PAY_000', 'final_status': 'matched'},
            # These three have no GT entry and no orphan/unmatched type,
            # so heuristic makes assumed_should_match=True → assumed TP
            {'payment_id': 'PAY_A', 'final_status': 'matched', 'counterpart': {'x': 1}},
            {'payment_id': 'PAY_B', 'final_status': 'matched', 'counterpart': {'x': 1}},
            {'payment_id': 'PAY_C', 'final_status': 'matched', 'counterpart': {'x': 1}},
        ]

        metrics = evaluator.evaluate(results, output_path=None, coverage_mode="assumed")

        # In assumed mode, unverified_count is 0 — all records are scored
        assert metrics['unverified_count'] == 0
        # TP includes both verified and assumed
        assert metrics['true_positives'] == 4
        # Assumed-only tally is surfaced separately
        assert 'assumed_true_positives' in metrics
        assert metrics['assumed_true_positives'] == 3

    def test_full_coverage_gives_coverage_1(self, tmp_path):
        """
        When every result record has a ground-truth entry,
        ground_truth_coverage must equal 1.0 in strict mode.
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
            {'payment_id': f'PAY_{i:03d}', 'final_status': 'matched'}
            for i in range(n)
        ]
        metrics = evaluator.evaluate(results, output_path=None, coverage_mode="strict")

        assert metrics['ground_truth_coverage'] == 1.0
        assert metrics['unverified_count'] == 0

    def test_duplicate_gt_mapping_does_not_inflate_coverage(self, tmp_path):
        """
        Adversarial test: If two result records try to map to the same GT record,
        it should not inflate unique coverage. The duplicate should be flagged as
        a false positive or explicitly tracked in duplicate_ground_truth_assignments.
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
        
        # Two results mapping to PAY_001, none mapping to PAY_002
        results = [
            {'payment_id': 'PAY_001', 'final_status': 'matched'},
            {'payment_id': 'PAY_001', 'final_status': 'matched'}
        ]
        
        metrics = evaluator.evaluate(results, output_path=None, coverage_mode="strict")
        
        # 2 GT records total, but only 1 unique GT record was mapped to.
        # File utilization must be 0.5 (1/2).
        # Batch coverage is 1.0 (2/2) since both results mapped to something in GT.
        assert metrics['ground_truth_file_utilization'] == 0.5
        assert metrics['ground_truth_coverage'] == 1.0
        assert metrics['duplicate_ground_truth_assignments'] == 1
        assert metrics['false_positives'] == 1  # The duplicate mapping is a false positive

    def test_partial_batch_coverage_reports_correctly(self):
        """
        Verify that ground_truth_coverage correctly reports the fraction of the
        processed batch that has labels, NOT just the utilization of the GT file.
        """
        evaluator = MetricsEvaluator()
        
        # evaluate() calls _load_ground_truth(), so we must patch that
        def mock_load():
            return [
                {'payment_id': 'PAY_1', 'should_match': True},
                {'payment_id': 'PAY_2', 'should_match': True},
            ]
        evaluator._load_ground_truth = mock_load
        
        # 5 results total. Only 2 have ground truth. 3 are unverified.
        results = [
            {'payment_id': 'PAY_1', 'final_status': 'matched'},
            {'payment_id': 'PAY_2', 'final_status': 'matched'},
            {'payment_id': 'PAY_3', 'final_status': 'matched'},
            {'payment_id': 'PAY_4', 'final_status': 'exception'},
            {'payment_id': 'PAY_5', 'final_status': 'llm_error'},
        ]
        
        metrics = evaluator.evaluate(results, output_path=None, coverage_mode="strict")
        
        assert metrics['total_records'] == 5
        assert metrics['unverified_count'] == 3
        # Batch coverage = (5 - 3) / 5 = 2 / 5 = 0.4
        assert metrics['ground_truth_coverage'] == 0.4
        # File utilization = 2/2 = 1.0 (both GT entries were used)
        assert metrics['ground_truth_file_utilization'] == 1.0
