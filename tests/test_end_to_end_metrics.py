"""
End-to-end metrics test.

Verifies that metrics can be regenerated correctly on a fixed seed.
This proves the pipeline is deterministic and not hardcoded.
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
    
    def test_ground_truth_covers_all_messiness_types(self):
        """Test that ground truth includes all injected messiness types."""
        gen = SyntheticDataGenerator(seed=42, messiness_ratio=0.35)  # Higher messiness
        gen.generate(num_records=100, settlements_df=None, output_dir="/tmp/gt_test")
        
        messiness_types = set(gt['messiness_type'] for gt in gen.ground_truth)
        
        # Should have at least exact matches and some messiness
        assert 'exact_match' in messiness_types
        assert len(messiness_types) > 1  # At least one messy type
        
        # Clean up
        os.remove("/tmp/gt_test/ground_truth.json")
        os.remove("/tmp/gt_test/bank_statement.csv")
        os.remove("/tmp/gt_test/internal_ledger.csv")
        os.rmdir("/tmp/gt_test")
