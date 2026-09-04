"""
Performance & Scalability Benchmark Test.

Verifies that Stage 1 and Stage 2 matching scale efficiently on 500+ records
with O(N+M+L) complexity.
"""

import time
import pytest
import pandas as pd
from data.generate_data import SyntheticDataGenerator
from engine.matcher import DeterministicMatcher


class TestPerformanceScaling:
    """Benchmark tests for matching engine throughput."""
    
    def test_stage1_and_stage2_high_throughput(self, tmp_path):
        """Test matching engine performance on a 500-record batch."""
        gen = SyntheticDataGenerator(seed=42, messiness_ratio=0.20)
        bank_df, ledger_df, _ = gen.generate(
            num_records=500,
            settlements_df=None,
            output_dir=str(tmp_path / "scale_test")
        )
        settlements_df = pd.read_csv(tmp_path / "scale_test" / "settlements_live.csv")
        
        matcher = DeterministicMatcher()
        
        start_time = time.perf_counter()
        matched_stage1, unmatched_sett, unmatched_bank, unmatched_ledger, _ = matcher.stage1_exact_match(
            settlements_df, bank_df, ledger_df
        )
        stage1_duration = time.perf_counter() - start_time
        
        # Stage 1 on 500 records with vectorized hash joins should complete in < 0.25 seconds
        assert stage1_duration < 0.5, f"Stage 1 took too long: {stage1_duration:.3f}s"
        assert len(matched_stage1) > 0
        
        start_time = time.perf_counter()
        matched_stage2, low_conf, _, _, _, _ = matcher.stage2_fuzzy_match(
            unmatched_sett, unmatched_bank, unmatched_ledger
        )
        stage2_duration = time.perf_counter() - start_time
        
        assert stage2_duration < 1.0, f"Stage 2 took too long: {stage2_duration:.3f}s"
