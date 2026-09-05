"""
Messiness Propagation Tests

Proves that the messiness_ratio parameter flows from API → pipeline → generator
and actually affects the ratio of exception records generated.

These tests exist specifically to prevent silent regression where the parameter
is received at the API but a hardcoded default is used downstream.
"""

import pytest
from data.generate_data import SyntheticDataGenerator
from engine.states import ReconciliationState


class TestMessinessPropagation:
    """messiness_ratio must actually affect the proportion of exception records."""

    def test_zero_messiness_produces_near_zero_exceptions(self, tmp_path):
        """
        messiness_ratio=0.0 means _decide_messiness() always returns 'exact_match'.
        Ground truth should have 0 or very few records with should_match=False.
        """
        gen = SyntheticDataGenerator(seed=42, messiness_ratio=0.0)
        gen.generate(num_records=40, settlements_df=None, output_dir=str(tmp_path))

        non_matching = [g for g in gen.ground_truth if not g['should_match']]
        # With messiness=0, orphan/duplicate types are never chosen → no false records
        # Allow a small tolerance for the 'exact_match' override path
        assert len(non_matching) == 0, (
            f"messiness_ratio=0.0 should produce 0 non-matching records. "
            f"Got {len(non_matching)}: {[g['messiness_type'] for g in non_matching]}"
        )

    def test_full_messiness_produces_high_exception_rate(self, tmp_path):
        """
        messiness_ratio=1.0 means every record gets injected with a messiness type.
        The majority of records should have messiness_type != 'exact_match'.
        """
        gen = SyntheticDataGenerator(seed=42, messiness_ratio=1.0)
        gen.generate(num_records=60, settlements_df=None, output_dir=str(tmp_path))

        messy = [g for g in gen.ground_truth if g['messiness_type'] != 'exact_match']
        messy_ratio = len(messy) / len(gen.ground_truth)

        assert messy_ratio >= 0.90, (
            f"messiness_ratio=1.0 should produce ≥90% messy records. "
            f"Got {messy_ratio:.1%} ({len(messy)}/{len(gen.ground_truth)})"
        )

    def test_25_percent_messiness_produces_approximately_correct_ratio(self, tmp_path):
        """
        messiness_ratio=0.25 → ~25% of records should be non-exact-match.
        Allow ±15% tolerance due to random sampling variance.
        """
        gen = SyntheticDataGenerator(seed=42, messiness_ratio=0.25)
        gen.generate(num_records=200, settlements_df=None, output_dir=str(tmp_path))

        messy = [g for g in gen.ground_truth if g['messiness_type'] != 'exact_match']
        messy_ratio = len(messy) / len(gen.ground_truth)

        assert 0.10 <= messy_ratio <= 0.40, (
            f"messiness_ratio=0.25 should produce 10-40% messy records. "
            f"Got {messy_ratio:.1%} ({len(messy)}/{len(gen.ground_truth)})"
        )

    def test_different_messiness_values_produce_different_ratios(self, tmp_path):
        """Higher messiness_ratio must produce more messy records than lower."""
        gen_low = SyntheticDataGenerator(seed=42, messiness_ratio=0.10)
        gen_low.generate(num_records=100, settlements_df=None,
                         output_dir=str(tmp_path / "low"))

        gen_high = SyntheticDataGenerator(seed=42, messiness_ratio=0.80)
        gen_high.generate(num_records=100, settlements_df=None,
                          output_dir=str(tmp_path / "high"))

        low_messy = sum(1 for g in gen_low.ground_truth if g['messiness_type'] != 'exact_match')
        high_messy = sum(1 for g in gen_high.ground_truth if g['messiness_type'] != 'exact_match')

        assert high_messy > low_messy, (
            f"messiness_ratio=0.80 ({high_messy} messy) must produce more "
            f"messy records than messiness_ratio=0.10 ({low_messy} messy)."
        )

    def test_pipeline_uses_messiness_ratio_from_caller(self, tmp_path):
        """
        ReconciliationPipeline.run(messiness_ratio=...) must pass it to generator.
        Regression test: previously hardcoded 0.40 was used regardless of argument.
        """
        import os
        from run_pipeline import ReconciliationPipeline

        # The pipeline writes data files to 'data/' relative to the working directory.
        # Set up a temporary working directory with the required subdirectory.
        data_dir = tmp_path / 'data'
        data_dir.mkdir()
        (tmp_path / 'runtime').mkdir()
        (tmp_path / 'metrics').mkdir()

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            pipeline = ReconciliationPipeline(use_llm=False)

            # Run with messiness_ratio=0.0 → should produce near-zero exceptions
            result = pipeline.run(
                settlements_path='data/settlements_live.csv',
                bank_path='data/bank_statement.csv',
                ledger_path='data/internal_ledger.csv',
                generate_if_missing=True,
                num_records=30,
                seed=42,
                messiness_ratio=0.0,
            )
        finally:
            os.chdir(original_cwd)

        assert 'results' in result or 'metrics' in result, (
            "Pipeline run failed to return results."
        )
        # With messiness=0.0, all records should be exact_match → should resolve to matched
        results = result.get('results', [])
        exceptions = [r for r in results if r.get('final_status') not in (
            ReconciliationState.EXACT_MATCH.value,
            ReconciliationState.FUZZY_MATCH.value,
            ReconciliationState.MATCHED.value,
            ReconciliationState.MATCHED_LLM_VERIFIED.value
        )]
        exception_rate = len(exceptions) / len(results) if results else 0

        assert exception_rate < 0.20, (
            f"Pipeline with messiness_ratio=0.0 produced {exception_rate:.1%} exceptions. "
            f"Expected near-zero. Hardcoded 0.40 is likely still being used."
        )
