"""

Pipeline-level reconciliation tests.



Verifies that:

1. No transaction ID appears more than once in the final result set.

2. orphan_bank and duplicate_suspect records never resolve to 'matched'.

3. total_records in the metrics report equals the number of unique transactions.



These tests run the pipeline in --no-llm mode on a controlled synthetic batch

so results are deterministic and don't require API keys.

"""



import json

import os

import tempfile

from collections import Counter

from pathlib import Path



import pytest

import pandas as pd



from data.generate_data import SyntheticDataGenerator

from engine.matcher import DeterministicMatcher

from run_pipeline import ReconciliationPipeline





# ---------------------------------------------------------------------------

# Helpers

# ---------------------------------------------------------------------------



def _run_pipeline_on_tmp(tmp_path, num_records=30, seed=42, messiness=0.35):

    """

    Generate a synthetic batch in tmp_path and run the pipeline (no LLM).

    Returns (all_results, metrics, ground_truth).

    """

    gen = SyntheticDataGenerator(seed=seed, messiness_ratio=messiness)

    _, _, ground_truth = gen.generate(

        num_records=num_records,

        settlements_df=None,

        output_dir=str(tmp_path),

    )



    sett_path = str(tmp_path / "settlements_live.csv")

    bank_path = str(tmp_path / "bank_statement.csv")

    ledger_path = str(tmp_path / "internal_ledger.csv")



    # Change cwd so pipeline writes results.json etc. into tmp_path

    original_cwd = os.getcwd()

    os.chdir(tmp_path)



    # Point ground_truth.json to the tmp copy

    try:

        pipeline = ReconciliationPipeline(use_llm=False)
        pipeline._validate_path = lambda path, allowed: None

        pipeline.evaluator.ground_truth_path = str(tmp_path / "ground_truth.json")

        pipeline.evaluator.ground_truth = pipeline.evaluator._load_ground_truth()



        result = pipeline.run(

            settlements_path=sett_path,

            bank_path=bank_path,

            ledger_path=ledger_path,

            generate_if_missing=False,

            num_records=num_records,

            seed=seed,

        )

    finally:

        os.chdir(original_cwd)



    return result.get('results', []), result.get('metrics', {}), ground_truth





def _payment_id_of(row: dict) -> str:

    """Extract the primary payment_id from a result row (top-level or nested)."""

    pid = row.get('payment_id', '')

    if pid and str(pid).lower() not in ('none', 'nan', ''):

        return str(pid).lower()

    sett = row.get('settlement') or {}

    if isinstance(sett, dict):

        pid = sett.get('payment_id', '')

        if pid and str(pid).lower() not in ('none', 'nan', ''):

            return str(pid).lower()

    return ''





# ---------------------------------------------------------------------------

# Tests

# ---------------------------------------------------------------------------



class TestPipelineReconciliation:

    """End-to-end pipeline deduplication and correctness tests."""



    def test_no_duplicate_payment_ids_in_results(self, tmp_path):

        """

        No payment_id must appear more than once in the final result set.

        If a transaction was processed by multiple pipeline legs, Step 4.5

        deduplication must collapse it to a single row.

        """

        all_results, metrics, _ = _run_pipeline_on_tmp(tmp_path, num_records=40, seed=42)



        pids = [_payment_id_of(r) for r in all_results if _payment_id_of(r)]

        counts = Counter(pids)

        duplicates = {pid: cnt for pid, cnt in counts.items() if cnt > 1}



        assert not duplicates, (

            f"These payment_ids appear more than once in results.json: {duplicates}. "

            "Bug 2 deduplication not working."

        )



    def test_orphan_bank_never_resolves_to_matched(self, tmp_path):

        """

        Records generated as orphan_bank must never have final_status='matched'.

        After Bug 1 fix, they fall through to unmatched_settlement exceptions.

        """

        all_results, _, ground_truth = _run_pipeline_on_tmp(

            tmp_path, num_records=50, seed=42, messiness=0.4

        )



        orphan_pids = {

            gt['payment_id']

            for gt in ground_truth

            if gt.get('messiness_type') in ('orphan_bank', 'orphan_ledger')

        }



        if not orphan_pids:

            pytest.skip("No orphan records generated at this seed/messiness — increase messiness.")



        matched_orphans = []

        for row in all_results:

            pid = _payment_id_of(row)

            if pid in {p.lower() for p in orphan_pids}:

                if row.get('final_status') in ('matched', 'matched_llm_verified'):

                    matched_orphans.append(pid)



        assert not matched_orphans, (

            f"Orphan records should NEVER resolve to matched. "

            f"These did: {matched_orphans}"

        )



    def test_duplicate_suspect_never_resolves_to_matched(self, tmp_path):

        """

        Records generated as duplicate_suspect must never have final_status='matched'.

        """

        all_results, _, ground_truth = _run_pipeline_on_tmp(

            tmp_path, num_records=50, seed=42, messiness=0.4

        )



        dup_pids = {

            gt['payment_id']

            for gt in ground_truth

            if gt.get('messiness_type') == 'duplicate_suspect'

        }



        if not dup_pids:

            pytest.skip("No duplicate_suspect records at this seed/messiness.")



        matched_dups = []

        for row in all_results:

            pid = _payment_id_of(row)

            if pid in {p.lower() for p in dup_pids}:

                if row.get('final_status') in ('matched', 'matched_llm_verified'):

                    matched_dups.append(pid)



        assert not matched_dups, (

            f"Duplicate-suspect records should NEVER resolve to matched. "

            f"These did: {matched_dups}"

        )



    def test_total_records_equals_unique_transactions(self, tmp_path):

        """

        metrics['total_records'] must equal len(all_results) after deduplication.

        Each result row is exactly one unique transaction (payment or bank txn),

        not a double-counted per-source row.

        """

        all_results, metrics, _ = _run_pipeline_on_tmp(tmp_path, num_records=30, seed=7)



        reported_total = metrics.get('total_records', -1)

        assert reported_total == len(all_results), (

            f"metrics.total_records ({reported_total}) != len(all_results) "

            f"({len(all_results)}). Pipeline is scoring a different list than it saved."

        )



        # No duplicate primary identifier across rows (dedup guarantee)

        def _primary_id(row):

            for k in ('payment_id', 'txn_id', 'bank_txn_id', 'entity_id', 'record_ids'):

                v = row.get(k, '')

                if v and str(v).lower() not in ('', 'none', 'nan'):

                    return str(v).lower()

            for nest in ('settlement', 'counterpart'):

                d = row.get(nest) or {}

                if isinstance(d, dict):

                    for k in ('payment_id', 'txn_id'):

                        v = d.get(k, '')

                        if v and str(v).lower() not in ('', 'none', 'nan'):

                            return str(v).lower()

            return None



        from collections import Counter

        ids = [_primary_id(r) for r in all_results if _primary_id(r)]

        dups = {k: v for k, v in Counter(ids).items() if v > 1}

        assert not dups, (

            f"Duplicate identifiers after deduplication: {dups}. "

            "Bug 2 deduplication is not preventing cross-source double-counting."

        )



    def test_deduplication_merges_contradictory_rows(self, tmp_path):

        """

        Directly test _deduplicate_results: two rows sharing a payment_id

        where one is 'matched' and one is 'unresolved_exception' must merge

        to 'unresolved_exception' (exception wins).

        """

        pipeline = ReconciliationPipeline(use_llm=False)
        pipeline._validate_path = lambda path, allowed: None



        rows = [

            {

                'payment_id': 'PAY_TEST_001',

                'final_status': 'matched',

                'type': 'payment',

                'confidence': 1.0,

            },

            {

                'payment_id': 'PAY_TEST_001',  # same transaction, different leg

                'final_status': 'unresolved_exception',

                'type': 'unmatched_settlement',

                'confidence': 0.0,

            },

        ]



        deduped = pipeline._deduplicate_results(rows)



        assert len(deduped) == 1, (

            f"Two rows for the same payment_id must merge into one, got {len(deduped)}."

        )

        assert deduped[0]['final_status'] == 'unresolved_exception', (

            f"Exception must win over 'matched' after merge. "

            f"Got: {deduped[0]['final_status']}"

        )


