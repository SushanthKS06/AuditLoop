"""

Pipeline-level reconciliation tests.



Verifies that:

1. No transaction ID appears more than once in the final result set.

2. orphan_bank and duplicate_suspect records never resolve to 'matched'.

3. Exactly one transaction-level result exists per input settlement, with
   orphan/duplicate events reported in separate streams.



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

    Returns (payload, metrics, ground_truth) where payload is the structured
    PipelineResult dict with transaction_results / orphan_events /
    duplicate_events / exception_events.

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



    return result.get('results', {}), result.get('metrics', {}), ground_truth





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





def _canonical_id_of(row: dict) -> str:
    """Extract the canonical transaction / evaluation-unit identity."""
    for key in ('canonical_transaction_id', 'evaluation_unit_id'):
        val = row.get(key, '')
        if val and str(val).lower() not in ('none', 'nan', ''):
            return str(val).lower()
    # Fall back to the strong settlement identity for legacy rows.
    sett = row.get('settlement') or {}
    if isinstance(sett, dict):
        for key in ('entity_id', 'settlement_id'):
            val = sett.get(key, '')
            if val and str(val).lower() not in ('none', 'nan', ''):
                return str(val).lower()
    for key in ('entity_id', 'settlement_id'):
        val = row.get(key, '')
        if val and str(val).lower() not in ('none', 'nan', ''):
            return str(val).lower()
    return ''





# ---------------------------------------------------------------------------

# Tests

# ---------------------------------------------------------------------------



class TestPipelineReconciliation:

    """End-to-end pipeline deduplication and correctness tests."""



    def test_no_duplicate_payment_ids_in_results(self, tmp_path):

        """

        No payment_id must appear more than once among transaction-level
        results, and no canonical transaction ID may repeat.

        If a transaction was processed by multiple pipeline legs, Step 4.5
        deduplication must collapse it to a single row.

        """

        payload, metrics, _ = _run_pipeline_on_tmp(tmp_path, num_records=40, seed=42)
        all_results = payload.get('transaction_results', [])



        pids = [_payment_id_of(r) for r in all_results if _payment_id_of(r)]

        counts = Counter(pids)

        duplicates = {pid: cnt for pid, cnt in counts.items() if cnt > 1}



        assert not duplicates, (

            f"These payment_ids appear more than once in results.json: {duplicates}. "

            "Bug 2 deduplication not working."

        )

        canonical_ids = [_canonical_id_of(r) for r in all_results if _canonical_id_of(r)]
        canonical_counts = Counter(canonical_ids)
        canonical_dups = {cid: cnt for cid, cnt in canonical_counts.items() if cnt > 1}

        assert not canonical_dups, (

            f"Duplicate canonical transaction IDs in transaction_results: {canonical_dups}. "

            "One transaction must equal one transaction-level result."

        )



    def test_orphan_bank_never_resolves_to_matched(self, tmp_path):

        """

        Records generated as orphan_bank must never have final_status='matched'.

        After Bug 1 fix, they fall through to unmatched_settlement exceptions.

        """

        payload, _, ground_truth = _run_pipeline_on_tmp(

            tmp_path, num_records=50, seed=42, messiness=0.4

        )
        all_results = payload.get('transaction_results', [])



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

        payload, _, ground_truth = _run_pipeline_on_tmp(

            tmp_path, num_records=50, seed=42, messiness=0.4

        )
        all_results = payload.get('transaction_results', [])



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

        One input settlement = exactly one transaction-level result.

        transaction_results length must equal the number of input
        settlements; orphan/duplicate events live in their own streams and
        never inflate the transaction population. metrics
        total_input_transactions must agree, with no impossible_state flag.

        """

        num_records = 30
        payload, metrics, _ = _run_pipeline_on_tmp(tmp_path, num_records=num_records, seed=7)

        transaction_results = payload.get('transaction_results', [])
        orphan_events = payload.get('orphan_events', [])
        duplicate_events = payload.get('duplicate_events', [])



        assert len(transaction_results) == num_records, (

            f"transaction_results ({len(transaction_results)}) != input "

            f"settlements ({num_records}). Orphan/duplicate events "

            f"(orphans={len(orphan_events)}, duplicates={len(duplicate_events)}) "

            "must not leak into transaction-level results."

        )

        assert metrics.get('total_input_transactions') == num_records, (

            f"metrics total_input_transactions ({metrics.get('total_input_transactions')}) "

            f"!= input settlements ({num_records})."

        )
        assert metrics.get('impossible_state') is None, (

            f"Impossible metric state flagged: {metrics.get('impossible_state')}"

        )



        # No duplicate canonical identity across transaction rows (dedup guarantee)

        def _primary_id(row):

            cid = _canonical_id_of(row)
            if cid:
                return cid

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

        ids = [_primary_id(r) for r in transaction_results if _primary_id(r)]

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



        grouped = pipeline._deduplicate_results(rows)
        deduped = grouped['transaction_results']



        assert len(deduped) == 1, (

            f"Two rows for the same payment_id must merge into one, got {len(deduped)}."

        )

        assert deduped[0]['final_status'] == 'unresolved_exception', (

            f"Exception must win over 'matched' after merge. "

            f"Got: {deduped[0]['final_status']}"

        )
        assert deduped[0].get('canonical_transaction_id'), (
            "Merged transaction result must carry a canonical_transaction_id."
        )
        assert deduped[0].get('evaluation_unit_id'), (
            "Merged transaction result must carry an evaluation_unit_id."
        )
        assert deduped[0].get('_merged_from_count') == 2, (
            "Merge evidence (_merged_from_count) must be preserved."
        )


    def test_weak_id_collision_keeps_distinct_transactions(self, tmp_path):
        """
        Two DISTINCT settlements (different strong entity_id) that share a
        weak payment_id must NOT collapse into one transaction result.
        Weak identifiers are evidence, never identity.
        """
        pipeline = ReconciliationPipeline(use_llm=False)
        pipeline._validate_path = lambda path, allowed: None

        rows = [
            {
                'entity_id': 'sett_AAAA',
                'payment_id': 'PAY_SHARED_001',
                'final_status': 'exact_match',
                'type': 'payment',
                'confidence': 1.0,
            },
            {
                'entity_id': 'sett_BBBB',
                'payment_id': 'PAY_SHARED_001',  # weak-ID collision
                'final_status': 'exact_match',
                'type': 'payment',
                'confidence': 1.0,
            },
        ]

        grouped = pipeline._deduplicate_results(rows)

        assert len(grouped['transaction_results']) == 2, (
            f"Distinct settlements sharing a weak payment_id must stay separate, "
            f"got {len(grouped['transaction_results'])} transaction results."
        )


    def test_orphan_and_duplicate_events_are_separated(self, tmp_path):
        """
        Orphan bank/ledger rows and duplicate-suspect rows must be routed to
        their own event streams, never emitted as transaction-level rows.
        """
        pipeline = ReconciliationPipeline(use_llm=False)
        pipeline._validate_path = lambda path, allowed: None

        rows = [
            {
                'entity_id': 'sett_0001',
                'payment_id': 'PAY_0001',
                'final_status': 'exact_match',
                'type': 'payment',
                'confidence': 1.0,
            },
            {
                'type': 'unmatched_bank',
                'txn_id': 'TXN_ORPHAN_1',
                'bank': {'txn_id': 'TXN_ORPHAN_1'},
                'final_status': 'unmatched_bank',
                'confidence': 0.0,
            },
            {
                'type': 'unmatched_ledger',
                'order_id': 'ORD_ORPHAN_1',
                'ledger': {'order_id': 'ORD_ORPHAN_1'},
                'final_status': 'unmatched_ledger',
                'confidence': 0.0,
            },
            {
                'payment_id': 'PAY_DUP_1',
                'final_status': 'duplicate_suspect',
                'type': 'payment',
                'confidence': 0.0,
            },
        ]

        grouped = pipeline._deduplicate_results(rows)

        assert len(grouped['transaction_results']) == 1, (
            f"Only the genuine settlement row is a transaction result, "
            f"got {len(grouped['transaction_results'])}."
        )
        assert len(grouped['orphan_events']) == 2, (
            f"Expected 2 orphan events, got {len(grouped['orphan_events'])}."
        )
        assert len(grouped['duplicate_events']) == 1, (
            f"Expected 1 duplicate event, got {len(grouped['duplicate_events'])}."
        )
