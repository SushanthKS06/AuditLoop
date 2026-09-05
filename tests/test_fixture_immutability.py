"""Source fixtures must survive normal execution untouched.

A normal pipeline run (CLI or in-process) must NEVER overwrite:
- data/fixtures/* (canonical benchmark inputs)
- data/*.csv / data/ground_truth.json (bundled snapshot)

All generated artifacts belong under runtime/runs/<run_id>/.

A second test pins reproducibility: two seeded runs produce identical
transaction identities, decisions, match types, and metrics after
normalising run-scoped metadata (timestamps, run IDs).
"""

import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRACKED_FIXTURES = [
    "data/fixtures/settlements_live.csv",
    "data/fixtures/bank_statement.csv",
    "data/fixtures/internal_ledger.csv",
    "data/fixtures/ground_truth.json",
    "data/settlements_live.csv",
    "data/bank_statement.csv",
    "data/internal_ledger.csv",
    "data/ground_truth.json",
]


def _hashes():
    out = {}
    for rel in TRACKED_FIXTURES:
        path = os.path.join(REPO, rel)
        assert os.path.exists(path), f"Expected fixture missing: {rel}"
        with open(path, "rb") as f:
            out[rel] = hashlib.sha256(f.read()).hexdigest()
    return out


def _run_in_tmp_cwd(tmp_path, run_id, monkeypatch):
    """Run the pipeline with all runtime output confined to tmp_path."""
    import os as _os

    from run_pipeline import ReconciliationPipeline

    monkeypatch.chdir(tmp_path)
    pipeline = ReconciliationPipeline(use_llm=False, run_id=run_id)
    pipeline._validate_path = lambda path, allowed: None
    gt = _os.path.join(REPO, "data", "fixtures", "ground_truth.json")
    pipeline.evaluator.ground_truth_path = gt
    pipeline.evaluator.ground_truth = pipeline.evaluator._load_ground_truth()
    return pipeline.run(
        settlements_path=_os.path.join(REPO, "data", "fixtures", "settlements_live.csv"),
        bank_path=_os.path.join(REPO, "data", "fixtures", "bank_statement.csv"),
        ledger_path=_os.path.join(REPO, "data", "fixtures", "internal_ledger.csv"),
        generate_if_missing=False,
        num_records=20,
        seed=42,
        messiness_ratio=0.25,
    )


def _normalised(payload, metrics):
    tx = sorted(
        (
            r.get("canonical_transaction_id") or r.get("evaluation_unit_id"),
            r.get("final_status"),
            r.get("match_type"),
        )
        for r in payload.get("transaction_results", [])
    )
    orphans = sorted(
        (r.get("type"), r.get("record_ids")) for r in payload.get("orphan_events", [])
    )
    duplicates = sorted(
        (r.get("final_status"), r.get("record_ids"))
        for r in payload.get("duplicate_events", [])
    )
    metric_keys = (
        "total_input_transactions", "evaluated_transactions", "matched_count",
        "exception_count", "true_positives", "false_positives", "true_negatives",
        "false_negatives", "precision", "recall", "f1_score", "match_rate",
        "coverage", "orphan_bank_records", "orphan_ledger_records",
        "duplicate_suspects",
    )
    return {
        "transactions": tx,
        "orphans": orphans,
        "duplicates": duplicates,
        "metrics": {k: metrics.get(k) for k in metric_keys},
    }


class TestFixtureImmutability:
    def test_pipeline_run_leaves_fixtures_untouched(self, tmp_path, monkeypatch):
        before = _hashes()
        result = _run_in_tmp_cwd(tmp_path, "immut_check", monkeypatch)
        assert "error" not in result
        after = _hashes()
        assert before == after, (
            "Pipeline execution mutated source fixtures: "
            + str([k for k in before if before[k] != after[k]])
        )

    def test_seeded_runs_are_reproducible(self, tmp_path, monkeypatch):
        first = _run_in_tmp_cwd(tmp_path, "repro_a", monkeypatch)
        second = _run_in_tmp_cwd(tmp_path, "repro_b", monkeypatch)
        norm_first = _normalised(first["results"], first["metrics"])
        norm_second = _normalised(second["results"], second["metrics"])
        assert norm_first == norm_second, (
            "Same seed produced different reconciliation outcomes."
        )
        assert len(norm_first["transactions"]) == 20
