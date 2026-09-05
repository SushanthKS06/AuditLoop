"""OPTIONAL live-provider check — NOT part of the submission gate.

Requires GROQ_API_KEY and network access. Exercises one small exception
batch against the real Groq provider to confirm the Stage-3 proposal path
works end-to-end. Deterministic correctness is proven offline by
``scripts/verify_submission.py``; this script only checks live wiring.

Usage:
    python scripts/verify_live_llm.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    if not os.getenv("GROQ_API_KEY"):
        print("LIVE INTEGRATION TEST skipped: GROQ_API_KEY is not set.")
        print("This is expected offline and does not affect the submission gate.")
        return 0

    from run_pipeline import ReconciliationPipeline

    print("LIVE INTEGRATION TEST: running 10-record batch with real provider...")
    pipeline = ReconciliationPipeline(use_llm=True)
    result = pipeline.run(
        settlements_path="data/fixtures/settlements_live.csv",
        bank_path="data/fixtures/bank_statement.csv",
        ledger_path="data/fixtures/internal_ledger.csv",
        generate_if_missing=True,
        num_records=10,
        seed=42,
        messiness_ratio=0.4,
    )
    if "error" in result:
        print(f"LIVE INTEGRATION TEST failed: {result['error']}")
        return 1
    metrics = result.get("metrics", {})
    print(
        "LIVE INTEGRATION TEST complete: "
        f"inputs={metrics.get('total_input_transactions')} "
        f"matched={metrics.get('matched_count')} "
        f"disagreements={metrics.get('disagreement_count')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
