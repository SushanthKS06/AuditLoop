"""Submission verifier — fully offline and runtime-bounded.

Every step runs WITHOUT external network and WITHOUT a real LLM provider
key. The demo smoke test uses ``--no-llm`` (deterministic matching only)
and the adversarial benchmark uses the in-repo MockLLM. Live-provider
behaviour is covered separately and explicitly by
``scripts/verify_live_llm.py``, which is NOT part of the submission gate.

Each subprocess carries an explicit timeout so the verifier can never hang
on a network retry loop or a stalled child process.
"""

import os
import sys
import json
import subprocess
import time
import pandas as pd

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.matcher import DeterministicMatcher
from engine.exceptions import ExceptionDispatcher

# Hard per-step bounds (seconds). The whole verifier completes in minutes.
TIMEOUT_TESTS = 900
TIMEOUT_BENCHMARK_BUILD = 120
TIMEOUT_DEMO = 300


def _offline_env() -> dict:
    """Child-process environment with provider keys stripped.

    Guarantees the verifier cannot accidentally reach a live LLM provider
    even if keys are exported in the surrounding shell.
    """
    env = os.environ.copy()
    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"):
        env.pop(key, None)
    return env


def _run(cmd, timeout, step_name):
    """Run a subprocess with a hard timeout; fail loudly with context."""
    started = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env=_offline_env(),
        )
    except subprocess.TimeoutExpired:
        print(f"{step_name} TIMED OUT after {timeout}s (bound enforced).")
        sys.exit(124)
    elapsed = time.monotonic() - started
    print(f"[{step_name} finished in {elapsed:.1f}s]")
    return result


def run_tests():
    print("====================================")
    print("1. Running Test Suite (offline)")
    print("====================================")
    # Get the correct pytest executable based on environment
    pytest_cmd = "pytest"
    if os.path.exists("myenv/Scripts/pytest.exe"):
        pytest_cmd = "myenv/Scripts/pytest.exe"

    result = _run([pytest_cmd, "tests/", "-q"], TIMEOUT_TESTS, "pytest")
    if result.returncode != 0:
        print("Tests failed!")
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        sys.exit(result.returncode)
    else:
        print("All tests passed.")
        print(result.stdout.strip().split('\n')[-1])
    print()


def generate_benchmark_data():
    print("====================================")
    print("2. Generating Adversarial Benchmark (offline)")
    print("====================================")
    _run(
        [sys.executable, "scripts/build_adversarial_benchmark.py"],
        TIMEOUT_BENCHMARK_BUILD, "build_benchmark",
    )
    print("Benchmark generated at data/adversarial_benchmark.json\n")


def run_benchmark():
    print("====================================")
    print("3. Running Benchmark Evaluation (MockLLM, offline)")
    print("====================================")

    with open("data/adversarial_benchmark.json", "r") as f:
        cases = json.load(f)

    from tests.mock_llm import MockLLMClient

    matcher = DeterministicMatcher()

    results = []
    correct_matches = 0

    print(f"Evaluating {len(cases)} adversarial edge cases (MockLLM, no network)...")

    for case in cases:
        sett = case.get("settlement")
        bank = case.get("bank")
        ledger = case.get("ledger")
        expected = case.get("expected_status")
        mode = case.get("mock_mode", "match")

        sett_df = pd.DataFrame([sett]) if sett else pd.DataFrame()
        bank_df = pd.DataFrame([bank]) if bank else pd.DataFrame()
        ledger_df = pd.DataFrame([ledger]) if ledger else pd.DataFrame()

        matched_df, unmatched_sett, unmatched_bank, unmatched_ledger, _ = matcher.stage1_exact_match(
            sett_df, bank_df, ledger_df
        )

        final_status = None
        if not matched_df.empty:
            final_status = 'matched'
        else:
            fuzzy_matched, low_conf, us2, ub2, ul2, _ = matcher.stage2_fuzzy_match(
                unmatched_sett, unmatched_bank, unmatched_ledger
            )

            if not fuzzy_matched.empty:
                final_status = 'matched'
            else:
                exceptions = matcher.get_exceptions(low_conf, us2, ub2, ul2)
                if exceptions:
                    llm_client = MockLLMClient(mode=mode)
                    dispatcher = ExceptionDispatcher(llm_client=llm_client)
                    processed = dispatcher.process_exceptions(exceptions)
                    if processed:
                        final_status = processed[0].get('final_status')

        matched_equiv = {'matched', 'exact_match', 'fuzzy_match', 'matched_llm_verified'}
        is_correct = final_status == expected or (
            expected == 'matched' and final_status in matched_equiv
        )
        if is_correct:
            correct_matches += 1
        else:
            print(f"  MISS: {case['case_id']} expected={expected} actual={final_status}")

        results.append({
            "case_id": case["case_id"],
            "description": case["description"],
            "expected": expected,
            "actual": final_status,
            "passed": is_correct
        })

    accuracy = (correct_matches / len(cases)) * 100
    print(f"Benchmark Accuracy: {accuracy:.1f}% ({correct_matches}/{len(cases)})")
    if correct_matches != len(cases):
        print("Benchmark is not 30/30 — submission gate FAILED.")
        sys.exit(1)

    # Generate Markdown Report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/benchmark_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Adversarial Benchmark Report\n\n")
        # No generation timestamp: the report is fully determined by the
        # fixed benchmark fixture + deterministic MockLLM evaluation, so
        # re-running verification leaves a clean working tree.
        f.write(f"**Accuracy:** {accuracy:.1f}%\n\n")
        f.write("## Test Cases\n\n")
        f.write("| Case ID | Description | Expected | Actual | Passed |\n")
        f.write("|---------|-------------|----------|--------|--------|\n")
        for r in results:
            pass_icon = "✅" if r["passed"] else "❌"
            f.write(f"| {r['case_id']} | {r['description']} | `{r['expected']}` | `{r['actual']}` | {pass_icon} |\n")

    print(f"\nReport written to {report_path}\n")


def run_demo():
    print("====================================")
    print("4. Executing Demo Pipeline (offline, --no-llm)")
    print("====================================")
    # --no-llm keeps the demo smoke test deterministic and offline.
    # Live-provider behaviour is covered by scripts/verify_live_llm.py.
    result = _run(
        [sys.executable, "run_pipeline.py",
         "--demo-disagreement", "--no-llm", "--records", "20", "--seed", "42"],
        TIMEOUT_DEMO, "demo_pipeline",
    )
    if result.returncode != 0:
        print("Demo pipeline failed!")
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        sys.exit(result.returncode)
    print("Demo pipeline complete.\n")


if __name__ == "__main__":
    started = time.monotonic()
    run_tests()
    generate_benchmark_data()
    run_benchmark()
    run_demo()
    print(f"Submission Verification Complete in {time.monotonic() - started:.1f}s (offline).")
