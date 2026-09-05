import os
import sys
import json
import subprocess
import pandas as pd
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.matcher import DeterministicMatcher
from engine.exceptions import ExceptionDispatcher
from llm.client import create_client

def run_tests():
    print("====================================")
    print("1. Running Test Suite")
    print("====================================")
    # Get the correct pytest executable based on environment
    pytest_cmd = "pytest"
    if os.path.exists("myenv/Scripts/pytest.exe"):
        pytest_cmd = "myenv/Scripts/pytest.exe"
        
    result = subprocess.run([pytest_cmd, "tests/"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Tests failed!")
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)
    else:
        print("All tests passed.")
        print(result.stdout.split('\n')[-2])
    print()

def generate_benchmark_data():
    print("====================================")
    print("2. Generating Adversarial Benchmark")
    print("====================================")
    subprocess.run([sys.executable, "scripts/build_adversarial_benchmark.py"], check=True)
    print("Benchmark generated at data/adversarial_benchmark.json\n")

def run_benchmark():
    print("====================================")
    print("3. Running Benchmark Evaluation")
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
            
        results.append({
            "case_id": case["case_id"],
            "description": case["description"],
            "expected": expected,
            "actual": final_status,
            "passed": is_correct
        })
        
    accuracy = (correct_matches / len(cases)) * 100
    print(f"Benchmark Accuracy: {accuracy:.1f}% ({correct_matches}/{len(cases)})")
    
    # Generate Markdown Report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/benchmark_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Adversarial Benchmark Report\n\n")
        f.write(f"**Generated:** {datetime.utcnow().isoformat()}Z\n")
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
    print("4. Executing Demo Pipeline")
    print("====================================")
    subprocess.run([sys.executable, "run_pipeline.py", "--demo-disagreement"], check=True)
    print("Demo pipeline complete.\n")

if __name__ == "__main__":
    run_tests()
    generate_benchmark_data()
    run_benchmark()
    run_demo()
    print("Submission Verification Complete.")
