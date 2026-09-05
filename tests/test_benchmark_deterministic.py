"""
Deterministic adversarial benchmark (30/30 target).

Uses MockLLM only. Does not require GROQ_API_KEY or network access.
Provider-resilience cases live in test_provider_resilience.py.
"""

import json
import os
import sys

import pandas as pd
import pytest

from engine.matcher import DeterministicMatcher
from engine.exceptions import ExceptionDispatcher
from engine.states import ReconciliationState


def _load_benchmark():
    benchmark_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'adversarial_benchmark.json'
    )
    if not os.path.exists(benchmark_path):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from scripts.build_adversarial_benchmark import create_benchmark
        create_benchmark()
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _mock_for_mode(mode: str):
    sys.path.insert(0, os.path.dirname(__file__))
    from mock_llm import MockLLMClient
    if mode in ("none", "unavailable_client"):
        return None
    return MockLLMClient(mode=mode)


def _run_case(case: dict) -> str:
    sett = case.get('settlement')
    bank = case.get('bank')
    ledger = case.get('ledger')

    sett_df = pd.DataFrame([sett]) if sett else pd.DataFrame()
    bank_df = pd.DataFrame([bank]) if bank else pd.DataFrame()
    ledger_df = pd.DataFrame([ledger]) if ledger else pd.DataFrame()

    matcher = DeterministicMatcher(
        confidence_threshold=0.85,
        amount_threshold_pct=1.0,
        date_window_days=3,
    )

    matched_df, unmatched_sett, unmatched_bank, unmatched_ledger, _ = \
        matcher.stage1_exact_match(sett_df, bank_df, ledger_df)

    if not matched_df.empty:
        return 'matched'

    fuzzy_matched, low_conf, us2, ub2, ul2, _ = \
        matcher.stage2_fuzzy_match(unmatched_sett, unmatched_bank, unmatched_ledger)

    if not fuzzy_matched.empty:
        return 'matched'

    exceptions = matcher.get_exceptions(low_conf, us2, ub2, ul2)
    if not exceptions:
        return 'unresolved_exception'

    llm = _mock_for_mode(case.get('mock_mode', 'match'))
    dispatcher = ExceptionDispatcher(llm_client=llm)
    processed = dispatcher.process_exceptions(exceptions)
    if processed:
        return processed[0].get('final_status', 'unresolved_exception')
    return 'unresolved_exception'


_STATUS_EQUIVALENTS = {
    'matched': {
        ReconciliationState.MATCHED.value,
        ReconciliationState.EXACT_MATCH.value,
        ReconciliationState.FUZZY_MATCH.value,
        ReconciliationState.MATCHED_LLM_VERIFIED.value,
        'matched',
    },
}


def _statuses_match(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    return actual in _STATUS_EQUIVALENTS.get(expected, {expected})


class TestAdversarialBenchmarkDeterministic:
    @pytest.fixture(autouse=True)
    def chdir_to_project(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'metrics').mkdir()

    def test_all_30_cases_pass(self):
        cases = _load_benchmark()
        assert len(cases) == 30, f"Expected 30 benchmark cases, found {len(cases)}"

        failures = []
        for case in cases:
            actual = _run_case(case)
            expected = case['expected_status']
            if not _statuses_match(expected, actual):
                failures.append({
                    'case_id': case['case_id'],
                    'description': case['description'],
                    'expected': expected,
                    'actual': actual,
                })

        score = len(cases) - len(failures)
        if failures:
            fail_lines = '\n'.join(
                f"  [{f['case_id']}] expected={f['expected']} actual={f['actual']} "
                f"({f['description']})"
                for f in failures
            )
            pytest.fail(
                f"Adversarial benchmark: {score}/{len(cases)} passed.\n"
                f"Failing cases:\n{fail_lines}"
            )

    @pytest.mark.parametrize("case_id", [
        "case_01_exact",
        "case_03_missing_bank",
        "case_04_missing_ledger",
        "case_05_missing_both",
        "case_06_amount_mismatch",
        "case_16_llm_match_rejected",
        "case_22_prompt_injection",
        "case_21_low_confidence_llm",
    ])
    def test_named_safety_case(self, case_id):
        cases = {c['case_id']: c for c in _load_benchmark()}
        case = cases[case_id]
        actual = _run_case(case)
        assert _statuses_match(case['expected_status'], actual), (
            f"[{case_id}] expected={case['expected_status']} actual={actual}"
        )

    def test_reproducible_across_two_runs(self):
        cases = _load_benchmark()
        first = [_run_case(c) for c in cases]
        second = [_run_case(c) for c in cases]
        assert first == second
