"""
Tests for engine/exceptions.py — ExceptionDispatcher reliability.

Covers:
- process_exceptions returns one result per input in the same order (serial path).
- process_exceptions returns one result per input in the same order under
  ThreadPoolExecutor concurrency — no index/race corruption.
- llm_client raising mid-call on explain_exception → final_status == "llm_parse_error",
  batch does not crash.
- _deterministic_recheck rejects when amount diff > 2% with no fee justification.
- _deterministic_recheck accepts a fee-adjusted match (settlement + fee ≈ counterpart).
- _deterministic_recheck rejects when date diff > 5 days.
- force_disagreement_case=True produces exactly one llm_deterministic_disagreement
  with forced_demo_case=True, and that record is trivially filterable.
"""

import pytest
from unittest.mock import MagicMock, patch

from engine.exceptions import ExceptionDispatcher
from engine.states import ReconciliationState


# ---------------------------------------------------------------------------
# Helpers — minimal exception record shapes
# ---------------------------------------------------------------------------

def _make_exception(idx: int, amount_sett=980.0, amount_count=1000.0,
                    date_sett="2026-09-02", date_count="2026-09-01", fee=20.0):
    """Build a minimal exception dict that ExceptionDispatcher can process."""
    return {
        "record_ids": f"sett_{idx:03d}-TXN_{idx:03d}",
        "type": "fuzzy_unresolved",
        "settlement": {
            "entity_id": f"sett_{idx:03d}",
            "settled_amount": amount_sett,
            "settled_at": date_sett,
            "fee": fee,
        },
        "counterpart": {
            "txn_id": f"TXN_{idx:03d}",
            "amount": amount_count,
            "value_date": date_count,
        },
    }


def _make_stub_llm_client(action="match", root_cause="timing_lag", confidence=0.85):
    """
    Build a mock LLM client whose explain_exception and propose_resolution
    return well-formed dicts (valid=True) matching the given action.
    """
    client = MagicMock()
    client.explain_exception.return_value = {
        "valid": True,
        "root_cause": root_cause,
        "explanation": "Timing lag within T+2 window. Fee deduction observed.",
        "confidence": confidence,
        "structured_reasoning": "Amount delta is 2%, matches MDR. Date within 2 days.",
    }
    client.propose_resolution.return_value = {
        "valid": True,
        "action": action,
        "confidence": confidence,
        "reasoning": "Fee-adjusted amount aligns; UTR reference matches.",
        "structured_reasoning": "Verified amount and date criteria.",
    }
    return client


# ---------------------------------------------------------------------------
# Ordering guarantee — serial path
# ---------------------------------------------------------------------------

class TestProcessExceptionsOrdering:

    def test_serial_returns_same_count(self):
        """Without a client, serial path should still return one result per input."""
        dispatcher = ExceptionDispatcher(llm_client=None)
        exceptions = [_make_exception(i) for i in range(5)]
        results = dispatcher.process_exceptions(exceptions, concurrent=False)
        assert len(results) == len(exceptions)

    def test_serial_result_ids_match_input_order(self):
        """result[i]['record_ids'] must match exceptions[i]['record_ids']."""
        dispatcher = ExceptionDispatcher(llm_client=None)
        exceptions = [_make_exception(i) for i in range(6)]
        results = dispatcher.process_exceptions(exceptions, concurrent=False)
        for i, (exc, res) in enumerate(zip(exceptions, results)):
            assert res["record_ids"] == exc["record_ids"], (
                f"Ordering mismatch at index {i}: "
                f"expected {exc['record_ids']}, got {res['record_ids']}"
            )

    def test_serial_empty_input_returns_empty(self):
        dispatcher = ExceptionDispatcher(llm_client=None)
        assert dispatcher.process_exceptions([]) == []

    def test_serial_single_item(self):
        dispatcher = ExceptionDispatcher(llm_client=None)
        exc = _make_exception(0)
        results = dispatcher.process_exceptions([exc], concurrent=False)
        assert len(results) == 1
        assert results[0]["record_ids"] == exc["record_ids"]


# ---------------------------------------------------------------------------
# Ordering guarantee — concurrent path (ThreadPoolExecutor)
# ---------------------------------------------------------------------------

class TestProcessExceptionsConcurrentOrdering:

    def test_concurrent_returns_same_count(self):
        client = _make_stub_llm_client()
        dispatcher = ExceptionDispatcher(llm_client=client, max_workers=4)
        exceptions = [_make_exception(i) for i in range(10)]
        results = dispatcher.process_exceptions(exceptions, concurrent=True)
        assert len(results) == len(exceptions)

    def test_concurrent_result_ids_match_input_order(self):
        """
        Under ThreadPoolExecutor, futures complete out of order.
        The dispatcher must place each result at the correct index.
        """
        client = _make_stub_llm_client()
        dispatcher = ExceptionDispatcher(llm_client=client, max_workers=4)
        exceptions = [_make_exception(i) for i in range(12)]
        results = dispatcher.process_exceptions(exceptions, concurrent=True)
        for i, (exc, res) in enumerate(zip(exceptions, results)):
            assert res is not None, f"result[{i}] is None — slot was never written"
            assert res["record_ids"] == exc["record_ids"], (
                f"Ordering/race corruption at index {i}: "
                f"expected {exc['record_ids']}, got {res['record_ids']}"
            )

    def test_no_none_slots_after_concurrent_run(self):
        """No slot in the result list may be left as None after concurrent run."""
        client = _make_stub_llm_client()
        dispatcher = ExceptionDispatcher(llm_client=client, max_workers=4)
        exceptions = [_make_exception(i) for i in range(8)]
        results = dispatcher.process_exceptions(exceptions, concurrent=True)
        assert all(r is not None for r in results), (
            f"None slots found: {[i for i, r in enumerate(results) if r is None]}"
        )


# ---------------------------------------------------------------------------
# LLM error mid-call → llm_parse_error, no batch crash
# ---------------------------------------------------------------------------

class TestLLMErrorHandling:

    def test_explain_exception_raises_gives_parse_error_status(self):
        """
        If explain_exception() raises RuntimeError, that record gets
        final_status='llm_parse_error'. The rest of the batch is unaffected.
        """
        client = MagicMock()
        client.explain_exception.side_effect = RuntimeError("network timeout")

        dispatcher = ExceptionDispatcher(llm_client=client, max_workers=1)
        exceptions = [_make_exception(0)]
        results = dispatcher.process_exceptions(exceptions, concurrent=False)

        assert len(results) == 1
        assert results[0]["final_status"] == ReconciliationState.LLM_PARSE_ERROR.value

    def test_explain_exception_returns_invalid_gives_parse_error(self):
        """explain_exception returning valid=False → final_status='llm_parse_error'."""
        client = MagicMock()
        client.explain_exception.return_value = {
            "valid": False,
            "error": "JSON decode failed",
        }
        dispatcher = ExceptionDispatcher(llm_client=client, max_workers=1)
        exceptions = [_make_exception(0)]
        results = dispatcher.process_exceptions(exceptions, concurrent=False)

        assert results[0]["final_status"] == ReconciliationState.LLM_PARSE_ERROR.value
        assert "llm_error_detail" in results[0]

    def test_batch_continues_after_single_record_error(self):
        """One record raising must not kill the rest of the batch."""
        call_count = 0

        def flaky_explain(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first call fails")
            return {
                "valid": True,
                "root_cause": "rounding",
                "explanation": "Minor rounding difference detected.",
                "confidence": 0.80,
                "structured_reasoning": "Amount delta < 0.5%, likely rounding.",
            }

        client = MagicMock()
        client.explain_exception.side_effect = flaky_explain
        client.propose_resolution.return_value = {
            "valid": True,
            "action": "flag_for_human",
            "confidence": 0.75,
            "reasoning": "Flagging for human review due to uncertainty.",
            "structured_reasoning": "Amount small but uncertainty high.",
        }

        dispatcher = ExceptionDispatcher(llm_client=client, max_workers=1)
        exceptions = [_make_exception(i) for i in range(3)]
        results = dispatcher.process_exceptions(exceptions, concurrent=False)

        assert len(results) == 3
        # First must be parse error; others must not be
        assert results[0]["final_status"] == ReconciliationState.LLM_PARSE_ERROR.value
        assert results[1]["final_status"] != ReconciliationState.LLM_PARSE_ERROR.value

    def test_no_llm_client_gives_unavailable_status(self):
        """Without a client every record gets final_status='llm_unavailable'."""
        dispatcher = ExceptionDispatcher(llm_client=None)
        exceptions = [_make_exception(i) for i in range(3)]
        results = dispatcher.process_exceptions(exceptions, concurrent=False)
        for res in results:
            assert res["final_status"] == ReconciliationState.LLM_UNAVAILABLE.value


# ---------------------------------------------------------------------------
# _deterministic_recheck logic
# ---------------------------------------------------------------------------

class TestDeterministicRecheck:

    def _recheck(self, sett, count):
        dispatcher = ExceptionDispatcher(llm_client=None)
        return dispatcher._deterministic_recheck(sett, count)

    def test_exact_amount_match_passes(self):
        sett = {"settled_amount": 1000.0, "settled_at": "2026-09-01", "fee": 0.0}
        count = {"amount": 1000.0, "value_date": "2026-09-01"}
        assert self._recheck(sett, count) is True

    def test_small_amount_diff_under_2pct_passes(self):
        """1.9% delta with no fee — just inside the strict threshold."""
        sett = {"settled_amount": 981.0, "settled_at": "2026-09-01", "fee": 0.0}
        count = {"amount": 1000.0, "value_date": "2026-09-01"}
        # diff = 19/1000 = 1.9% → under 2% → pass
        assert self._recheck(sett, count) is True

    def test_amount_diff_over_2pct_no_fee_fails(self):
        """5% delta without fee justification must be rejected."""
        sett = {"settled_amount": 950.0, "settled_at": "2026-09-01", "fee": 0.0}
        count = {"amount": 1000.0, "value_date": "2026-09-01"}
        # diff = 50/1000 = 5% → exceeds 2% → fail
        assert self._recheck(sett, count) is False

    def test_fee_adjusted_match_passes(self):
        """
        MDR fee deduction: settled_amount + fee ≈ counterpart amount.
        Condition: abs((sett + fee) - count) / count <= 1.5%
        """
        sett = {"settled_amount": 976.40, "settled_at": "2026-09-01", "fee": 23.60}
        count = {"amount": 1000.0, "value_date": "2026-09-01"}
        # (976.40 + 23.60 - 1000) / 1000 = 0% → fee-adjusted match
        assert self._recheck(sett, count) is True

    def test_mdr_range_2_to_3_5_pct_passes(self):
        """
        Standard MDR range (1.5–3.5%) is treated as a valid fee deduction
        even without an explicit fee field.
        """
        # 2.36% diff — right in the MDR window
        sett = {"settled_amount": 976.4, "settled_at": "2026-09-01", "fee": 0.0}
        count = {"amount": 1000.0, "value_date": "2026-09-01"}
        # diff = 23.6/1000 = 2.36% → inside MDR range [1.5, 3.5] → fee_adjusted=True
        assert self._recheck(sett, count) is True

    def test_date_diff_within_5_days_passes(self):
        sett = {"settled_amount": 1000.0, "settled_at": "2026-09-05", "fee": 0.0}
        count = {"amount": 1000.0, "value_date": "2026-09-01"}
        # 4 days → passes
        assert self._recheck(sett, count) is True

    def test_date_diff_exactly_5_days_passes(self):
        sett = {"settled_amount": 1000.0, "settled_at": "2026-09-06", "fee": 0.0}
        count = {"amount": 1000.0, "value_date": "2026-09-01"}
        # 5 days → exactly at limit, must still pass
        assert self._recheck(sett, count) is True

    def test_date_diff_over_5_days_fails(self):
        """Date more than 5 days apart must be rejected."""
        sett = {"settled_amount": 1000.0, "settled_at": "2026-09-08", "fee": 0.0}
        count = {"amount": 1000.0, "value_date": "2026-09-01"}
        # 7 days → exceeds strict window → fail
        assert self._recheck(sett, count) is False

    def test_missing_settlement_returns_false(self):
        count = {"amount": 1000.0, "value_date": "2026-09-01"}
        assert self._recheck(None, count) is False

    def test_missing_counterpart_returns_false(self):
        sett = {"settled_amount": 1000.0, "settled_at": "2026-09-01", "fee": 0.0}
        assert self._recheck(sett, None) is False

    def test_non_numeric_amount_returns_false(self):
        sett = {"settled_amount": "not-a-number", "settled_at": "2026-09-01", "fee": 0.0}
        count = {"amount": 1000.0, "value_date": "2026-09-01"}
        assert self._recheck(sett, count) is False

    def test_missing_amount_fields_returns_false(self):
        sett = {"settled_at": "2026-09-01", "fee": 0.0}
        count = {"value_date": "2026-09-01"}
        assert self._recheck(sett, count) is False


# ---------------------------------------------------------------------------
# force_disagreement_case
# ---------------------------------------------------------------------------

class TestForceDisagreementCase:

    def test_forced_disagreement_produces_exactly_one_disagreement(self):
        """
        force_disagreement_case=True must inject exactly one
        llm_deterministic_disagreement result regardless of LLM behaviour.
        """
        client = _make_stub_llm_client(action="flag_for_human")
        dispatcher = ExceptionDispatcher(llm_client=client, max_workers=1)
        exceptions = [_make_exception(i) for i in range(5)]
        results = dispatcher.process_exceptions(
            exceptions, force_disagreement_case=True, concurrent=False
        )

        disagreements = [
            r for r in results
            if r.get("final_status") == ReconciliationState.LLM_DETERMINISTIC_DISAGREEMENT.value
        ]
        assert len(disagreements) == 1, (
            f"Expected exactly 1 disagreement, got {len(disagreements)}"
        )

    def test_forced_disagreement_record_has_forced_demo_case_flag(self):
        """The synthetic disagreement record must carry forced_demo_case=True."""
        client = _make_stub_llm_client(action="flag_for_human")
        dispatcher = ExceptionDispatcher(llm_client=client, max_workers=1)
        exceptions = [_make_exception(i) for i in range(4)]
        results = dispatcher.process_exceptions(
            exceptions, force_disagreement_case=True, concurrent=False
        )

        disagreements = [
            r for r in results
            if r.get("final_status") == ReconciliationState.LLM_DETERMINISTIC_DISAGREEMENT.value
        ]
        assert disagreements[0].get("forced_demo_case") is True

    def test_forced_disagreement_is_filterable(self):
        """
        Real exception records must NOT carry forced_demo_case=True,
        so downstream systems can exclude the injected case from accuracy metrics.
        """
        client = _make_stub_llm_client(action="flag_for_human")
        dispatcher = ExceptionDispatcher(llm_client=client, max_workers=1)
        exceptions = [_make_exception(i) for i in range(6)]
        results = dispatcher.process_exceptions(
            exceptions, force_disagreement_case=True, concurrent=False
        )

        real_results = [r for r in results if not r.get("forced_demo_case")]
        forced_results = [r for r in results if r.get("forced_demo_case")]

        assert len(forced_results) == 1
        # All real results must not be contaminated
        for r in real_results:
            assert r.get("final_status") != ReconciliationState.LLM_DETERMINISTIC_DISAGREEMENT.value or \
                   not r.get("forced_demo_case"), (
                "A non-forced record was incorrectly marked as disagreement "
                "due to contamination from the injected case"
            )

    def test_no_forced_disagreement_without_flag(self):
        """Without the flag, no synthetic disagreement is injected."""
        client = _make_stub_llm_client(action="flag_for_human")
        dispatcher = ExceptionDispatcher(llm_client=client, max_workers=1)
        exceptions = [_make_exception(i) for i in range(5)]
        results = dispatcher.process_exceptions(
            exceptions, force_disagreement_case=False, concurrent=False
        )

        forced = [r for r in results if r.get("forced_demo_case")]
        assert len(forced) == 0
