"""
Mock LLM Client for Deterministic Testing

Provides a configurable mock LLM that returns predictable responses without
any external API calls. Used by the adversarial benchmark test suite so that
benchmark accuracy is reproducible and does not depend on API availability.

Design:
- Default behaviour: propose 'match' with high confidence (mimics a healthy LLM)
- Configurable per-scenario: 'flag_for_human', 'reject_duplicate', parse errors
- Never makes network calls
"""

from typing import Optional, Dict, Any


class MockLLMClient:
    """
    Deterministic mock LLM for testing.

    All calls return immediately with pre-configured responses.
    No external network calls are ever made.

    Args:
        explain_response: Custom explain_exception response dict.
                          Defaults to a valid 'timing_lag' explanation.
        propose_response: Custom propose_resolution response dict.
                          Defaults to a valid 'match' proposal.
        always_fail: If True, both calls return invalid responses (simulates LLM outage).
    """

    DEFAULT_EXPLAIN = {
        'valid': True,
        'root_cause': 'rounding',
        'explanation': 'Minor fee deduction difference, within expected MDR range.',
        'structured_reasoning': '1. Amounts compared. 2. Within 2% threshold.',
        'confidence': 0.92,
    }

    DEFAULT_PROPOSE_MATCH = {
        'valid': True,
        'action': 'match',
        'reasoning': 'Fee-adjusted amounts align within tolerance.',
        'structured_reasoning': '1. Settlement net + fee = ledger gross. 2. Dates within window.',
        'confidence': 0.92,
    }

    DEFAULT_PROPOSE_FLAG = {
        'valid': True,
        'action': 'flag_for_human',
        'reasoning': 'Amount discrepancy exceeds automated matching threshold.',
        'structured_reasoning': '1. Amount diff > 5%. 2. No fee explanation found.',
        'confidence': 0.70,
    }

    DEFAULT_PROPOSE_REJECT = {
        'valid': True,
        'action': 'reject_duplicate',
        'reasoning': 'Same amount but conflicting identifiers suggest duplicate.',
        'structured_reasoning': '1. Amounts match. 2. Order IDs conflict.',
        'confidence': 0.85,
    }

    INVALID_RESPONSE = {
        'valid': False,
        'error': 'MockLLMClient: simulated parse failure',
    }

    def __init__(
        self,
        explain_response: Optional[Dict[str, Any]] = None,
        propose_response: Optional[Dict[str, Any]] = None,
        always_fail: bool = False,
    ):
        self._explain = explain_response or self.DEFAULT_EXPLAIN
        self._propose = propose_response or self.DEFAULT_PROPOSE_MATCH
        self._always_fail = always_fail

    def explain_exception(
        self, record_a: Optional[Dict], record_b: Optional[Dict]
    ) -> Dict[str, Any]:
        if self._always_fail:
            return self.INVALID_RESPONSE
        return dict(self._explain)

    def propose_resolution(
        self, record_a: Optional[Dict], record_b: Optional[Dict]
    ) -> Dict[str, Any]:
        if self._always_fail:
            return self.INVALID_RESPONSE
        return dict(self._propose)


class FlagForHumanMockLLM(MockLLMClient):
    """LLM that always flags cases for human review."""
    def __init__(self):
        super().__init__(propose_response=MockLLMClient.DEFAULT_PROPOSE_FLAG)


class RejectDuplicateMockLLM(MockLLMClient):
    """LLM that always rejects as duplicate."""
    def __init__(self):
        super().__init__(propose_response=MockLLMClient.DEFAULT_PROPOSE_REJECT)


class UnavailableMockLLM:
    """Simulates complete LLM unavailability (no client at all)."""
    # Used by passing None as llm_client to ExceptionDispatcher
    pass
