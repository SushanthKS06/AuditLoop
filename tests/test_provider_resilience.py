"""LLM provider resilience is not a reconciliation accuracy failure."""

from engine.exceptions import ExceptionDispatcher
from engine.states import ReconciliationState
from tests.mock_llm import MockLLMClient


def _exc():
    return {
        'type': 'unmatched_settlement',
        'record_ids': 'sett_prov',
        'settlement': {'payment_id': 'PAY_P', 'amount': 1000.0, 'settled_amount': 980.0},
        'bank': {'amount': 980.0, 'txn_id': 'TXN_P'},
        'ledger': {'expected_amount': 1000.0, 'order_id': 'ORD_P'},
        'counterpart': {'amount': 980.0, 'txn_id': 'TXN_P'},
        'source': 'synthetic',
    }


class TestProviderResilience:
    def test_no_client_is_llm_unavailable(self):
        dispatcher = ExceptionDispatcher(llm_client=None)
        status = dispatcher.process_exceptions([_exc()])[0]['final_status']
        assert status == ReconciliationState.LLM_UNAVAILABLE.value
        assert status != ReconciliationState.MATCHED_LLM_VERIFIED.value

    def test_malformed_provider_payload(self):
        dispatcher = ExceptionDispatcher(llm_client=MockLLMClient(mode='malformed'))
        status = dispatcher.process_exceptions([_exc()])[0]['final_status']
        assert status == ReconciliationState.LLM_PARSE_ERROR.value

    def test_schema_invalid_payload(self):
        dispatcher = ExceptionDispatcher(llm_client=MockLLMClient(mode='schema_invalid'))
        status = dispatcher.process_exceptions([_exc()])[0]['final_status']
        assert status == ReconciliationState.LLM_PARSE_ERROR.value
