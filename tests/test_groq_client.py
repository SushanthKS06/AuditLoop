"""
Tests for llm/client.py — GroqClient and create_client factory.

All HTTP calls are mocked via unittest.mock — no real network calls are made.

Covers:
- GroqClient raises ValueError when GROQ_API_KEY is absent.
- create_client() returns None (not a crash) when key is missing.
- _parse_tool_response with a valid mock API response returns valid=True and
  correct data fields.
- _parse_tool_response with non-JSON / malformed tool-call arguments returns
  valid=False without raising.
- _parse_tool_response with a response missing required Pydantic fields returns
  valid=False.
- _execute_with_retry retries on transient failure and exhausts max_retries;
  verify retry count through mock call counter.
- 429-style errors trigger the longer delay branch (mock time.sleep to verify).
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, call

from llm.schemas import ExplainExceptionResponse, ProposeResolutionResponse


# ---------------------------------------------------------------------------
# Helpers — build mock Groq API response objects
# ---------------------------------------------------------------------------

def _build_mock_response(tool_name: str, arguments: dict | str):
    """
    Build a mock chat-completions response that looks like:
      response.choices[0].message.tool_calls[0].function.arguments = <json str>
    """
    if isinstance(arguments, dict):
        args_str = json.dumps(arguments)
    else:
        args_str = arguments  # intentionally broken string

    mock_function = MagicMock()
    mock_function.name = tool_name
    mock_function.arguments = args_str

    mock_tool_call = MagicMock()
    mock_tool_call.function = mock_function

    mock_message = MagicMock()
    mock_message.tool_calls = [mock_tool_call]
    mock_message.content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def _valid_explain_args():
    return {
        "structured_reasoning": "Amount delta is 2%, matches MDR fee. Date within T+2 window.",
        "root_cause": "timing_lag",
        "explanation": "Settlement date two days after bank transaction, within T+2 window.",
        "confidence": 0.87,
    }


def _valid_propose_args():
    return {
        "structured_reasoning": "UTR matches. Fee deduction explains delta. Date within lag.",
        "action": "match",
        "confidence": 0.91,
        "reasoning": "Fee-adjusted amount aligns and UTR reference is identical.",
    }


# ---------------------------------------------------------------------------
# Fixture — a GroqClient with injected dummy key and mocked Groq constructor
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_groq_client():
    """
    Return an initialized GroqClient with:
      - A fake API key injected via the constructor argument
      - The Groq SDK constructor patched so no real connection is made
    """
    from llm.client import GroqClient

    with patch("llm.client.Groq") as mock_groq_cls:
        mock_groq_instance = MagicMock()
        mock_groq_cls.return_value = mock_groq_instance

        client = GroqClient(api_key="test-key-for-unit-tests")
        client._groq_instance = mock_groq_instance  # expose for test assertions
        yield client


# ---------------------------------------------------------------------------
# Missing API key — documented failure mode
# ---------------------------------------------------------------------------

class TestMissingAPIKey:

    def test_raises_value_error_when_no_key_env_or_arg(self):
        """
        README states: 'LLM disabled if missing.'
        Client must raise ValueError cleanly, not crash with AttributeError.
        """
        from llm.client import GroqClient

        with patch.dict(os.environ, {}, clear=True):
            # Ensure GROQ_API_KEY is not present
            os.environ.pop("GROQ_API_KEY", None)
            with patch("llm.client.Groq"):
                with pytest.raises(ValueError, match="GROQ_API_KEY"):
                    GroqClient(api_key=None)

    def test_create_client_returns_none_when_key_missing(self):
        """
        create_client() is the safe factory — must return None, not raise.
        """
        from llm.client import create_client

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GROQ_API_KEY", None)
            with patch("llm.client.Groq"):
                result = create_client()
        assert result is None

    def test_create_client_returns_client_when_key_present(self):
        """When the key IS set, create_client() returns a GroqClient instance."""
        from llm.client import create_client, GroqClient

        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy-key"}):
            with patch("llm.client.Groq"):
                result = create_client()
        assert isinstance(result, GroqClient)


# ---------------------------------------------------------------------------
# _parse_tool_response — valid cases
# ---------------------------------------------------------------------------

class TestParseToolResponseValid:

    def test_valid_explain_response_returns_valid_true(self, mock_groq_client):
        response = _build_mock_response("explain_exception", _valid_explain_args())
        result = mock_groq_client._parse_tool_response(response, ExplainExceptionResponse)
        assert result["valid"] is True
        assert result["root_cause"] == "timing_lag"
        assert result["confidence"] == pytest.approx(0.87)

    def test_valid_propose_response_returns_valid_true(self, mock_groq_client):
        response = _build_mock_response("propose_resolution", _valid_propose_args())
        result = mock_groq_client._parse_tool_response(response, ProposeResolutionResponse)
        assert result["valid"] is True
        assert result["action"] == "match"
        assert result["confidence"] == pytest.approx(0.91)

    def test_valid_explain_response_includes_all_fields(self, mock_groq_client):
        response = _build_mock_response("explain_exception", _valid_explain_args())
        result = mock_groq_client._parse_tool_response(response, ExplainExceptionResponse)
        for key in ("structured_reasoning", "root_cause", "explanation", "confidence"):
            assert key in result, f"Missing field '{key}' in parsed result"

    def test_unknown_root_cause_is_mapped_to_unclassified(self, mock_groq_client):
        """
        The client has a fallback: unknown root_cause enum values are
        remapped to 'unclassified' rather than rejected entirely.
        This ensures LLM responses with novel phrasing don't hard-fail.
        """
        args = {**_valid_explain_args(), "root_cause": "completely_unknown_cause"}
        response = _build_mock_response("explain_exception", args)
        result = mock_groq_client._parse_tool_response(response, ExplainExceptionResponse)
        assert result["valid"] is True
        assert result["root_cause"] == "unclassified"


# ---------------------------------------------------------------------------
# _parse_tool_response — invalid / rejection cases
# ---------------------------------------------------------------------------

class TestParseToolResponseInvalid:

    def test_non_json_arguments_returns_valid_false(self, mock_groq_client):
        """Raw non-JSON string in tool arguments must not raise — return valid=False."""
        response = _build_mock_response("explain_exception", "THIS IS NOT JSON {{{{")
        result = mock_groq_client._parse_tool_response(response, ExplainExceptionResponse)
        assert result["valid"] is False
        assert "error" in result

    def test_missing_required_field_returns_valid_false(self, mock_groq_client):
        """JSON that is missing 'confidence' must fail Pydantic validation."""
        args = {k: v for k, v in _valid_explain_args().items() if k != "confidence"}
        response = _build_mock_response("explain_exception", args)
        result = mock_groq_client._parse_tool_response(response, ExplainExceptionResponse)
        assert result["valid"] is False

    def test_wrong_type_confidence_returns_valid_false(self, mock_groq_client):
        """confidence as a non-numeric string must fail Pydantic validation."""
        args = {**_valid_explain_args(), "confidence": "very_high"}
        response = _build_mock_response("explain_exception", args)
        result = mock_groq_client._parse_tool_response(response, ExplainExceptionResponse)
        assert result["valid"] is False

    def test_confidence_out_of_range_returns_valid_false(self, mock_groq_client):
        """confidence > 1.0 must be rejected by Pydantic ge/le constraints."""
        args = {**_valid_explain_args(), "confidence": 1.5}
        response = _build_mock_response("explain_exception", args)
        result = mock_groq_client._parse_tool_response(response, ExplainExceptionResponse)
        assert result["valid"] is False

    def test_empty_tool_calls_and_no_content_returns_valid_false(self, mock_groq_client):
        """Response with no tool_calls and no content → valid=False."""
        mock_message = MagicMock()
        mock_message.tool_calls = None
        mock_message.content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = mock_groq_client._parse_tool_response(
            mock_response, ExplainExceptionResponse
        )
        assert result["valid"] is False

    def test_invalid_action_in_propose_returns_valid_false(self, mock_groq_client):
        """action value not in Literal set must be rejected for ProposeResolutionResponse."""
        args = {**_valid_propose_args(), "action": "approve_immediately"}
        response = _build_mock_response("propose_resolution", args)
        result = mock_groq_client._parse_tool_response(response, ProposeResolutionResponse)
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# _execute_with_retry — retry count and give-up behaviour
# ---------------------------------------------------------------------------

class TestExecuteWithRetry:

    def test_succeeds_on_first_try(self, mock_groq_client):
        """No retry needed — callable succeeds immediately."""
        counter = {"calls": 0}

        def api_call():
            counter["calls"] += 1
            return "success"

        with patch("time.sleep"):
            result = mock_groq_client._execute_with_retry(api_call, max_retries=3)

        assert result == "success"
        assert counter["calls"] == 1

    def test_retries_on_transient_failure_then_succeeds(self, mock_groq_client):
        """Fails twice, succeeds on third attempt; verifies retry count."""
        counter = {"calls": 0}

        def flaky_call():
            counter["calls"] += 1
            if counter["calls"] < 3:
                raise RuntimeError("transient error")
            return "ok"

        with patch("time.sleep"):
            result = mock_groq_client._execute_with_retry(flaky_call, max_retries=4)

        assert result == "ok"
        assert counter["calls"] == 3

    def test_gives_up_after_max_retries_and_raises(self, mock_groq_client):
        """Exhausting all retries must re-raise the last exception."""
        counter = {"calls": 0}

        def always_fails():
            counter["calls"] += 1
            raise RuntimeError("always broken")

        with patch("time.sleep"):
            with pytest.raises(RuntimeError, match="always broken"):
                mock_groq_client._execute_with_retry(always_fails, max_retries=2)

        # Called: attempt 0, 1, 2 = 3 total (max_retries=2 means 2 retries after first)
        assert counter["calls"] == 3

    def test_429_error_triggers_sleep(self, mock_groq_client):
        """Rate-limit errors must call time.sleep (delay path is exercised)."""
        counter = {"calls": 0}

        def rate_limited():
            counter["calls"] += 1
            if counter["calls"] <= 1:
                raise Exception("Error 429: rate limit exceeded")
            return "ok"

        with patch("time.sleep") as mock_sleep:
            result = mock_groq_client._execute_with_retry(rate_limited, max_retries=2)

        assert result == "ok"
        # sleep must have been called at least once due to the 429
        mock_sleep.assert_called()

    def test_retry_sleep_is_called_between_attempts(self, mock_groq_client):
        """time.sleep must be called for each retry, not skipped."""
        attempts = {"n": 0}

        def fail_twice():
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise Exception("network error")
            return "done"

        with patch("time.sleep") as mock_sleep:
            mock_groq_client._execute_with_retry(fail_twice, max_retries=3)

        # Two failures → two sleeps
        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# explain_exception and propose_resolution — request shape (function calling)
# ---------------------------------------------------------------------------

class TestClientRequestShape:

    def test_explain_exception_calls_chat_completions(self, mock_groq_client):
        """
        explain_exception must invoke chat.completions.create with the
        correct tool_choice forcing the 'explain_exception' function.
        """
        # Patch _execute_with_retry to bypass real call logic
        mock_api_response = _build_mock_response("explain_exception", _valid_explain_args())
        mock_groq_client.client.chat.completions.create.return_value = mock_api_response

        with patch.object(mock_groq_client, "_execute_with_retry",
                          side_effect=lambda fn, **kw: fn()):
            result = mock_groq_client.explain_exception(
                record_a={"settled_amount": 980.0, "settled_at": "2026-09-01", "fee": 20.0},
                record_b={"amount": 1000.0, "value_date": "2026-09-01"},
            )

        assert mock_groq_client.client.chat.completions.create.called
        call_kwargs = mock_groq_client.client.chat.completions.create.call_args
        # tool_choice must force explain_exception
        tool_choice = call_kwargs.kwargs.get("tool_choice") or call_kwargs[1].get("tool_choice")
        assert tool_choice["function"]["name"] == "explain_exception"
        assert result["valid"] is True

    def test_propose_resolution_calls_chat_completions(self, mock_groq_client):
        """
        propose_resolution must invoke chat.completions.create with
        tool_choice forcing the 'propose_resolution' function.
        """
        mock_api_response = _build_mock_response("propose_resolution", _valid_propose_args())
        mock_groq_client.client.chat.completions.create.return_value = mock_api_response

        with patch.object(mock_groq_client, "_execute_with_retry",
                          side_effect=lambda fn, **kw: fn()):
            result = mock_groq_client.propose_resolution(
                record_a={"settled_amount": 980.0, "settled_at": "2026-09-01", "fee": 20.0},
                record_b={"amount": 1000.0, "value_date": "2026-09-01"},
            )

        call_kwargs = mock_groq_client.client.chat.completions.create.call_args
        tool_choice = call_kwargs.kwargs.get("tool_choice") or call_kwargs[1].get("tool_choice")
        assert tool_choice["function"]["name"] == "propose_resolution"
        assert result["valid"] is True

    def test_explain_exception_uses_temperature_zero(self, mock_groq_client):
        """Temperature must be 0.0 — deterministic output required for finance."""
        mock_api_response = _build_mock_response("explain_exception", _valid_explain_args())
        mock_groq_client.client.chat.completions.create.return_value = mock_api_response

        with patch.object(mock_groq_client, "_execute_with_retry",
                          side_effect=lambda fn, **kw: fn()):
            mock_groq_client.explain_exception(
                record_a={"settled_amount": 1000.0, "settled_at": "2026-09-01"},
            )

        call_kwargs = mock_groq_client.client.chat.completions.create.call_args
        temperature = call_kwargs.kwargs.get("temperature") or call_kwargs[1].get("temperature")
        assert temperature == 0.0

    def test_explain_exception_catches_api_exception_returns_invalid(self, mock_groq_client):
        """
        If the whole chain raises (beyond retries), explain_exception must
        return valid=False — not propagate an uncaught exception.
        """
        with patch.object(mock_groq_client, "_execute_with_retry",
                          side_effect=RuntimeError("connection refused")):
            result = mock_groq_client.explain_exception(
                record_a={"settled_amount": 1000.0, "settled_at": "2026-09-01"},
            )

        assert result["valid"] is False
        assert "error" in result
