"""
Tests for llm/schemas.py — Pydantic response schema validation.

Covers:
- Valid ExplainExceptionResponse and ProposeResolutionResponse payloads parse correctly.
- Invalid / malformed LLM output (missing required field, wrong enum value, wrong type)
  is rejected with a ValidationError — fail closed, never silently accepted.
- Boundary cases: confidence outside [0, 1], empty structured_reasoning,
  action value not in allowed set.
- ValidatedResponse factory methods (.success() / .failure()).
"""

import pytest
from pydantic import ValidationError

from llm.schemas import (
    ExplainExceptionResponse,
    ProposeResolutionResponse,
    ValidatedResponse,
)


# ---------------------------------------------------------------------------
# Fixtures — canonical valid payloads
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_explain_payload():
    """Minimal valid payload for ExplainExceptionResponse."""
    return {
        "structured_reasoning": (
            "Evaluating amount delta: Settlement is 980, counterpart is 1000. "
            "Difference is 2%, matching MDR fee. Date within T+2."
        ),
        "root_cause": "timing_lag",
        "explanation": "Settlement date is two days after bank transaction, within T+2 window.",
        "confidence": 0.87,
    }


@pytest.fixture
def valid_propose_payload():
    """Minimal valid payload for ProposeResolutionResponse."""
    return {
        "structured_reasoning": (
            "UTR matches exactly. Fee deduction explains 2% delta. "
            "Date difference is within normal settlement lag."
        ),
        "action": "match",
        "confidence": 0.91,
        "reasoning": "Fee-adjusted amount aligns and UTR is identical. Safe to match.",
    }


# ---------------------------------------------------------------------------
# ExplainExceptionResponse — valid cases
# ---------------------------------------------------------------------------

class TestExplainExceptionResponseValid:

    def test_full_valid_payload_parses(self, valid_explain_payload):
        obj = ExplainExceptionResponse(**valid_explain_payload)
        assert obj.root_cause == "timing_lag"
        assert obj.confidence == pytest.approx(0.87)
        assert len(obj.structured_reasoning) >= 10
        assert len(obj.explanation) >= 10

    def test_all_allowed_root_causes_are_accepted(self, valid_explain_payload):
        """Every value in the root_cause Literal must be accepted."""
        allowed = [
            "rounding",
            "timing_lag",
            "duplicate_suspected",
            "partial_refund",
            "no_counterpart",
            "currency_formatting",
            "unclassified",
        ]
        for cause in allowed:
            payload = {**valid_explain_payload, "root_cause": cause}
            obj = ExplainExceptionResponse(**payload)
            assert obj.root_cause == cause

    def test_confidence_at_zero_boundary(self, valid_explain_payload):
        obj = ExplainExceptionResponse(**{**valid_explain_payload, "confidence": 0.0})
        assert obj.confidence == 0.0

    def test_confidence_at_one_boundary(self, valid_explain_payload):
        obj = ExplainExceptionResponse(**{**valid_explain_payload, "confidence": 1.0})
        assert obj.confidence == 1.0

    def test_model_dump_round_trips(self, valid_explain_payload):
        obj = ExplainExceptionResponse(**valid_explain_payload)
        dumped = obj.model_dump()
        reconstructed = ExplainExceptionResponse(**dumped)
        assert reconstructed.root_cause == obj.root_cause
        assert reconstructed.confidence == obj.confidence


# ---------------------------------------------------------------------------
# ExplainExceptionResponse — invalid / rejection cases
# ---------------------------------------------------------------------------

class TestExplainExceptionResponseInvalid:

    def test_missing_root_cause_raises(self, valid_explain_payload):
        payload = {k: v for k, v in valid_explain_payload.items() if k != "root_cause"}
        with pytest.raises(ValidationError):
            ExplainExceptionResponse(**payload)

    def test_missing_explanation_raises(self, valid_explain_payload):
        payload = {k: v for k, v in valid_explain_payload.items() if k != "explanation"}
        with pytest.raises(ValidationError):
            ExplainExceptionResponse(**payload)

    def test_missing_structured_reasoning_raises(self, valid_explain_payload):
        payload = {k: v for k, v in valid_explain_payload.items() if k != "structured_reasoning"}
        with pytest.raises(ValidationError):
            ExplainExceptionResponse(**payload)

    def test_missing_confidence_raises(self, valid_explain_payload):
        payload = {k: v for k, v in valid_explain_payload.items() if k != "confidence"}
        with pytest.raises(ValidationError):
            ExplainExceptionResponse(**payload)

    def test_wrong_root_cause_enum_raises(self, valid_explain_payload):
        """An arbitrary string not in the Literal set must be rejected."""
        payload = {**valid_explain_payload, "root_cause": "made_up_cause"}
        with pytest.raises(ValidationError):
            ExplainExceptionResponse(**payload)

    def test_empty_string_root_cause_raises(self, valid_explain_payload):
        payload = {**valid_explain_payload, "root_cause": ""}
        with pytest.raises(ValidationError):
            ExplainExceptionResponse(**payload)

    def test_confidence_above_one_raises(self, valid_explain_payload):
        payload = {**valid_explain_payload, "confidence": 1.5}
        with pytest.raises(ValidationError):
            ExplainExceptionResponse(**payload)

    def test_confidence_below_zero_raises(self, valid_explain_payload):
        payload = {**valid_explain_payload, "confidence": -0.1}
        with pytest.raises(ValidationError):
            ExplainExceptionResponse(**payload)

    def test_confidence_as_string_raises(self, valid_explain_payload):
        """Wrong type — string instead of float — must be rejected."""
        payload = {**valid_explain_payload, "confidence": "high"}
        with pytest.raises(ValidationError):
            ExplainExceptionResponse(**payload)

    def test_structured_reasoning_too_short_raises(self, valid_explain_payload):
        """structured_reasoning has min_length=10; a short string must fail."""
        payload = {**valid_explain_payload, "structured_reasoning": "short"}
        with pytest.raises(ValidationError):
            ExplainExceptionResponse(**payload)

    def test_explanation_too_short_raises(self, valid_explain_payload):
        """explanation has min_length=10; a short string must fail."""
        payload = {**valid_explain_payload, "explanation": "tiny"}
        with pytest.raises(ValidationError):
            ExplainExceptionResponse(**payload)

    def test_extra_keys_do_not_raise(self, valid_explain_payload):
        """
        Pydantic v2 by default strips extra keys rather than rejecting them.
        We document this explicitly: extra fields are ignored, not silently
        promoted to attributes. This is intentional — the LLM may include
        incidental annotation keys that we safely discard.
        """
        payload = {**valid_explain_payload, "unexpected_field": "some value"}
        obj = ExplainExceptionResponse(**payload)
        assert not hasattr(obj, "unexpected_field")


# ---------------------------------------------------------------------------
# ProposeResolutionResponse — valid cases
# ---------------------------------------------------------------------------

class TestProposeResolutionResponseValid:

    def test_full_valid_payload_parses(self, valid_propose_payload):
        obj = ProposeResolutionResponse(**valid_propose_payload)
        assert obj.action == "match"
        assert obj.confidence == pytest.approx(0.91)

    def test_all_allowed_actions_accepted(self, valid_propose_payload):
        """Every value in the action Literal must be accepted."""
        for action in ["match", "flag_for_human", "reject_duplicate"]:
            payload = {**valid_propose_payload, "action": action}
            obj = ProposeResolutionResponse(**payload)
            assert obj.action == action

    def test_confidence_boundary_zero(self, valid_propose_payload):
        obj = ProposeResolutionResponse(**{**valid_propose_payload, "confidence": 0.0})
        assert obj.confidence == 0.0

    def test_confidence_boundary_one(self, valid_propose_payload):
        obj = ProposeResolutionResponse(**{**valid_propose_payload, "confidence": 1.0})
        assert obj.confidence == 1.0

    def test_model_dump_round_trips(self, valid_propose_payload):
        obj = ProposeResolutionResponse(**valid_propose_payload)
        dumped = obj.model_dump()
        reconstructed = ProposeResolutionResponse(**dumped)
        assert reconstructed.action == obj.action


# ---------------------------------------------------------------------------
# ProposeResolutionResponse — invalid / rejection cases
# ---------------------------------------------------------------------------

class TestProposeResolutionResponseInvalid:

    def test_missing_action_raises(self, valid_propose_payload):
        payload = {k: v for k, v in valid_propose_payload.items() if k != "action"}
        with pytest.raises(ValidationError):
            ProposeResolutionResponse(**payload)

    def test_missing_reasoning_raises(self, valid_propose_payload):
        payload = {k: v for k, v in valid_propose_payload.items() if k != "reasoning"}
        with pytest.raises(ValidationError):
            ProposeResolutionResponse(**payload)

    def test_missing_confidence_raises(self, valid_propose_payload):
        payload = {k: v for k, v in valid_propose_payload.items() if k != "confidence"}
        with pytest.raises(ValidationError):
            ProposeResolutionResponse(**payload)

    def test_action_not_in_allowed_set_raises(self, valid_propose_payload):
        """action value outside Literal set must be rejected."""
        for bad_action in ["approve", "defer", "accept", ""]:
            payload = {**valid_propose_payload, "action": bad_action}
            with pytest.raises(ValidationError):
                ProposeResolutionResponse(**payload)

    def test_confidence_above_one_raises(self, valid_propose_payload):
        payload = {**valid_propose_payload, "confidence": 1.01}
        with pytest.raises(ValidationError):
            ProposeResolutionResponse(**payload)

    def test_confidence_below_zero_raises(self, valid_propose_payload):
        payload = {**valid_propose_payload, "confidence": -0.01}
        with pytest.raises(ValidationError):
            ProposeResolutionResponse(**payload)

    def test_confidence_as_string_raises(self, valid_propose_payload):
        payload = {**valid_propose_payload, "confidence": "very_high"}
        with pytest.raises(ValidationError):
            ProposeResolutionResponse(**payload)

    def test_structured_reasoning_too_short_raises(self, valid_propose_payload):
        payload = {**valid_propose_payload, "structured_reasoning": "tiny"}
        with pytest.raises(ValidationError):
            ProposeResolutionResponse(**payload)

    def test_reasoning_too_short_raises(self, valid_propose_payload):
        payload = {**valid_propose_payload, "reasoning": "ok"}
        with pytest.raises(ValidationError):
            ProposeResolutionResponse(**payload)


# ---------------------------------------------------------------------------
# ValidatedResponse factory methods
# ---------------------------------------------------------------------------

class TestValidatedResponse:

    def test_success_factory_sets_valid_true(self, valid_explain_payload):
        inner = ExplainExceptionResponse(**valid_explain_payload)
        result = ValidatedResponse.success(inner)
        assert result.valid is True
        assert result.data is inner
        assert result.error is None

    def test_failure_factory_sets_valid_false(self):
        result = ValidatedResponse.failure("JSON decode failed")
        assert result.valid is False
        assert result.error == "JSON decode failed"
        assert result.data is None

    def test_default_instance_is_valid(self):
        result = ValidatedResponse()
        assert result.valid is True

    def test_failure_with_empty_error_string(self):
        result = ValidatedResponse.failure("")
        assert result.valid is False
        assert result.error == ""
