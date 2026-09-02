"""
Tests for LLM schema validation.

Verifies that Pydantic schemas correctly validate/reject LLM responses.
"""

import pytest
from pydantic import ValidationError

from llm.schemas import (
    ExplainExceptionResponse,
    ProposeResolutionResponse
)


class TestExplainExceptionSchema:
    """Test the explain_exception response schema."""
    
    def test_valid_response(self):
        """Test valid explanation response with Chain-of-Thought."""
        data = {
            "thought_process": "Comparing timestamps: Settlement lag is within normal T+2 window for UPI.",
            "root_cause": "timing_lag",
            "explanation": "The settlement date is 2 days after the bank transaction, which is within normal T+2 settlement windows.",
            "confidence": 0.87
        }
        
        response = ExplainExceptionResponse(**data)
        assert response.thought_process.startswith("Comparing")
        assert response.root_cause == "timing_lag"
        assert response.confidence == 0.87
    
    def test_all_root_causes(self):
        """Test all valid root cause values."""
        valid_causes = [
            "rounding", "timing_lag", "duplicate_suspected",
            "partial_refund", "no_counterpart", "currency_formatting",
            "unclassified"
        ]
        
        for cause in valid_causes:
            data = {
                "thought_process": "Testing root cause deduction logic for valid causes.",
                "root_cause": cause,
                "explanation": "Test explanation here",
                "confidence": 0.5
            }
            response = ExplainExceptionResponse(**data)
            assert response.root_cause == cause
    
    def test_invalid_root_cause(self):
        """Test that invalid root causes are rejected."""
        data = {
            "thought_process": "Testing invalid root cause deduction logic.",
            "root_cause": "invalid_cause",  # Not in enum
            "explanation": "Test explanation",
            "confidence": 0.5
        }
        
        with pytest.raises(ValidationError):
            ExplainExceptionResponse(**data)
    
    def test_confidence_bounds(self):
        """Test confidence value bounds."""
        # Valid bounds
        assert ExplainExceptionResponse(
            thought_process="Valid deduction reasoning process.",
            root_cause="rounding",
            explanation="Valid test explanation string",
            confidence=0.0
        ).confidence == 0.0
        
        assert ExplainExceptionResponse(
            thought_process="Valid deduction reasoning process.",
            root_cause="rounding",
            explanation="Valid test explanation string",
            confidence=1.0
        ).confidence == 1.0
        
        # Out of bounds should fail
        with pytest.raises(ValidationError):
            ExplainExceptionResponse(
                thought_process="Valid deduction reasoning process.",
                root_cause="rounding",
                explanation="Valid test explanation string",
                confidence=-0.1
            )
        
        with pytest.raises(ValidationError):
            ExplainExceptionResponse(
                thought_process="Valid deduction reasoning process.",
                root_cause="rounding",
                explanation="Valid test explanation string",
                confidence=1.1
            )
    
    def test_explanation_length(self):
        """Test explanation length constraints."""
        # Too short (< 10 chars)
        with pytest.raises(ValidationError):
            ExplainExceptionResponse(
                thought_process="Valid deduction reasoning process.",
                root_cause="rounding",
                explanation="Short",
                confidence=0.5
            )
        
        # Valid length
        response = ExplainExceptionResponse(
            thought_process="Valid deduction reasoning process.",
            root_cause="rounding",
            explanation="This is a valid explanation.",
            confidence=0.5
        )
        assert len(response.explanation) >= 10


class TestProposeResolutionSchema:
    """Test the propose_resolution response schema."""
    
    def test_valid_match_action(self):
        """Test valid match proposal with thought_process."""
        data = {
            "thought_process": "UTR correlation matches exactly and fee calculation matches expected MDR rate.",
            "action": "match",
            "confidence": 0.91,
            "reasoning": "UTR matches exactly and amount difference is within fee tolerance."
        }
        
        response = ProposeResolutionResponse(**data)
        assert response.action == "match"
        assert response.confidence == 0.91
    
    def test_all_actions(self):
        """Test all valid action values."""
        valid_actions = ["match", "flag_for_human", "reject_duplicate"]
        
        for action in valid_actions:
            data = {
                "thought_process": f"Deduction analysis for action {action}.",
                "action": action,
                "confidence": 0.7,
                "reasoning": "Test reasoning for " + action
            }
            response = ProposeResolutionResponse(**data)
            assert response.action == action
    
    def test_invalid_action(self):
        """Test that invalid actions are rejected."""
        data = {
            "thought_process": "Testing invalid action proposal.",
            "action": "auto_approve",  # Not in enum
            "confidence": 0.5,
            "reasoning": "Test"
        }
        
        with pytest.raises(ValidationError):
            ProposeResolutionResponse(**data)
    
    def test_reasoning_required(self):
        """Test that reasoning is required."""
        data = {
            "thought_process": "Testing missing reasoning.",
            "action": "match",
            "confidence": 0.8
            # Missing reasoning
        }
        
        with pytest.raises(ValidationError):
            ProposeResolutionResponse(**data)
