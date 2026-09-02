"""
LLM Layer - Pydantic Response Schemas

Strict schemas for LLM responses. Any response that fails validation
is rejected and treated as an error - fail closed, never fail open.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional


class ExplainExceptionResponse(BaseModel):
    """
    Schema for LLM explanation of an exception.
    
    The LLM analyzes why a record couldn't be matched deterministically
    and proposes a root cause classification after explicit Chain-of-Thought reasoning.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "thought_process": "Evaluating amount delta: Settlement amount is 980, counterpart is 1000. Difference is 20 (2%), which matches standard MDR fee. Dates are within 1 day.",
                "root_cause": "timing_lag",
                "explanation": "The settlement shows a date 2 days after the bank transaction, which is within normal T+2 settlement windows for UPI transactions in India.",
                "confidence": 0.87
            }
        }
    )
    
    thought_process: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Step-by-step mathematical and temporal deduction before classifying root cause"
    )
    
    root_cause: Literal[
        "rounding",
        "timing_lag", 
        "duplicate_suspected",
        "partial_refund",
        "no_counterpart",
        "currency_formatting",
        "unclassified"
    ] = Field(..., description="Primary reason for the matching failure")
    
    explanation: str = Field(
        ..., 
        min_length=10,
        max_length=2000,
        description="Human-readable explanation of the root cause"
    )
    
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="LLM confidence in this explanation (0-1)"
    )


class ProposeResolutionResponse(BaseModel):
    """
    Schema for LLM resolution proposal.
    
    CRITICAL: A proposal with action="match" does NOT commit a match.
    It triggers deterministic re-verification. Only if both the LLM
    and deterministic engine agree does the match get committed.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "thought_process": "Comparing records: Settlement UTR matches counterpart reference. Fee deduction explains the 2% difference. Amount and date criteria satisfy deterministic recheck.",
                "action": "match",
                "confidence": 0.91,
                "reasoning": "Despite the 1.2% amount difference, the UTR matches exactly and the date difference is within normal settlement lag. This appears to be a fee deduction case."
            }
        }
    )
    
    thought_process: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Chain-of-thought analysis verifying amount difference, date lag, and reference overlap"
    )
    
    action: Literal[
        "match",
        "flag_for_human",
        "reject_duplicate"
    ] = Field(..., description="Proposed resolution action")
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM confidence in this resolution (0-1)"
    )
    
    reasoning: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Justification for the proposed action"
    )


class ValidatedResponse(BaseModel):
    """Wrapper indicating a validated LLM response."""
    
    valid: bool = True
    data: Optional[BaseModel] = None
    error: Optional[str] = None
    
    @classmethod
    def success(cls, data: BaseModel) -> "ValidatedResponse":
        return cls(valid=True, data=data)
    
    @classmethod
    def failure(cls, error: str) -> "ValidatedResponse":
        return cls(valid=False, error=error)

