"""
LLM Layer - Pydantic Response Schemas

Strict schemas for LLM responses. Any response that fails validation
is rejected and treated as an error - fail closed, never fail open.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class ExplainExceptionResponse(BaseModel):
    """
    Schema for LLM explanation of an exception.
    
    The LLM analyzes why a record couldn't be matched deterministically
    and proposes a root cause classification.
    """
    
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
        max_length=500,
        description="Human-readable explanation of the root cause"
    )
    
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="LLM confidence in this explanation (0-1)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "root_cause": "timing_lag",
                "explanation": "The settlement shows a date 2 days after the bank transaction, which is within normal T+2 settlement windows for UPI transactions in India.",
                "confidence": 0.87
            }
        }


class ProposeResolutionResponse(BaseModel):
    """
    Schema for LLM resolution proposal.
    
    CRITICAL: A proposal with action="match" does NOT commit a match.
    It triggers deterministic re-verification. Only if both the LLM
    and deterministic engine agree does the match get committed.
    """
    
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
        max_length=500,
        description="Justification for the proposed action"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "action": "match",
                "confidence": 0.91,
                "reasoning": "Despite the 1.2% amount difference, the UTR matches exactly and the date difference is within normal settlement lag. This appears to be a fee deduction case."
            }
        }


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
