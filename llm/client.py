"""
Claude API Client with Tool Calling

Wraps the Anthropic SDK with strict Pydantic schema validation.
Fail closed on any parse error or timeout.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

try:
    from anthropic import Anthropic
    AnthropicMessage = Anthropic.types.Message
except ImportError:
    Anthropic = None
    AnthropicMessage = None

from .schemas import (
    ExplainExceptionResponse,
    ProposeResolutionResponse,
    ValidatedResponse
)
from .prompts import (
    EXPLAIN_EXCEPTION_SYSTEM,
    PROPOSE_RESOLUTION_SYSTEM,
    build_explain_prompt,
    build_propose_prompt
)

load_dotenv()


class ClaudeClient:
    """
    Claude API client for exception explanation and resolution proposals.
    
    Uses tool/function calling with strict JSON schema validation.
    Any response that fails Pydantic validation is rejected.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        """
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.client = None
        
        if Anthropic is None:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY in .env or environment."
            )
        
        self.client = Anthropic(api_key=self.api_key)
    
    def explain_exception(
        self,
        record_a: Optional[Dict],
        record_b: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Ask Claude to explain why an exception occurred.
        
        Args:
            record_a: Settlement record
            record_b: Counterpart record (bank or ledger), may be None
            
        Returns:
            Dict with root_cause, explanation, confidence, and valid flag
        """
        if not self.client:
            return {"valid": False, "error": "Client not initialized"}
        
        try:
            user_prompt = build_explain_prompt(record_a, record_b)
            
            # Use structured output via Pydantic
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system=EXPLAIN_EXCEPTION_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[{
                    "name": "explain_exception",
                    "description": "Explain the root cause of a reconciliation exception",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "root_cause": {
                                "type": "string",
                                "enum": ["rounding", "timing_lag", "duplicate_suspected", 
                                        "partial_refund", "no_counterpart", 
                                        "currency_formatting", "unclassified"]
                            },
                            "explanation": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["root_cause", "explanation", "confidence"]
                    }
                }],
                tool_choice={"type": "tool", "name": "explain_exception"}
            )
            
            # Parse the tool response
            result = self._parse_tool_response(response, ExplainExceptionResponse)
            return result
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def propose_resolution(
        self,
        record_a: Dict,
        record_b: Dict
    ) -> Dict[str, Any]:
        """
        Ask Claude to propose a resolution for an exception.
        
        Args:
            record_a: Settlement record
            record_b: Counterpart record
            
        Returns:
            Dict with action, confidence, reasoning, and valid flag
        """
        if not self.client:
            return {"valid": False, "error": "Client not initialized"}
        
        try:
            user_prompt = build_propose_prompt(record_a, record_b)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system=PROPOSE_RESOLUTION_SYSTEM,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[{
                    "name": "propose_resolution",
                    "description": "Propose a resolution for a reconciliation exception",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["match", "flag_for_human", "reject_duplicate"]
                            },
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "reasoning": {"type": "string"}
                        },
                        "required": ["action", "confidence", "reasoning"]
                    }
                }],
                tool_choice={"type": "tool", "name": "propose_resolution"}
            )
            
            # Parse the tool response
            result = self._parse_tool_response(response, ProposeResolutionResponse)
            return result
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def _parse_tool_response(self, response: 'AnthropicMessage', schema_class) -> Dict[str, Any]:
        """
        Parse and validate a tool response using Pydantic.
        
        Fail closed - any validation error returns invalid result.
        """
        try:
            # Extract tool use from response
            tool_use = None
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_use = content_block
                    break
            
            if not tool_use:
                return {"valid": False, "error": "No tool use in response"}
            
            input_data = tool_use.input
            
            # Validate against Pydantic schema
            validated = schema_class(**input_data)
            
            return {
                "valid": True,
                **validated.model_dump()
            }
            
        except Exception as e:
            return {"valid": False, "error": f"Validation failed: {str(e)}"}


def create_client() -> Optional[ClaudeClient]:
    """Factory function to create a Claude client, returning None if unavailable."""
    try:
        return ClaudeClient()
    except (ImportError, ValueError):
        return None
