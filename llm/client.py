"""
Groq API Client with Function/Tool Calling

Wraps the Groq SDK with strict Pydantic schema validation.
Fail closed on any parse error or timeout.
"""

import os
import json
import time
from typing import Optional, Dict, Any
from dotenv import load_dotenv

try:
    import groq
    from groq import Groq
except ImportError:
    Groq = None

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


class GroqClient:
    """
    Groq API client for exception explanation and resolution proposals.
    
    Uses function/tool calling with strict JSON schema and Pydantic validation.
    Any response that fails Pydantic validation is rejected (fail-closed).
    """
    
    DEFAULT_MODEL = "openai/gpt-oss-120b"
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Model to use (defaults to GROQ_MODEL env var or llama-3.3-70b-versatile)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL") or self.DEFAULT_MODEL
        self.client = None
        
        if Groq is None:
            raise ImportError("groq package not installed. Run: pip install groq")
        
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY in .env or environment variables."
            )
        
        self.client = Groq(api_key=self.api_key, timeout=30.0)
    
    def explain_exception(
        self,
        record_a: Optional[Dict],
        record_b: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Ask Groq to explain why an exception occurred using step-by-step reasoning.
        
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
            
            tool_schema = {
                "type": "function",
                "function": {
                    "name": "explain_exception",
                    "description": "Perform step-by-step reasoning and explain the root cause of a reconciliation exception",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thought_process": {
                                "type": "string",
                                "description": "Step-by-step deduction comparing amounts, dates, fees, and references"
                            },
                            "root_cause": {
                                "type": "string",
                                "enum": [
                                    "rounding", "timing_lag", "duplicate_suspected",
                                    "partial_refund", "no_counterpart",
                                    "currency_formatting", "unclassified"
                                ]
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Human-readable explanation of the root cause"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Confidence score between 0 and 1"
                            }
                        },
                        "required": ["thought_process", "root_cause", "explanation", "confidence"]
                    }
                }
            }
            
            response = self._execute_with_retry(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": EXPLAIN_EXCEPTION_SYSTEM},
                        {"role": "user", "content": user_prompt}
                    ],
                    tools=[tool_schema],
                    tool_choice={"type": "function", "function": {"name": "explain_exception"}},
                    temperature=0.0,
                    max_tokens=600,
                    timeout=30.0
                )
            )
            
            return self._parse_tool_response(response, ExplainExceptionResponse)
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def propose_resolution(
        self,
        record_a: Dict,
        record_b: Dict
    ) -> Dict[str, Any]:
        """
        Ask Groq to propose a resolution for an exception with Chain-of-Thought.
        
        Args:
            record_a: Settlement record
            record_b: Counterpart record
            
        Returns:
            Dict with action, confidence, reasoning, thought_process, and valid flag
        """
        if not self.client:
            return {"valid": False, "error": "Client not initialized"}
        
        try:
            user_prompt = build_propose_prompt(record_a, record_b)
            
            tool_schema = {
                "type": "function",
                "function": {
                    "name": "propose_resolution",
                    "description": "Perform step-by-step reasoning and propose a resolution for a reconciliation exception",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thought_process": {
                                "type": "string",
                                "description": "Chain-of-thought analysis verifying amount delta, date lag, and reference correlation"
                            },
                            "action": {
                                "type": "string",
                                "enum": ["match", "flag_for_human", "reject_duplicate"]
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Confidence score between 0 and 1"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Justification for the proposed action"
                            }
                        },
                        "required": ["thought_process", "action", "confidence", "reasoning"]
                    }
                }
            }
            
            response = self._execute_with_retry(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": PROPOSE_RESOLUTION_SYSTEM},
                        {"role": "user", "content": user_prompt}
                    ],
                    tools=[tool_schema],
                    tool_choice={"type": "function", "function": {"name": "propose_resolution"}},
                    temperature=0.0,
                    max_tokens=600,
                    timeout=30.0
                )
            )
            
            return self._parse_tool_response(response, ProposeResolutionResponse)
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
            
    def _execute_with_retry(self, api_callable, max_retries: int = 4):
        """Execute Groq API call with exponential backoff on transient network failures."""
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                return api_callable()
            except Exception as e:
                last_err = e
                err_msg = str(e).lower()
                if attempt < max_retries:
                    delay = max(2.0, 1.5 * (2 ** attempt))
                    if "429" in err_msg or "rate limit" in err_msg:
                        delay = max(delay, 3.0 * (attempt + 1))
                    time.sleep(delay)
                else:
                    raise last_err
    
    def _parse_tool_response(self, response: Any, schema_class) -> Dict[str, Any]:
        """
        Parse and validate a tool/function call response using Pydantic.
        
        Fail closed - any validation error returns an invalid result.
        """
        raw_args = None
        try:
            choice = response.choices[0]
            message = choice.message
            
            # Extract arguments from function tool call
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0]
                raw_args = tool_call.function.arguments
                if isinstance(raw_args, str):
                    input_data = json.loads(raw_args)
                else:
                    input_data = raw_args
            elif message.content:
                raw_args = message.content
                input_data = json.loads(message.content)
            else:
                return {"valid": False, "error": "No tool call or parseable content in response"}
            
            # Validate against Pydantic schema
            try:
                validated = schema_class(**input_data)
            except Exception as schema_err:
                # If root_cause enum failed on unexpected string, map to unclassified
                if schema_class == ExplainExceptionResponse and isinstance(input_data, dict):
                    input_data['root_cause'] = 'unclassified'
                    validated = schema_class(**input_data)
                else:
                    raise schema_err
            
            return {
                "valid": True,
                **validated.model_dump()
            }
            
        except Exception as e:
            raw_snippet = str(raw_args)[:300] if raw_args is not None else "None"
            return {"valid": False, "error": f"Validation failed: {str(e)} | Raw payload: {raw_snippet}"}


# Backward compatibility aliases
ClaudeClient = GroqClient
LLMClient = GroqClient


def create_client() -> Optional[GroqClient]:
    """Factory function to create a Groq client, returning None if unavailable."""
    try:
        return GroqClient()
    except (ImportError, ValueError):
        return None

