"""
Tests for GroqClient LLM integration.

Verifies:
- Initialization with GROQ_API_KEY and custom model
- Fail-closed error handling on invalid responses
- Function/tool call schema structure
- Pydantic schema validation of tool call arguments
"""

import pytest
import os
from unittest.mock import MagicMock, patch

from llm.client import GroqClient, create_client
from llm.schemas import ExplainExceptionResponse, ProposeResolutionResponse


class TestGroqClientInitialization:
    """Test client initialization and configuration."""
    
    def test_init_with_explicit_key(self):
        """Client initializes with explicitly provided API key."""
        client = GroqClient(api_key="gsk_test_key_12345", model="llama-3.3-70b-versatile")
        assert client.api_key == "gsk_test_key_12345"
        assert client.model == "llama-3.3-70b-versatile"
        assert client.client is not None
        
    def test_init_missing_key_raises(self, monkeypatch):
        """Client raises ValueError when no API key is provided or found in environment."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="Groq API key not found"):
            GroqClient(api_key=None)
            
    def test_create_client_factory_none_when_no_key(self, monkeypatch):
        """create_client returns None gracefully if no API key is set."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client = create_client()
        assert client is None


class TestGroqResponseParsing:
    """Test parse and validation logic for tool responses."""
    
    def test_parse_valid_tool_call(self):
        """Successfully parses and validates a well-formed tool call."""
        client = GroqClient(api_key="gsk_test")
        
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.arguments = (
            '{"thought_process": "Amount matches after 2% fee deduction. Date lag is 1 day.", '
            '"root_cause": "rounding", '
            '"explanation": "MDR fee deduction of 2% explains the amount difference.", '
            '"confidence": 0.95}'
        )
        mock_choice = MagicMock()
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_response.choices = [mock_choice]
        
        res = client._parse_tool_response(mock_response, ExplainExceptionResponse)
        assert res["valid"] is True
        assert res["root_cause"] == "rounding"
        assert res["confidence"] == 0.95
        assert "MDR fee" in res["explanation"]
        
    def test_parse_invalid_tool_call_fails_closed(self):
        """Fails closed when tool response violates schema (e.g. invalid enum or out of bounds)."""
        client = GroqClient(api_key="gsk_test")
        
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.arguments = (
            '{"thought_process": "Short", '
            '"root_cause": "non_existent_category", '
            '"explanation": "Too short", '
            '"confidence": 1.5}'
        )
        mock_choice = MagicMock()
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_response.choices = [mock_choice]
        
        res = client._parse_tool_response(mock_response, ExplainExceptionResponse)
        assert res["valid"] is False
        assert "Validation failed" in res["error"]

    def test_parse_malformed_json_fails_closed(self):
        """Fails closed when tool response is malformed JSON."""
        client = GroqClient(api_key="gsk_test")
        
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.arguments = "{broken json..."
        mock_choice = MagicMock()
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_response.choices = [mock_choice]
        
        res = client._parse_tool_response(mock_response, ExplainExceptionResponse)
        assert res["valid"] is False
        assert "Validation failed" in res["error"]
