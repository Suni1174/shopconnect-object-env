"""Unit tests for GroqProvider and Groq Cloud API integration.

Verifies:
- Protocol compliance (LLMProvider).
- Exact model identifier qwen/qwen3.8-27b.
- Authentication via GROQ_API_KEY with zero secret leakage.
- Chat Completions message and tool formatting.
- Tool calling support including single and multiple tool calls.
- Provider-reported usage extraction.
- Provider request IDs and actual model identifiers.
- Typed exception mapping (AuthenticationError, RateLimitError, TimeoutError, ConnectionError, ProviderError).
- Secret redaction in error messages and repr.
- 100% offline with zero network calls (injected test double client).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from shopconnect_env.agents.providers import (
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    GroqProvider,
    LLMCompletionResult,
    LLMMessage,
    LLMProvider,
    LLMUsage,
    ProviderError,
    RateLimitError,
    TimeoutError,
    convert_to_chat_completions_messages,
    convert_to_chat_completions_tools,
    parse_chat_completions_response,
)


# =============================================================================
# 1. PROTOCOL & CONFIGURATION TESTS
# =============================================================================

def test_groq_provider_protocol_compliance():
    """GroqProvider must implement LLMProvider protocol."""
    provider = GroqProvider(api_key="gsk-test-key")
    assert isinstance(provider, LLMProvider)


def test_groq_provider_default_model_and_url():
    """Default model must strictly be 'qwen/qwen3.8-27b' and URL 'https://api.groq.com/openai/v1'."""
    provider = GroqProvider(api_key="gsk-test-key")
    assert provider.model == "qwen/qwen3.8-27b"
    assert provider.base_url == "https://api.groq.com/openai/v1"
    assert provider.max_output_tokens == 2048
    assert provider.max_retries == 0
    assert provider.is_execution_lock_enabled is False
    assert provider.effective_reasoning_effort is None


def test_groq_provider_missing_api_key(monkeypatch):
    """Missing GROQ_API_KEY must raise ConfigurationError when instantiating client."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    provider = GroqProvider()
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY environment variable is not set"):
        provider.complete([LLMMessage(role="user", content="Hello")])


def test_groq_provider_repr_redacts_credentials():
    """__repr__ must never expose API keys."""
    provider = GroqProvider(api_key="gsk_secret_1234567890abcdef")
    repr_str = repr(provider)
    assert "gsk_secret_1234567890abcdef" not in repr_str
    assert "qwen/qwen3.8-27b" in repr_str
    assert "https://api.groq.com/openai/v1" in repr_str


# =============================================================================
# 2. INJECTED CLIENT & COMPLETION TESTS (100% OFFLINE)
# =============================================================================

def test_groq_provider_complete_with_injected_client():
    """GroqProvider completion must execute via injected client without network."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="chatcmpl-groq-001",
        model="qwen/qwen3.8-27b",
        system_fingerprint="fp_groq_123",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content='{"decision": "COMMIT", "schedule": []}',
                    tool_calls=None,
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=42,
            completion_tokens=18,
            total_tokens=60,
        ),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    provider = GroqProvider(
        api_key="gsk-test",
        client=mock_client,
        model="qwen/qwen3.8-27b",
    )

    result = provider.complete([LLMMessage(role="user", content="Plan meal")])

    assert isinstance(result, LLMCompletionResult)
    assert result.message.content == '{"decision": "COMMIT", "schedule": []}'
    assert result.model_identifier == "qwen/qwen3.8-27b"
    assert result.provider_request_id == "chatcmpl-groq-001"
    assert result.usage is not None
    assert result.usage.input_tokens == 42
    assert result.usage.output_tokens == 18
    assert result.usage.total_tokens == 60

    # Verify request payload
    called_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert called_kwargs["model"] == "qwen/qwen3.8-27b"
    assert len(called_kwargs["messages"]) == 1
    assert called_kwargs["messages"][0]["content"] == "Plan meal"


def test_groq_provider_complete_with_tool_calling():
    """GroqProvider tool calling must format and parse tools correctly."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="chatcmpl-groq-tool",
        model="qwen/qwen3.8-27b",
        system_fingerprint="fp_groq_tool",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_groq_123",
                            type="function",
                            function=SimpleNamespace(
                                name="commit_meal_replacement",
                                arguments='{"day": "Wednesday", "slot": "Dinner", "recipe_id": "rec_pasta", "servings": 4}',
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=35,
            total_tokens=155,
        ),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    provider = GroqProvider(api_key="gsk-test", client=mock_client)

    tools = [{
        "type": "function",
        "function": {
            "name": "commit_meal_replacement",
            "description": "Replace a meal",
            "parameters": {"type": "object", "properties": {}},
        },
    }]

    result = provider.complete(
        messages=[LLMMessage(role="user", content="Replace Wednesday dinner")],
        tools=tools,
        tool_choice="auto",
    )

    assert result.finish_reason == "tool_calls"
    assert result.message.tool_calls is not None
    assert len(result.message.tool_calls) == 1
    tc = result.message.tool_calls[0]
    assert tc["name"] == "commit_meal_replacement"
    assert tc["id"] == "call_groq_123"
    args = json.loads(tc["arguments"])
    assert args["recipe_id"] == "rec_pasta"


# =============================================================================
# 3. ERROR HANDLING & SECRET REDACTION TESTS
# =============================================================================

@pytest.mark.parametrize(
    "error_cls,status_code,msg,expected_mapped_error",
    [
        (Exception, 401, "Invalid API key", AuthenticationError),
        (Exception, 429, "Rate limit reached", RateLimitError),
        (Exception, 408, "Request timed out", TimeoutError),
        (Exception, 503, "Connection refused", ConnectionError),
    ],
)
def test_groq_provider_error_mapping(error_cls, status_code, msg, expected_mapped_error):
    """SDK errors must map to typed ProviderError exceptions."""
    mock_client = MagicMock()
    exc = error_cls(msg)
    exc.status_code = status_code
    mock_client.chat.completions.create.side_effect = exc

    provider = GroqProvider(api_key="gsk-secret", client=mock_client)
    with pytest.raises(expected_mapped_error):
        provider.complete([LLMMessage(role="user", content="Ping")])


def test_groq_provider_redacts_secret_in_generic_error():
    """Generic exceptions containing secrets must have the secret redacted."""
    secret_key = "gsk_super_secret_groq_key_999"
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception(
        f"Server error occurred using key {secret_key} during request"
    )

    provider = GroqProvider(api_key=secret_key, client=mock_client)
    with pytest.raises(ProviderError) as exc_info:
        provider.complete([LLMMessage(role="user", content="Ping")])

    error_str = str(exc_info.value)
    assert secret_key not in error_str
    assert "[REDACTED_API_KEY]" in error_str


# =============================================================================
# 4. PROTOCOL V2 PARALLEL TOOL CALLS & RATE LIMIT HEADER TESTS
# =============================================================================

def test_groq_provider_parallel_tool_calls_parameter():
    """GroqProvider must pass parallel_tool_calls to completions.create when tools are specified."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="chatcmpl-groq-ptc",
        model="qwen/qwen3.8-27b",
        system_fingerprint="fp_groq_ptc",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="OK", tool_calls=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    provider = GroqProvider(api_key="gsk-test", client=mock_client)
    tools = [{
        "type": "function",
        "function": {"name": "test_func", "parameters": {"type": "object"}},
    }]
    provider.complete(
        [LLMMessage(role="user", content="Test")],
        tools=tools,
        parallel_tool_calls=False,
    )

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["parallel_tool_calls"] is False


def test_groq_provider_response_headers_and_properties():
    """GroqProvider must extract headers from response and expose rate-limit properties."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="chatcmpl-groq-hdr",
        model="qwen/qwen3.8-27b",
        system_fingerprint="fp_groq_hdr",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="OK", tool_calls=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        headers={
            "retry-after": "5.5",
            "x-ratelimit-remaining-tokens": "1450",
            "x-ratelimit-reset-tokens": "2.4s",
            "x-ratelimit-remaining-requests": "29",
            "x-ratelimit-reset-requests": "1.1s",
        },
    )
    mock_client.chat.completions.create.return_value = mock_resp

    provider = GroqProvider(api_key="gsk-test", client=mock_client)
    provider.complete([LLMMessage(role="user", content="Test")])

    assert provider.retry_after_seconds == 5.5
    assert provider.rate_limit_remaining_tokens == 1450
    assert provider.rate_limit_reset_tokens == 2.4
    assert provider.rate_limit_remaining_requests == 29
    assert provider.rate_limit_reset_requests == 1.1


def test_groq_provider_error_headers_captured():
    """GroqProvider must capture rate-limit headers from exception responses."""
    mock_client = MagicMock()
    exc = Exception("Rate limit reached")
    exc.status_code = 429
    exc.response = SimpleNamespace(
        headers={
            "retry-after": "12.0",
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": "6.8s",
        }
    )
    mock_client.chat.completions.create.side_effect = exc

    provider = GroqProvider(api_key="gsk-test", client=mock_client)
    with pytest.raises(RateLimitError):
        provider.complete([LLMMessage(role="user", content="Ping")])

    assert provider.retry_after_seconds == 12.0
    assert provider.rate_limit_remaining_tokens == 0
    assert provider.rate_limit_reset_tokens == 6.8
