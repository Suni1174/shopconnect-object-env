"""Unit tests for OpenRouterProvider and OpenRouter API integration.

Verifies:
- Protocol compliance (LLMProvider).
- Exact model identifier qwen/qwen3.8-27b:free with mandatory ':free' suffix.
- Rejection of non-free models and dynamic routing 'openrouter/free'.
- Authentication via OPENROUTER_API_KEY with zero secret leakage.
- Chat Completions message and tool formatting.
- Tool calling support including multiple tool calls in a single response.
- Function-call response correlation (tool role with tool_call_id).
- Provider-reported usage extraction and missing usage preservation.
- Provider request IDs and actual model identifiers.
- Typed exception mapping (AuthenticationError, RateLimitError, TimeoutError, ConnectionError).
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
    LLMCompletionResult,
    LLMMessage,
    LLMProvider,
    LLMUsage,
    OpenRouterProvider,
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

def test_openrouter_provider_protocol_compliance():
    """OpenRouterProvider must implement LLMProvider protocol."""
    provider = OpenRouterProvider(api_key="test-key")
    assert isinstance(provider, LLMProvider)


def test_openrouter_provider_default_model():
    """Default model must strictly be 'qwen/qwen3.8-27b:free'."""
    provider = OpenRouterProvider(api_key="test-key")
    assert provider.model == "qwen/qwen3.8-27b:free"
    assert provider.base_url == "https://openrouter.ai/api/v1"
    assert provider.max_output_tokens == 2048


def test_openrouter_provider_requires_free_suffix():
    """Model identifiers without ':free' suffix must raise ConfigurationError."""
    with pytest.raises(ConfigurationError, match="must have ':free' suffix"):
        OpenRouterProvider(api_key="test-key", model="qwen/qwen3.8-27b")

    with pytest.raises(ConfigurationError, match="must have ':free' suffix"):
        OpenRouterProvider(api_key="test-key", model="meta-llama/llama-3-70b-instruct")


def test_openrouter_provider_rejects_dynamic_routing():
    """Dynamic model routing 'openrouter/free' must be explicitly rejected."""
    with pytest.raises(ConfigurationError, match="Dynamic routing model 'openrouter/free' is prohibited"):
        OpenRouterProvider(api_key="test-key", model="openrouter/free")


def test_openrouter_provider_missing_api_key(monkeypatch):
    """Missing OPENROUTER_API_KEY must raise ConfigurationError when instantiating client."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    provider = OpenRouterProvider()
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY environment variable is not set"):
        provider.complete([LLMMessage(role="user", content="Hello")])


def test_openrouter_provider_repr_redacts_credentials():
    """__repr__ must never expose API keys."""
    provider = OpenRouterProvider(api_key="sk-or-v1-secret-key-123456789")
    repr_str = repr(provider)
    assert "sk-or-v1-secret-key-123456789" not in repr_str
    assert "qwen/qwen3.8-27b:free" in repr_str


# =============================================================================
# 2. MESSAGE & TOOL CONVERSION TESTS
# =============================================================================

def test_convert_to_chat_completions_messages():
    """Messages must convert to standard OpenAI Chat Completions dicts."""
    messages = [
        LLMMessage(role="system", content="System instruction"),
        LLMMessage(role="user", content="User question"),
        LLMMessage(
            role="assistant",
            content="Calling a tool",
            tool_calls=(
                {
                    "id": "call_123",
                    "function": {"name": "inspect_object", "arguments": '{"target": "p1"}'},
                },
            ),
        ),
        LLMMessage(
            role="tool",
            content='{"status": "ok"}',
            tool_call_id="call_123",
        ),
    ]

    converted = convert_to_chat_completions_messages(messages)
    assert len(converted) == 4
    assert converted[0] == {"role": "system", "content": "System instruction"}
    assert converted[1] == {"role": "user", "content": "User question"}
    assert converted[2]["role"] == "assistant"
    assert converted[2]["content"] == "Calling a tool"
    assert len(converted[2]["tool_calls"]) == 1
    assert converted[2]["tool_calls"][0]["id"] == "call_123"
    assert converted[2]["tool_calls"][0]["function"]["name"] == "inspect_object"
    assert converted[3] == {"role": "tool", "tool_call_id": "call_123", "content": '{"status": "ok"}'}


def test_convert_to_chat_completions_tools():
    """Tool definitions must convert to OpenAI Chat Completions function tool format."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect_object",
                "description": "Inspect object fields",
                "parameters": {"type": "object", "properties": {"target": {"type": "string"}}},
            },
        }
    ]
    converted = convert_to_chat_completions_tools(tools)
    assert converted is not None
    assert len(converted) == 1
    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["name"] == "inspect_object"
    assert "parameters" in converted[0]["function"]


# =============================================================================
# 3. RESPONSE PARSING & TOOL CALLING TESTS
# =============================================================================

def test_parse_chat_completions_response_single_tool_call():
    """Single tool call in response must be parsed correctly."""
    mock_resp = SimpleNamespace(
        id="gen-12345",
        model="qwen/qwen3.8-27b:free",
        system_fingerprint="fp_abc",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_001",
                            function=SimpleNamespace(
                                name="inspect_object",
                                arguments='{"target": "meal_plan.wednesday"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=25,
            total_tokens=125,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=10),
        ),
    )

    result = parse_chat_completions_response(mock_resp, latency_ms=45.0, default_model="qwen/qwen3.8-27b:free")
    assert result.message.role == "assistant"
    assert result.message.tool_calls is not None
    assert len(result.message.tool_calls) == 1
    tc = result.message.tool_calls[0]
    assert tc["id"] == "call_001"
    assert tc["name"] == "inspect_object"
    assert json.loads(tc["arguments"]) == {"target": "meal_plan.wednesday"}

    assert result.usage is not None
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 25
    assert result.usage.total_tokens == 125
    assert result.usage.cached_input_tokens == 20
    assert result.usage.reasoning_tokens == 10
    assert result.usage.is_provider_reported is True
    assert result.provider_request_id == "gen-12345"
    assert result.model_identifier == "qwen/qwen3.8-27b:free"
    assert result.finish_reason == "tool_calls"


def test_parse_chat_completions_response_multiple_tool_calls():
    """Multiple tool calls in a single response must all be preserved."""
    mock_resp = SimpleNamespace(
        id="gen-multi",
        model="qwen/qwen3.8-27b:free",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="tool_a", arguments='{"k1": "v1"}'),
                        ),
                        SimpleNamespace(
                            id="call_2",
                            function=SimpleNamespace(name="tool_b", arguments='{"k2": "v2"}'),
                        ),
                    ],
                ),
            )
        ],
        usage=None,
    )

    result = parse_chat_completions_response(mock_resp, latency_ms=30.0, default_model="qwen/qwen3.8-27b:free")
    assert result.message.tool_calls is not None
    assert len(result.message.tool_calls) == 2
    assert result.message.tool_calls[0]["name"] == "tool_a"
    assert result.message.tool_calls[1]["name"] == "tool_b"
    assert result.usage is None  # Missing usage stays None


def test_parse_chat_completions_response_missing_usage():
    """Missing usage metadata must stay None and never be fabricated."""
    mock_resp = SimpleNamespace(
        id="gen-no-usage",
        model="qwen/qwen3.8-27b:free",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="Hello world", tool_calls=None),
            )
        ],
        usage=None,
    )

    result = parse_chat_completions_response(mock_resp, latency_ms=20.0, default_model="qwen/qwen3.8-27b:free")
    assert result.usage is None
    assert result.message.content == "Hello world"


def test_parse_chat_completions_derived_total_tokens():
    """When total_tokens is omitted, derived_total_tokens is set and total_tokens stays None."""
    mock_resp = SimpleNamespace(
        id="gen-derived",
        model="qwen/qwen3.8-27b:free",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="Done", tool_calls=None),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=80,
            completion_tokens=20,
            total_tokens=None,
        ),
    )

    result = parse_chat_completions_response(mock_resp, latency_ms=15.0, default_model="qwen/qwen3.8-27b:free")
    assert result.usage is not None
    assert result.usage.input_tokens == 80
    assert result.usage.output_tokens == 20
    assert result.usage.total_tokens is None
    assert result.usage.derived_total_tokens == 100


# =============================================================================
# 4. INJECTED CLIENT & ERROR HANDLING TESTS
# =============================================================================

def test_openrouter_provider_complete_with_injected_client():
    """OpenRouterProvider.complete executes cleanly with injected client."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="gen-test-inj",
        model="qwen/qwen3.8-27b:free",
        system_fingerprint="fp_test",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="Simulation OK", tool_calls=None),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
        ),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    provider = OpenRouterProvider(api_key="test-key", client=mock_client)
    res = provider.complete([LLMMessage(role="user", content="Ping")])

    assert res.message.content == "Simulation OK"
    assert res.provider_request_id == "gen-test-inj"
    assert res.usage.total_tokens == 60
    assert mock_client.chat.completions.create.called


@pytest.mark.parametrize(
    "exc,expected_type",
    [
        (Exception("Unauthorized: 401 invalid key"), AuthenticationError),
        (Exception("Rate limit reached 429 quota exceeded"), RateLimitError),
        (Exception("Request timed out"), TimeoutError),
        (Exception("Connection refused / failed to connect"), ConnectionError),
    ],
)
def test_openrouter_provider_error_mapping(exc, expected_type):
    """SDK exceptions must map to typed ProviderError exceptions."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = exc

    provider = OpenRouterProvider(api_key="secret-api-key", client=mock_client)
    with pytest.raises(expected_type):
        provider.complete([LLMMessage(role="user", content="Test")])


def test_openrouter_provider_redacts_secret_in_generic_error():
    """Generic ProviderError messages must redact the API key."""
    secret = "sk-or-v1-my-secret-key-9999"
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError(f"Unknown upstream error on key {secret}")

    provider = OpenRouterProvider(api_key=secret, client=mock_client)
    with pytest.raises(ProviderError) as exc_info:
        provider.complete([LLMMessage(role="user", content="Test")])

    err_msg = str(exc_info.value)
    assert secret not in err_msg
    assert "[REDACTED_API_KEY]" in err_msg
