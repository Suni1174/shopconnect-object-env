"""Unit tests for LLM provider abstraction, MockLLMProvider, and OpenAIProvider.

Verifies Phase 6.1 requirements:
- Provider protocol compatibility.
- Deterministic and sequential mock responses.
- Token usage extraction (including cached and reasoning tokens).
- Missing usage handling without fabrication.
- Error mapping (AuthenticationError, RateLimitError, TimeoutError, ConnectionError, ConfigurationError).
- Responses API parameter enforcement (gpt-6-luna, reasoning effort low, temperature omitted).
- Multi-tool parsing and function_call_output formatting.
- Credential safety and zero network calls during tests.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

import openai
from shopconnect_env.agents.providers import (
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    LLMCompletionResult,
    LLMMessage,
    LLMProvider,
    LLMUsage,
    MockLLMProvider,
    OpenAIProvider,
    ProviderError,
    RateLimitError,
    TimeoutError,
    convert_to_responses_api_tools,
)


# =============================================================================
# 1. PROTOCOL & MOCK PROVIDER TESTS
# =============================================================================

def test_provider_protocol_compatibility():
    """Both MockLLMProvider and OpenAIProvider must satisfy LLMProvider Protocol."""
    mock_p = MockLLMProvider()
    real_p = OpenAIProvider(api_key="test-key")

    assert isinstance(mock_p, LLMProvider)
    assert isinstance(real_p, LLMProvider)


def test_deterministic_mock_responses():
    """MockLLMProvider returns deterministic completion and marks simulated usage."""
    provider = MockLLMProvider()
    messages = [
        LLMMessage(role="system", content="System instruction"),
        LLMMessage(role="user", content="Hello mock model"),
    ]

    result = provider.complete(messages)

    assert isinstance(result, LLMCompletionResult)
    assert result.message.role == "assistant"
    assert "mock_default_response" in result.message.content
    assert result.usage is not None
    assert result.usage.is_provider_reported is False
    assert result.usage.total_tokens == result.usage.input_tokens + result.usage.output_tokens
    assert result.model_identifier == "mock-eval-model"
    assert result.latency_ms > 0
    assert len(provider.call_history) == 1


def test_sequential_mock_responses():
    """MockLLMProvider sequentially pops responses from responses_queue for multi-turn testing."""
    turn_1 = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            tool_calls=(
                {
                    "id": "call_inspect_1",
                    "call_id": "call_inspect_1",
                    "type": "function",
                    "function": {
                        "name": "inspect_recipe",
                        "arguments": '{"recipe_id": "rec_berry_oatmeal"}',
                    },
                },
            ),
        ),
        usage=LLMUsage(input_tokens=100, output_tokens=30, total_tokens=130, is_provider_reported=False),
        model_identifier="mock-seq-model",
    )
    turn_2 = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            content='{"operation": "replace_meal", "status": "final_proposal"}',
        ),
        usage=LLMUsage(input_tokens=150, output_tokens=50, total_tokens=200, is_provider_reported=False),
        model_identifier="mock-seq-model",
    )

    provider = MockLLMProvider(responses=[turn_1, turn_2])

    res1 = provider.complete([LLMMessage(role="user", content="Turn 1 prompt")])
    assert res1.message.tool_calls is not None
    assert res1.message.tool_calls[0]["id"] == "call_inspect_1"

    res2 = provider.complete([LLMMessage(role="user", content="Turn 2 prompt")])
    assert res2.message.content is not None
    assert "final_proposal" in res2.message.content

    assert len(provider.call_history) == 2


def test_canned_mock_responses():
    """MockLLMProvider matches canned prompts based on user message content."""
    canned_res = LLMCompletionResult(
        message=LLMMessage(role="assistant", content="Canned answer for specific prompt"),
        usage=LLMUsage(input_tokens=50, output_tokens=20, total_tokens=70, is_provider_reported=False),
        model_identifier="mock-canned-model",
    )
    provider = MockLLMProvider(canned_responses={"budget query": canned_res})

    res = provider.complete([LLMMessage(role="user", content="Here is my budget query for lunch")])
    assert res.message.content == "Canned answer for specific prompt"


# =============================================================================
# 2. TOKEN USAGE EXTRACTION & MISSING USAGE TESTS
# =============================================================================

def test_token_usage_extraction():
    """OpenAIProvider correctly captures provider-reported input, output, and total tokens."""
    mock_response = SimpleNamespace(
        id="resp_001",
        model="gpt-6-luna",
        status="completed",
        system_fingerprint="fp_gpt6_test",
        created_at=1731250000,
        output=[
            SimpleNamespace(
                type="message",
                content="Plan generated successfully.",
            )
        ],
        output_text="Plan generated successfully.",
        usage=SimpleNamespace(
            input_tokens=210,
            output_tokens=95,
            total_tokens=305,
            input_tokens_details=None,
            output_tokens_details=None,
        ),
    )

    fake_client = MagicMock()
    fake_client.responses.create.return_value = mock_response

    provider = OpenAIProvider(client=fake_client)
    res = provider.complete([LLMMessage(role="user", content="Generate plan")])

    assert res.usage is not None
    assert res.usage.input_tokens == 210
    assert res.usage.output_tokens == 95
    assert res.usage.total_tokens == 305
    assert res.usage.cached_input_tokens is None
    assert res.usage.reasoning_tokens is None
    assert res.usage.is_provider_reported is True
    assert res.model_identifier == "gpt-6-luna"
    assert res.system_fingerprint == "fp_gpt6_test"
    assert res.provider_request_id == "resp_001"


def test_missing_usage_handling():
    """When usage is missing, usage must be None rather than fabricated."""
    mock_response = SimpleNamespace(
        id="resp_no_usage",
        model="gpt-6-luna",
        status="completed",
        output=[SimpleNamespace(type="message", content="Result without usage")],
        output_text="Result without usage",
        usage=None,  # Missing usage object
    )

    fake_client = MagicMock()
    fake_client.responses.create.return_value = mock_response

    provider = OpenAIProvider(client=fake_client)
    res = provider.complete([LLMMessage(role="user", content="Task")])

    assert res.usage is None
    assert res.message.content == "Result without usage"


def test_cached_and_reasoning_token_extraction():
    """OpenAIProvider captures cached_input_tokens and reasoning_tokens when reported."""
    mock_response = SimpleNamespace(
        id="resp_cached",
        model="gpt-6-luna",
        status="completed",
        output_text="Answer with reasoning and caching.",
        output=[SimpleNamespace(type="message", content="Answer with reasoning and caching.")],
        usage=SimpleNamespace(
            input_tokens=500,
            output_tokens=150,
            total_tokens=650,
            input_tokens_details=SimpleNamespace(cached_tokens=200),
            output_tokens_details=SimpleNamespace(reasoning_tokens=80),
        ),
    )

    fake_client = MagicMock()
    fake_client.responses.create.return_value = mock_response

    provider = OpenAIProvider(client=fake_client)
    res = provider.complete([LLMMessage(role="user", content="Calculate budget")])

    assert res.usage is not None
    assert res.usage.input_tokens == 500
    assert res.usage.output_tokens == 150
    assert res.usage.total_tokens == 650
    assert res.usage.cached_input_tokens == 200
    assert res.usage.reasoning_tokens == 80
    assert res.usage.is_provider_reported is True


# =============================================================================
# 3. CREDENTIAL SAFETY & ERROR MAPPING TESTS
# =============================================================================

def test_missing_api_key_handling(monkeypatch):
    """Missing OPENAI_API_KEY environment variable raises ConfigurationError."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = OpenAIProvider()
    with pytest.raises(ConfigurationError) as exc_info:
        provider.complete([LLMMessage(role="user", content="Hello")])

    assert "OPENAI_API_KEY environment variable is not set" in str(exc_info.value)


def test_authentication_and_rate_limit_error_mapping():
    """Authentication and rate limit exceptions are mapped to typed provider errors."""
    # 1. Authentication failure (401)
    client_auth = MagicMock()
    client_auth.responses.create.side_effect = openai.AuthenticationError(
        message="Incorrect API key provided: sk-secret-12345",
        response=MagicMock(status_code=401),
        body={},
    )
    provider_auth = OpenAIProvider(client=client_auth, api_key="sk-secret-12345")
    with pytest.raises(AuthenticationError) as exc_auth:
        provider_auth.complete([LLMMessage(role="user", content="Test auth")])
    assert "sk-secret-12345" not in str(exc_auth.value)

    # 2. Rate limit failure (429)
    client_rate = MagicMock()
    client_rate.responses.create.side_effect = openai.RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body={},
    )
    provider_rate = OpenAIProvider(client=client_rate)
    with pytest.raises(RateLimitError):
        provider_rate.complete([LLMMessage(role="user", content="Test rate limit")])


def test_timeout_and_connection_error_handling():
    """Timeout and network connection failures are mapped to typed provider errors."""
    # 1. Timeout error
    client_timeout = MagicMock()
    client_timeout.responses.create.side_effect = openai.APITimeoutError(
        request=MagicMock()
    )
    provider_timeout = OpenAIProvider(client=client_timeout)
    with pytest.raises(TimeoutError):
        provider_timeout.complete([LLMMessage(role="user", content="Test timeout")])

    # 2. Connection error
    client_conn = MagicMock()
    client_conn.responses.create.side_effect = openai.APIConnectionError(
        request=MagicMock()
    )
    provider_conn = OpenAIProvider(client=client_conn)
    with pytest.raises(ConnectionError):
        provider_conn.complete([LLMMessage(role="user", content="Test connection")])


def test_no_secret_leakage_in_repr_and_errors():
    """API credentials must never appear in object representation or error strings."""
    secret = "sk-proj-SUPER_CONFIDENTIAL_KEY_999"
    provider = OpenAIProvider(api_key=secret)

    # Verify repr does not leak key
    assert secret not in repr(provider)
    assert "gpt-6-luna" in repr(provider)

    # Verify generic provider error redacts key
    client_err = MagicMock()
    client_err.responses.create.side_effect = openai.APIError(
        message=f"Backend failure occurred while processing key {secret}",
        request=MagicMock(),
        body={},
    )
    provider_err = OpenAIProvider(client=client_err, api_key=secret)
    with pytest.raises(ProviderError) as exc_err:
        provider_err.complete([LLMMessage(role="user", content="Test redaction")])

    assert secret not in str(exc_err.value)
    assert "[REDACTED_API_KEY]" in str(exc_err.value)


# =============================================================================
# 4. RESPONSES API REQUEST CONFIGURATION & TOOL CALL TESTS
# =============================================================================

def test_correct_responses_api_request_configuration():
    """Responses API request configuration enforces gpt-6-luna, reasoning effort, and no temperature."""
    fake_client = MagicMock()
    fake_client.responses.create.return_value = SimpleNamespace(
        id="resp_test",
        model="gpt-6-luna",
        output_text="OK",
        output=[],
        usage=None,
    )

    provider = OpenAIProvider(client=fake_client, model="gpt-6-luna", reasoning_effort="low")

    messages = [
        LLMMessage(role="system", content="System instruction context"),
        LLMMessage(role="user", content="What is on the schedule?"),
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "replace_meal",
                "description": "Replace scheduled meal.",
                "parameters": {"type": "object", "properties": {"day": {"type": "string"}}},
            },
        }
    ]

    provider.complete(messages, tools=tools, tool_choice="auto")

    # Inspect call kwargs passed to client.responses.create
    assert fake_client.responses.create.call_count == 1
    call_kwargs = fake_client.responses.create.call_args.kwargs

    # 1. Model identifier
    assert call_kwargs["model"] == "gpt-6-luna"

    # 2. Reasoning effort
    assert call_kwargs["reasoning"] == {"effort": "low"}

    # 3. Explicit architectural rule: NO temperature passed
    assert "temperature" not in call_kwargs

    # 4. Instructions separated from conversation input
    assert call_kwargs["instructions"] == "System instruction context"
    assert call_kwargs["input"] == [{"role": "user", "content": "What is on the schedule?"}]

    # 5. Tool format converted to Responses API schema
    assert len(call_kwargs["tools"]) == 1
    assert call_kwargs["tools"][0]["name"] == "replace_meal"
    assert call_kwargs["tool_choice"] == "auto"


def test_responses_api_multi_tool_call_parsing():
    """Multiple tool calls returned in a single Responses API output are all parsed."""
    call_1 = SimpleNamespace(
        type="function_call",
        id="call_001",
        call_id="call_001",
        name="inspect_recipe",
        arguments='{"recipe_id": "rec_berry_oatmeal"}',
    )
    call_2 = SimpleNamespace(
        type="function_call",
        id="call_002",
        call_id="call_002",
        name="inspect_slot",
        arguments='{"day": "Monday", "meal_type": "breakfast"}',
    )

    fake_client = MagicMock()
    fake_client.responses.create.return_value = SimpleNamespace(
        id="resp_multi_tool",
        model="gpt-6-luna",
        output=[call_1, call_2],
        output_text="",
        usage=None,
    )

    provider = OpenAIProvider(client=fake_client)
    res = provider.complete([LLMMessage(role="user", content="Inspect slot and recipe")])

    assert res.message.tool_calls is not None
    assert len(res.message.tool_calls) == 2

    tc1, tc2 = res.message.tool_calls
    assert tc1["id"] == "call_001"
    assert tc1["function"]["name"] == "inspect_recipe"
    assert tc1["function"]["arguments"] == '{"recipe_id": "rec_berry_oatmeal"}'

    assert tc2["id"] == "call_002"
    assert tc2["function"]["name"] == "inspect_slot"
    assert tc2["function"]["arguments"] == '{"day": "Monday", "meal_type": "breakfast"}'


def test_responses_api_function_call_output_submission():
    """Sending a tool output turn formats input item as type='function_call_output'."""
    fake_client = MagicMock()
    fake_client.responses.create.return_value = SimpleNamespace(
        id="resp_after_tool",
        model="gpt-6-luna",
        output_text="Recipe inspected successfully.",
        output=[SimpleNamespace(type="message", content="Recipe inspected successfully.")],
        usage=None,
    )

    provider = OpenAIProvider(client=fake_client)

    messages = [
        LLMMessage(role="user", content="Inspect oatmeal"),
        LLMMessage(
            role="assistant",
            tool_calls=(
                {
                    "id": "call_abc",
                    "call_id": "call_abc",
                    "type": "function",
                    "function": {"name": "inspect_recipe", "arguments": '{"recipe_id": "rec_berry_oatmeal"}'},
                },
            ),
        ),
        LLMMessage(
            role="tool",
            tool_call_id="call_abc",
            content='{"name": "Berry Oatmeal", "tags": ["vegetarian"]}',
        ),
    ]

    provider.complete(messages)

    call_kwargs = fake_client.responses.create.call_args.kwargs
    input_items = call_kwargs["input"]

    assert len(input_items) == 3
    assert input_items[0] == {"role": "user", "content": "Inspect oatmeal"}
    assert input_items[1]["type"] == "function_call"
    assert input_items[1]["call_id"] == "call_abc"
    assert input_items[2] == {
        "type": "function_call_output",
        "call_id": "call_abc",
        "output": '{"name": "Berry Oatmeal", "tags": ["vegetarian"]}',
    }


def test_convert_to_responses_api_tools_preserves_plain_or_empty():
    """Tool conversion handles None, empty lists, and non-function tools cleanly."""
    assert convert_to_responses_api_tools(None) is None
    assert convert_to_responses_api_tools([]) is None

    plain = [{"type": "custom", "data": 123}]
    assert convert_to_responses_api_tools(plain) == plain
