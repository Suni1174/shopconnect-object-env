"""Unit tests for OllamaProvider local inference integration.

Verifies:
- Protocol compliance (LLMProvider).
- Exact model identifier qwen3.5:0.8b.
- Default endpoint http://localhost:11434/v1 and configurable base_url.
- No API key requirement (zero credentials needed, never requests Ollama Cloud key).
- Conservative context length: 4096 tokens configured via extra_body options num_ctx.
- Thinking mode disabled via documented OpenAI-compatible reasoning_effort="none".
- Does NOT rely on or send invalid think=False in extra_body for /v1/chat/completions.
- Wire-level serialization verifies reasoning_effort reaches endpoint in HTTP JSON body.
- Sequential execution lock (one local request at a time).
- System, user, assistant, and tool message conversion.
- Tool calling, multiple tool calls, and tool response correlation.
- Usage extraction, missing usage, and missing total tokens handling.
- Local runtime metadata and duration extraction (prompt_eval_duration, eval_duration, total_duration, digest).
- Reasoning metadata distinguished from final user-facing content (never exposing hidden thoughts).
- finish_reason retention.
- Local inference provenance (deployment_mode="local_inference").
- Clean detection of offline Ollama server with instructions (ollama serve, ollama pull qwen3.5:0.8b).
- Zero remote cloud fallback invariant.
- Monetary cost evaluates to None (hardware/energy unmeasured, never falsely reported as $0.00).
- 100% offline with zero external network calls (injected test double client).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock
import httpx
import openai
import pytest

from shopconnect_env.agents.providers import (
    ConnectionError,
    LLMCompletionResult,
    LLMMessage,
    LLMProvider,
    LLMUsage,
    OllamaProvider,
    ProviderError,
    TimeoutError,
    convert_to_chat_completions_messages,
    convert_to_chat_completions_tools,
    parse_chat_completions_response,
)
from shopconnect_env.evaluation.metrics import calculate_cost, create_ollama_local_pricing


# =============================================================================
# 1. PROTOCOL & CONFIGURATION TESTS
# =============================================================================

def test_ollama_provider_protocol_compliance():
    """OllamaProvider must implement LLMProvider protocol."""
    provider = OllamaProvider()
    assert isinstance(provider, LLMProvider)


def test_ollama_provider_default_configuration():
    """OllamaProvider defaults to qwen3.5:0.8b, localhost endpoint, 4096 context, 180s timeout."""
    provider = OllamaProvider()
    assert provider.model == "qwen3.5:0.8b"
    assert provider.base_url == "http://localhost:11434/v1"
    assert provider.context_length == 4096
    assert provider.timeout_seconds == 180.0
    assert provider.enable_thinking is False
    assert provider.max_output_tokens == 2048


def test_ollama_provider_configurable_parameters():
    """Base URL, context length, timeout, and thinking mode must be configurable."""
    provider = OllamaProvider(
        model="qwen3.5:0.8b",
        base_url="http://127.0.0.1:11435/v1/",
        context_length=2048,
        timeout_seconds=60.0,
        enable_thinking=True,
    )
    assert provider.base_url == "http://127.0.0.1:11435/v1"
    assert provider.context_length == 2048
    assert provider.timeout_seconds == 60.0
    assert provider.enable_thinking is True


def test_ollama_provider_no_api_key_required(monkeypatch):
    """OllamaProvider does not require or look for an API key environment variable."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    provider = OllamaProvider()
    client = provider._get_client()
    assert client is not None


def test_ollama_provider_repr():
    """__repr__ cleanly displays model, base_url, context_length, and enable_thinking."""
    provider = OllamaProvider(context_length=4096, enable_thinking=False)
    repr_str = repr(provider)
    assert "qwen3.5:0.8b" in repr_str
    assert "http://localhost:11434/v1" in repr_str
    assert "4096" in repr_str
    assert "enable_thinking=False" in repr_str


# =============================================================================
# 2. MESSAGE AND REQUEST PAYLOAD CONVERSION TESTS
# =============================================================================

def test_ollama_provider_request_sends_reasoning_effort_none():
    """complete() must pass reasoning_effort='none' to client to disable thinking on /v1/chat/completions."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="ollama-123",
        model="qwen3.5:0.8b",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="Hello", tool_calls=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=40, completion_tokens=15, total_tokens=55),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    provider = OllamaProvider(context_length=4096, client=mock_client)
    res = provider.complete([LLMMessage(role="user", content="Hi")])

    assert res.message.content == "Hello"
    assert res.model_identifier == "qwen3.5:0.8b"
    assert res.raw_response.get("deployment_mode") == "local_inference"

    # Verify call kwargs passed to client.chat.completions.create
    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["model"] == "qwen3.5:0.8b"
    assert kwargs["extra_body"]["options"]["num_ctx"] == 4096


def test_ollama_provider_does_not_rely_on_think_extra_body():
    """The wrong think=False extra-body parameter is NOT sent or relied upon for /v1/chat/completions."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="ollama-124",
        model="qwen3.5:0.8b",
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(role="assistant", content="OK"))],
        usage=None,
    )
    mock_client.chat.completions.create.return_value = mock_resp

    provider = OllamaProvider(client=mock_client)
    provider.complete([LLMMessage(role="user", content="Ping")])

    kwargs = mock_client.chat.completions.create.call_args[1]
    extra_body = kwargs.get("extra_body", {})
    assert "think" not in extra_body


def test_ollama_provider_serialized_http_body_contains_reasoning_effort_none():
    """Wire-level check: OpenAI client serializes reasoning_effort='none' in HTTP request body."""
    last_request = None

    def handler(request: httpx.Request):
        nonlocal last_request
        last_request = request
        return httpx.Response(200, json={
            "id": "test-wire-1",
            "choices": [{
                "message": {"role": "assistant", "content": "SHOPCONNECT_OLLAMA_CONNECTION_OK"},
                "finish_reason": "stop"
            }],
            "model": "qwen3.5:0.8b",
        })

    transport = httpx.MockTransport(handler)
    real_client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        http_client=httpx.Client(transport=transport),
    )

    provider = OllamaProvider(client=real_client)
    res = provider.complete([LLMMessage(role="user", content="Test prompt")])

    assert res.message.content == "SHOPCONNECT_OLLAMA_CONNECTION_OK"
    assert last_request is not None

    body = json.loads(last_request.content.decode("utf-8"))
    assert body.get("reasoning_effort") == "none"
    assert "think" not in body
    assert body.get("options", {}).get("num_ctx") == 4096


def test_ollama_message_conversion_all_roles():
    """Message conversion supports system, user, assistant, and tool responses with tool_call_id."""
    messages = [
        LLMMessage(role="system", content="System instruction"),
        LLMMessage(role="user", content="User query"),
        LLMMessage(
            role="assistant",
            content=None,
            tool_calls=({"id": "call_1", "function": {"name": "inspect", "arguments": "{}"}},),
        ),
        LLMMessage(role="tool", content='{"result": "ok"}', tool_call_id="call_1"),
    ]
    converted = convert_to_chat_completions_messages(messages)
    assert len(converted) == 4
    assert converted[0] == {"role": "system", "content": "System instruction"}
    assert converted[1] == {"role": "user", "content": "User query"}
    assert converted[2]["role"] == "assistant"
    assert converted[2]["tool_calls"][0]["id"] == "call_1"
    assert converted[3] == {"role": "tool", "content": '{"result": "ok"}', "tool_call_id": "call_1"}


def test_ollama_provider_sequential_lock():
    """OllamaProvider uses _global_execution_lock during completion."""
    assert hasattr(OllamaProvider, "_global_execution_lock")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = SimpleNamespace(
        id="ollama-lock",
        model="qwen3.5:0.8b",
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(role="assistant", content="OK"))],
        usage=None,
    )
    provider = OllamaProvider(client=mock_client)
    res = provider.complete([LLMMessage(role="user", content="Test")])
    assert res.message.content == "OK"


# =============================================================================
# 3. TOOL CALLING TESTS
# =============================================================================

def test_ollama_provider_tool_calls_and_correlation():
    """Ollama tool calls and multiple tool calls are parsed and correlation preserved."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="ollama-tools",
        model="qwen3.5:0.8b",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_inspect_1",
                            function=SimpleNamespace(
                                name="inspect_object",
                                arguments='{"target": "meal_plan"}',
                            ),
                        ),
                        SimpleNamespace(
                            id="call_inspect_2",
                            function=SimpleNamespace(
                                name="get_day_schedule",
                                arguments='{"day": "Wednesday"}',
                            ),
                        ),
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=85, completion_tokens=30, total_tokens=115),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    provider = OllamaProvider(client=mock_client)
    res = provider.complete([LLMMessage(role="user", content="Inspect plan")])

    assert res.message.tool_calls is not None
    assert len(res.message.tool_calls) == 2
    assert res.message.tool_calls[0]["name"] == "inspect_object"
    assert res.message.tool_calls[1]["name"] == "get_day_schedule"


# =============================================================================
# 4. USAGE EXTRACTION & RUNTIME METADATA TESTS
# =============================================================================

def test_ollama_provider_runtime_durations_and_metadata():
    """Ollama local-runtime durations (prompt_eval, eval, total) are captured in raw_response."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="ollama-durations",
        model="qwen3.5:0.8b",
        system_fingerprint="fp_qwen",
        total_duration=500000000,
        load_duration=50000000,
        prompt_eval_duration=150000000,
        eval_duration=300000000,
        digest="sha256:abc12345",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="Done"),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=20, total_tokens=70),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    provider = OllamaProvider(client=mock_client)
    res = provider.complete([LLMMessage(role="user", content="Test")])

    assert res.raw_response.get("total_duration") == 500000000
    assert res.raw_response.get("load_duration") == 50000000
    assert res.raw_response.get("prompt_eval_duration") == 150000000
    assert res.raw_response.get("eval_duration") == 300000000
    assert res.raw_response.get("digest") == "sha256:abc12345"
    assert res.raw_response.get("deployment_mode") == "local_inference"


def test_ollama_provider_reasoning_metadata_distinguished_from_content():
    """Reasoning metadata is detected and distinguished from final content; hidden text not stored."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="ollama-reasoning",
        model="qwen3.5:0.8b",
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(
                    role="assistant",
                    content="",  # empty user-facing content
                    reasoning_content="Thinking step 1 2 3...",
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=21,
            completion_tokens=64,
            total_tokens=85,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=64),
        ),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    provider = OllamaProvider(client=mock_client)
    res = provider.complete([LLMMessage(role="user", content="Test")])

    # Final content remains empty string (never substituted with reasoning)
    assert res.message.content == ""
    # finish_reason is retained
    assert res.finish_reason == "length"
    # Metadata captures reasoning presence and length
    assert res.raw_response.get("reasoning_present") is True
    assert res.raw_response.get("reasoning_length") == len("Thinking step 1 2 3...")
    # Hidden reasoning text is NOT stored in raw_response
    assert "Thinking" not in str(res.raw_response)


def test_ollama_provider_missing_usage():
    """When Ollama omits usage, usage is preserved as None without error."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="ollama-no-usage",
        model="qwen3.5:0.8b",
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(role="assistant", content="Hello"))],
        usage=None,
    )
    mock_client.chat.completions.create.return_value = mock_resp

    provider = OllamaProvider(client=mock_client)
    res = provider.complete([LLMMessage(role="user", content="Hi")])
    assert res.usage is None


def test_ollama_provider_missing_total_tokens():
    """When total_tokens is omitted, total_tokens is None and derived_total_tokens is populated."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="ollama-derived-usage",
        model="qwen3.5:0.8b",
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(role="assistant", content="Hello"))],
        usage=SimpleNamespace(prompt_tokens=30, completion_tokens=10, total_tokens=None),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    provider = OllamaProvider(client=mock_client)
    res = provider.complete([LLMMessage(role="user", content="Hi")])

    assert res.usage is not None
    assert res.usage.input_tokens == 30
    assert res.usage.output_tokens == 10
    assert res.usage.total_tokens is None
    assert res.usage.derived_total_tokens == 40


# =============================================================================
# 5. OFFLINE SERVER DETECTION & NO CLOUD FALLBACK
# =============================================================================

def test_ollama_provider_offline_server_error_message():
    """When Ollama server is offline, raises ConnectionError with setup instructions for qwen3.5:0.8b."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Connection refused to http://localhost:11434")

    provider = OllamaProvider(client=mock_client)
    with pytest.raises(ConnectionError) as exc_info:
        provider.complete([LLMMessage(role="user", content="Hello")])

    err_msg = str(exc_info.value)
    assert "ollama serve" in err_msg
    assert "ollama pull qwen3.5:0.8b" in err_msg
    assert "never falling back to remote cloud inference" in err_msg


def test_ollama_provider_timeout_error():
    """Timeout during local inference raises TimeoutError."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Request timed out after 180s")

    provider = OllamaProvider(client=mock_client)
    with pytest.raises(TimeoutError, match="timed out"):
        provider.complete([LLMMessage(role="user", content="Hello")])


# =============================================================================
# 6. LOCAL PRICING & COST EVALUATION
# =============================================================================

def test_ollama_local_pricing_evaluates_cost_to_none():
    """Local inference pricing must evaluate monetary cost strictly to None."""
    pricing = create_ollama_local_pricing()
    assert pricing.provider_name == "Ollama"
    assert pricing.model_name == "qwen3.5:0.8b"
    assert pricing.billing_tier == "local_inference"
    assert pricing.input_cost_per_million is None
    assert pricing.output_cost_per_million is None
    assert pricing.is_complete() is False

    usage = LLMUsage(
        input_tokens=500,
        output_tokens=150,
        total_tokens=650,
        is_provider_reported=True,
    )
    cost = calculate_cost(usage, pricing)
    # Must evaluate to None (never falsely declare $0.00)
    assert cost is None
