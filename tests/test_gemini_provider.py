"""Unit tests for GeminiProvider and Google GenAI SDK integration.

Verifies Phase 6.5 Gemini provider requirements:
- Provider protocol compliance (LLMProvider).
- gemini-3.8-flash model configuration, thinking level low, output ceiling 2048.
- Automatic function calling explicitly disabled (manual tool dispatch).
- Message mapping (system instruction, user, model, function calls, function responses).
- Function call ID and name preservation.
- Multiple function calls in a single model response.
- JSON response format configuration.
- Provider-reported usage extraction (input, output, cached, thoughts/reasoning).
- Missing usage metadata handled safely (None, not fabricated).
- Model version and response ID extraction.
- Typed exception mappings (AuthenticationError, RateLimitError, TimeoutError, ConnectionError, ConfigurationError, ProviderError).
- Missing GEMINI_API_KEY error handling and secret redaction.
- 100% offline with zero network calls (injected test double client).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError

from shopconnect_env.agents.providers import (
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    GeminiProvider,
    LLMCompletionResult,
    LLMMessage,
    LLMProvider,
    LLMUsage,
    ProviderError,
    RateLimitError,
    TimeoutError,
    convert_to_gemini_tools,
)


# =============================================================================
# 1. PROTOCOL & CONFIGURATION TESTS
# =============================================================================

def test_gemini_provider_protocol_compliance():
    """GeminiProvider must implement the LLMProvider Protocol."""
    provider = GeminiProvider(api_key="test-api-key")
    assert isinstance(provider, LLMProvider)


def test_gemini_provider_default_configuration():
    """GeminiProvider must default to gemini-3.8-flash, thinking level low, max_tokens 2048."""
    provider = GeminiProvider(api_key="test-api-key")
    assert provider.model == "gemini-3.8-flash"
    assert provider.thinking_level == "low"
    assert provider.max_output_tokens == 2048
    assert provider.sdk_version == "3.1.0"


def test_gemini_provider_repr_redacts_api_key():
    """GeminiProvider repr must never expose secrets."""
    secret_key = "AIzaSySecretGeminiKey123456789"
    provider = GeminiProvider(api_key=secret_key)
    repr_str = repr(provider)

    assert secret_key not in repr_str
    assert "gemini-3.8-flash" in repr_str


def test_gemini_provider_missing_api_key_raises_configuration_error(monkeypatch):
    """Missing GEMINI_API_KEY must raise ConfigurationError without network calls."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiProvider()

    with pytest.raises(ConfigurationError) as exc_info:
        provider.complete([LLMMessage(role="user", content="hello")])

    assert "GEMINI_API_KEY" in str(exc_info.value)


# =============================================================================
# 2. GENERATE CONTENT CONFIG & CALL VERIFICATION
# =============================================================================

def test_gemini_provider_generate_content_config_parameters():
    """Verify parameters passed to client.models.generate_content."""
    fake_client = MagicMock()
    mock_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(
                    role="model",
                    parts=[SimpleNamespace(text="Test response", thought=False, function_call=None)],
                ),
            )
        ],
        text="Test response",
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=25,
            total_token_count=125,
            cached_content_token_count=None,
            thoughts_token_count=None,
        ),
        model_version="gemini-3.8-flash",
        response_id="resp_test_01",
    )
    fake_client.models.generate_content.return_value = mock_resp

    provider = GeminiProvider(client=fake_client)
    messages = [
        LLMMessage(role="system", content="System instruction context"),
        LLMMessage(role="user", content="User prompt"),
    ]

    res = provider.complete(
        messages=messages,
        reasoning_effort="low",
        max_output_tokens=1024,
    )

    assert fake_client.models.generate_content.called
    call_kwargs = fake_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.8-flash"

    # Verify config
    config: types.GenerateContentConfig = call_kwargs["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.max_output_tokens == 1024
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert config.automatic_function_calling.disable is True
    assert config.temperature is None  # Temperature omitted
    assert config.system_instruction == "System instruction context"

    # Verify response
    assert res.message.content == "Test response"
    assert res.message.role == "assistant"
    assert res.usage.input_tokens == 100
    assert res.usage.output_tokens == 25
    assert res.usage.total_tokens == 125
    assert res.usage.is_provider_reported is True
    assert res.provider_request_id == "resp_test_01"
    assert res.model_identifier == "gemini-3.8-flash"


def test_gemini_provider_json_response_format():
    """Verify response_format={'type': 'json_object'} sets response_mime_type."""
    fake_client = MagicMock()
    mock_resp = SimpleNamespace(
        candidates=[],
        text='{"key": "value"}',
        usage_metadata=None,
        model_version="gemini-3.8-flash",
        response_id=None,
    )
    fake_client.models.generate_content.return_value = mock_resp

    provider = GeminiProvider(client=fake_client)
    provider.complete(
        messages=[LLMMessage(role="user", content="JSON request")],
        response_format={"type": "json_object"},
    )

    config = fake_client.models.generate_content.call_args.kwargs["config"]
    assert config.response_mime_type == "application/json"


# =============================================================================
# 3. MESSAGE & TOOL CONVERSION TESTS
# =============================================================================

def test_gemini_message_conversion_and_call_id_preservation():
    """Test full multi-turn conversation conversion with tool calls and tool responses."""
    fake_client = MagicMock()
    mock_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(
                    role="model",
                    parts=[SimpleNamespace(text="Finished inspection.", thought=False, function_call=None)],
                ),
            )
        ],
        text="Finished inspection.",
        usage_metadata=None,
        model_version="gemini-3.8-flash",
        response_id="resp_002",
    )
    fake_client.models.generate_content.return_value = mock_resp

    provider = GeminiProvider(client=fake_client)

    messages = [
        LLMMessage(role="system", content="System setup"),
        LLMMessage(role="user", content="Check spinach"),
        LLMMessage(
            role="assistant",
            tool_calls=(
                {
                    "id": "gemini_call_101",
                    "type": "function",
                    "function": {
                        "name": "inspect_ingredient",
                        "arguments": '{"ingredient_id": "ing_spinach"}',
                    },
                },
            ),
        ),
        LLMMessage(
            role="tool",
            content='{"name": "Baby Spinach", "allergens": []}',
            tool_call_id="gemini_call_101",
            name="inspect_ingredient",
        ),
    ]

    provider.complete(messages=messages)

    contents = fake_client.models.generate_content.call_args.kwargs["contents"]
    assert len(contents) == 3  # user, model (with function call), user (with function response)

    # 1. First user message
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Check spinach"

    # 2. Assistant message with function call
    assert contents[1].role == "model"
    fc_part = contents[1].parts[0].function_call
    assert fc_part is not None
    assert fc_part.id == "gemini_call_101"
    assert fc_part.name == "inspect_ingredient"
    assert fc_part.args == {"ingredient_id": "ing_spinach"}

    # 3. Tool response message
    assert contents[2].role in ("tool", "user")
    fr_part = contents[2].parts[0].function_response
    assert fr_part is not None
    assert fr_part.id == "gemini_call_101"
    assert fr_part.name == "inspect_ingredient"
    assert fr_part.response == {"name": "Baby Spinach", "allergens": []}


def test_gemini_multiple_function_calls_in_response():
    """Verify GeminiProvider captures multiple tool calls in a single candidate."""
    fake_client = MagicMock()

    part1 = SimpleNamespace(
        function_call=SimpleNamespace(
            id="call_inspect_1",
            name="inspect_recipe",
            args={"recipe_id": "rec_01"},
        ),
        thought=False,
    )
    part2 = SimpleNamespace(
        function_call=SimpleNamespace(
            id="call_inspect_2",
            name="inspect_ingredient",
            args={"ingredient_id": "ing_02"},
        ),
        thought=False,
    )

    mock_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(role="model", parts=[part1, part2]),
            )
        ],
        text=None,
        usage_metadata=None,
        model_version="gemini-3.8-flash",
        response_id="resp_multi_call",
    )
    fake_client.models.generate_content.return_value = mock_resp

    provider = GeminiProvider(client=fake_client)
    res = provider.complete([LLMMessage(role="user", content="Inspect multiple")])

    assert res.message.tool_calls is not None
    assert len(res.message.tool_calls) == 2
    assert res.message.tool_calls[0]["id"] == "call_inspect_1"
    assert res.message.tool_calls[0]["function"]["name"] == "inspect_recipe"
    assert json.loads(res.message.tool_calls[0]["function"]["arguments"]) == {"recipe_id": "rec_01"}

    assert res.message.tool_calls[1]["id"] == "call_inspect_2"
    assert res.message.tool_calls[1]["function"]["name"] == "inspect_ingredient"
    assert json.loads(res.message.tool_calls[1]["function"]["arguments"]) == {"ingredient_id": "ing_02"}


def test_convert_to_gemini_tools():
    """Test converting standard tool definitions to types.Tool declarations."""
    tools_def = [
        {
            "type": "function",
            "function": {
                "name": "replace_meal",
                "description": "Schedule a meal into a slot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "day": {"type": "string"},
                        "meal_type": {"type": "string"},
                    },
                    "required": ["day", "meal_type"],
                },
            },
        }
    ]

    tools = convert_to_gemini_tools(tools_def)
    assert tools is not None
    assert len(tools) == 1
    tool = tools[0]
    assert isinstance(tool, types.Tool)
    assert len(tool.function_declarations) == 1
    fn_decl = tool.function_declarations[0]
    assert fn_decl.name == "replace_meal"
    assert fn_decl.description == "Schedule a meal into a slot."


# =============================================================================
# 4. TOKEN USAGE & METADATA TESTS
# =============================================================================

def test_gemini_usage_metadata_extraction():
    """Verify input, output, cached, and thought tokens extraction."""
    fake_client = MagicMock()
    mock_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(
                    role="model",
                    parts=[SimpleNamespace(text="Done", thought=False, function_call=None)],
                ),
            )
        ],
        text="Done",
        usage_metadata=SimpleNamespace(
            prompt_token_count=350,
            candidates_token_count=120,
            total_token_count=470,
            cached_content_token_count=80,
            thoughts_token_count=45,
        ),
        model_version="gemini-3.8-flash-001",
        response_id="req_usage_999",
    )
    fake_client.models.generate_content.return_value = mock_resp

    provider = GeminiProvider(client=fake_client)
    res = provider.complete([LLMMessage(role="user", content="Usage test")])

    assert res.usage is not None
    assert res.usage.input_tokens == 350
    assert res.usage.output_tokens == 120
    assert res.usage.total_tokens == 470
    assert res.usage.cached_input_tokens == 80
    assert res.usage.reasoning_tokens == 45
    assert res.usage.is_provider_reported is True
    assert res.model_identifier == "gemini-3.8-flash-001"
    assert res.provider_request_id == "req_usage_999"
    assert res.system_fingerprint is None  # Gemini API does not report system fingerprint


def test_gemini_missing_usage_metadata():
    """When usage_metadata is missing, usage is None rather than fabricated."""
    fake_client = MagicMock()
    mock_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(
                    role="model",
                    parts=[SimpleNamespace(text="No usage", thought=False, function_call=None)],
                ),
            )
        ],
        text="No usage",
        usage_metadata=None,
        model_version="gemini-3.8-flash",
        response_id="req_no_usage",
    )
    fake_client.models.generate_content.return_value = mock_resp

    provider = GeminiProvider(client=fake_client)
    res = provider.complete([LLMMessage(role="user", content="Test")])

    assert res.usage is None
    assert res.message.content == "No usage"


def test_gemini_missing_model_version_preserves_none():
    """When Gemini response omits model_version, preserve model_identifier as None."""
    fake_client = MagicMock()
    mock_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(
                    role="model",
                    parts=[SimpleNamespace(text="Response without model version", thought=False, function_call=None)],
                ),
            )
        ],
        text="Response without model version",
        usage_metadata=None,
        model_version=None,
        response_id="req_no_model_version",
    )
    fake_client.models.generate_content.return_value = mock_resp

    provider = GeminiProvider(client=fake_client)
    res = provider.complete([LLMMessage(role="user", content="Test")])

    assert res.model_identifier is None
    assert "model_version" not in res.raw_response


def test_gemini_usage_missing_total_tokens_preserves_none_with_derived_total():
    """When Gemini omits total_token_count, preserve total_tokens as None without fabricating provider usage."""
    fake_client = MagicMock()
    mock_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(
                    role="model",
                    parts=[SimpleNamespace(text="Partial usage test", thought=False, function_call=None)],
                ),
            )
        ],
        text="Partial usage test",
        usage_metadata=SimpleNamespace(
            prompt_token_count=150,
            candidates_token_count=60,
            total_token_count=None,  # Provider omitted total count
            cached_content_token_count=None,
            thoughts_token_count=None,
        ),
        model_version="gemini-3.8-flash",
        response_id="req_no_tot",
    )
    fake_client.models.generate_content.return_value = mock_resp

    provider = GeminiProvider(client=fake_client)
    res = provider.complete([LLMMessage(role="user", content="Test partial")])

    assert res.usage is not None
    assert res.usage.input_tokens == 150
    assert res.usage.output_tokens == 60
    # Must be None, not silently calculated as 210 and claimed to be provider-reported
    assert res.usage.total_tokens is None
    # Derived total must be explicitly captured with derived provenance
    assert res.usage.derived_total_tokens == 210
    assert res.usage.is_provider_reported is True


# =============================================================================
# 5. ERROR MAPPINGS & SECURITY TESTS
# =============================================================================

def test_gemini_authentication_error_mapping():
    """ClientError with 401/403 or PERMISSION_DENIED maps to AuthenticationError."""
    fake_client = MagicMock()
    err = ClientError(code=403, response_json={"error": {"message": "API_KEY_INVALID: The provided API key is invalid."}})
    fake_client.models.generate_content.side_effect = err

    provider = GeminiProvider(client=fake_client)
    with pytest.raises(AuthenticationError) as exc_info:
        provider.complete([LLMMessage(role="user", content="test")])

    assert "Authentication failed" in str(exc_info.value)


def test_gemini_rate_limit_error_mapping():
    """ClientError with 429 or RESOURCE_EXHAUSTED maps to RateLimitError."""
    fake_client = MagicMock()
    err = ClientError(code=429, response_json={"error": {"message": "RESOURCE_EXHAUSTED: Quota exceeded for gemini-3.8-flash."}})
    fake_client.models.generate_content.side_effect = err

    provider = GeminiProvider(client=fake_client)
    with pytest.raises(RateLimitError) as exc_info:
        provider.complete([LLMMessage(role="user", content="test")])

    assert "rate limit or quota exceeded" in str(exc_info.value)


def test_gemini_timeout_error_mapping():
    """Timeout exceptions map to TimeoutError."""
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = TimeoutError("DEADLINE_EXCEEDED: request timed out")

    provider = GeminiProvider(client=fake_client)
    with pytest.raises(TimeoutError) as exc_info:
        provider.complete([LLMMessage(role="user", content="test")])

    assert "timed out" in str(exc_info.value)


def test_gemini_connection_error_mapping():
    """Connection failures map to ConnectionError."""
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = Exception("Failed to connect: httpx.ConnectError")

    provider = GeminiProvider(client=fake_client)
    with pytest.raises(ConnectionError) as exc_info:
        provider.complete([LLMMessage(role="user", content="test")])

    assert "Failed to connect" in str(exc_info.value)


def test_gemini_configuration_error_mapping():
    """Model not found (404) maps to ConfigurationError."""
    fake_client = MagicMock()
    err = ClientError(code=404, response_json={"error": {"message": "NOT_FOUND: Model 'gemini-nonexistent' does not exist."}})
    fake_client.models.generate_content.side_effect = err

    provider = GeminiProvider(client=fake_client, model="gemini-nonexistent")
    with pytest.raises(ConfigurationError) as exc_info:
        provider.complete([LLMMessage(role="user", content="test")])

    assert "not found" in str(exc_info.value)


def test_gemini_safety_block_mapping():
    """Candidate with finish_reason='SAFETY' maps to ProviderError."""
    fake_client = MagicMock()
    mock_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="SAFETY",
                content=None,
            )
        ],
        text=None,
        usage_metadata=None,
        model_version="gemini-3.8-flash",
        response_id="resp_blocked",
    )
    fake_client.models.generate_content.return_value = mock_resp

    provider = GeminiProvider(client=fake_client)
    with pytest.raises(ProviderError) as exc_info:
        provider.complete([LLMMessage(role="user", content="harmful query")])

    assert "safety filters" in str(exc_info.value)


def test_gemini_secret_redaction_in_exception_messages():
    """Underlying exceptions containing the API key must have the key redacted."""
    secret_key = "AIzaSySecretSensitiveKey123"
    fake_client = MagicMock()
    err = Exception(f"Internal server error: failed using key {secret_key}")
    fake_client.models.generate_content.side_effect = err

    provider = GeminiProvider(api_key=secret_key, client=fake_client)
    with pytest.raises(ProviderError) as exc_info:
        provider.complete([LLMMessage(role="user", content="test")])

    err_str = str(exc_info.value)
    assert secret_key not in err_str
    assert "[REDACTED_API_KEY]" in err_str


def test_gemini_connect_timeout_classification_and_cause_preservation():
    """httpx.ConnectTimeout maps to TimeoutError (not ConnectionError) with cause preserved."""
    import httpx
    fake_client = MagicMock()
    original_err = httpx.ConnectTimeout("Connection timed out while connecting to 142.250.190.42:443")
    fake_client.models.generate_content.side_effect = original_err

    provider = GeminiProvider(client=fake_client)
    with pytest.raises(TimeoutError) as exc_info:
        provider.complete([LLMMessage(role="user", content="test")])

    err_str = str(exc_info.value)
    assert "connection timed out (ConnectTimeout)" in err_str
    assert exc_info.value.__cause__ is original_err
    assert isinstance(exc_info.value, TimeoutError)
    assert not isinstance(exc_info.value, ConnectionError)


def test_gemini_read_timeout_classification_and_cause_preservation():
    """httpx.ReadTimeout maps to TimeoutError with cause preserved."""
    import httpx
    fake_client = MagicMock()
    original_err = httpx.ReadTimeout("Read timed out after 30.0 seconds")
    fake_client.models.generate_content.side_effect = original_err

    provider = GeminiProvider(client=fake_client)
    with pytest.raises(TimeoutError) as exc_info:
        provider.complete([LLMMessage(role="user", content="test")])

    err_str = str(exc_info.value)
    assert "read timed out (ReadTimeout)" in err_str
    assert exc_info.value.__cause__ is original_err


def test_gemini_tls_ssl_error_classification_and_cause_preservation():
    """SSL verification errors map to ConnectionError identifying TLS/SSL failure."""
    import ssl
    fake_client = MagicMock()
    original_err = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    fake_client.models.generate_content.side_effect = original_err

    provider = GeminiProvider(client=fake_client)
    with pytest.raises(ConnectionError) as exc_info:
        provider.complete([LLMMessage(role="user", content="test")])

    err_str = str(exc_info.value)
    assert "TLS/SSL error (SSLCertVerificationError)" in err_str
    assert exc_info.value.__cause__ is original_err


def test_gemini_connect_error_classification_and_cause_preservation():
    """httpx.ConnectError maps to ConnectionError identifying connect failure."""
    import httpx
    fake_client = MagicMock()
    original_err = httpx.ConnectError("[WinError 10061] No connection could be made because the target machine actively refused it")
    fake_client.models.generate_content.side_effect = original_err

    provider = GeminiProvider(client=fake_client)
    with pytest.raises(ConnectionError) as exc_info:
        provider.complete([LLMMessage(role="user", content="test")])

    err_str = str(exc_info.value)
    assert "Failed to connect to Gemini API endpoint (ConnectError)" in err_str
    assert exc_info.value.__cause__ is original_err


def test_gemini_http_5xx_server_error_classification_and_cause_preservation():
    """HTTP 500 and 503 ServerError map to ProviderError identifying server status."""
    fake_client = MagicMock()
    err_500 = ServerError(code=500, response_json={"error": {"message": "Internal server error"}}, response=None)
    fake_client.models.generate_content.side_effect = err_500

    provider = GeminiProvider(client=fake_client)
    with pytest.raises(ProviderError) as exc_info:
        provider.complete([LLMMessage(role="user", content="test")])

    assert "server error (HTTP 500)" in str(exc_info.value)
    assert exc_info.value.__cause__ is err_500

    err_503 = ServerError(code=503, response_json={"error": {"message": "Service unavailable"}}, response=None)
    fake_client.models.generate_content.side_effect = err_503
    with pytest.raises(ProviderError) as exc_info2:
        provider.complete([LLMMessage(role="user", content="test")])

    assert "server error (HTTP 503)" in str(exc_info2.value)
    assert exc_info2.value.__cause__ is err_503


def test_gemini_timeout_units_and_retries_configuration(monkeypatch):
    """Verify timeout is converted to milliseconds and retries=0 sets attempts=1."""
    captured_client_kwargs = {}

    def fake_genai_client(*args, **kwargs):
        captured_client_kwargs.update(kwargs)
        mock = MagicMock()
        return mock

    monkeypatch.setattr("google.genai.Client", fake_genai_client)

    # 1. 30 seconds -> 30,000 ms, max_retries=0 -> attempts=1
    prov30 = GeminiProvider(api_key="AIzaSyTestKey", timeout_seconds=30.0, max_retries=0)
    _ = prov30._get_client()

    http_opts = captured_client_kwargs.get("http_options")
    assert http_opts is not None
    assert http_opts.timeout == 30000  # Exactly 30 seconds in milliseconds
    assert http_opts.retry_options is not None
    assert http_opts.retry_options.attempts == 1  # 0 retries = 1 single attempt

    # 2. Per-request GenerateContentConfig http_options check
    fake_client = MagicMock()
    mock_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(
                    role="model",
                    parts=[SimpleNamespace(text="OK", thought=False, function_call=None)],
                ),
            )
        ],
        text="OK",
        usage_metadata=None,
        model_version="gemini-3.8-flash",
        response_id="req_ok",
    )
    fake_client.models.generate_content.return_value = mock_resp

    prov_req = GeminiProvider(client=fake_client, timeout_seconds=30.0, max_retries=0)
    prov_req.complete([LLMMessage(role="user", content="test")])

    config_passed = fake_client.models.generate_content.call_args.kwargs["config"]
    assert config_passed.http_options is not None
    assert config_passed.http_options.timeout == 30000
    assert config_passed.http_options.retry_options.attempts == 1
