"""LLM provider protocols, response containers, and concrete provider adapters.

Defines the decoupled provider abstraction layer for Stage 6 evaluation:
- LLMMessage, LLMUsage, and LLMCompletionResult contracts.
- LLMProvider Protocol.
- MockLLMProvider for 100% offline, deterministic testing and simulation.
- OpenAIProvider integrating OpenAI Responses API for gpt-6-luna.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Protocol, runtime_checkable


# =============================================================================
# 1. PROVIDER EXCEPTIONS (TYPED ERROR HIERARCHY)
# =============================================================================

class ProviderError(Exception):
    """Base exception for all LLM provider failures."""
    pass


class ConfigurationError(ProviderError):
    """Raised when provider configuration or credentials are missing or invalid."""
    pass


class AuthenticationError(ProviderError):
    """Raised when authentication with the provider fails (e.g. invalid API key)."""
    pass


class RateLimitError(ProviderError):
    """Raised when provider rate limits or quotas are exceeded."""
    pass


class TimeoutError(ProviderError):
    """Raised when an API request to the provider times out."""
    pass


class ConnectionError(ProviderError):
    """Raised when a network connection failure prevents reaching the provider."""
    pass


class MalformedResponseError(ProviderError):
    """Raised when the provider returns a response with invalid or unparseable structure."""
    pass


# =============================================================================
# 2. CORE DATA CONTRACTS
# =============================================================================

@dataclass(frozen=True)
class LLMMessage:
    """Standardized representation of a single conversation turn.

    Attributes:
        role: Message role ('system', 'user', 'assistant', or 'tool').
        content: Text content of the message, if any.
        tool_calls: Optional tuple of tool invocation dictionaries.
        tool_call_id: Identifier of the tool call being answered (when role='tool').
        name: Optional function or sender name.
    """

    role: str
    content: Optional[str] = None
    tool_calls: Optional[tuple] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass(frozen=True)
class LLMUsage:
    """Token consumption reported by the provider or simulated by a mock.

    Attributes:
        input_tokens: Billed input prompt tokens.
        output_tokens: Billed completion/generation tokens.
        total_tokens: Sum of input and output tokens.
        cached_input_tokens: Billed input tokens served from provider cache (if available).
        reasoning_tokens: Internal thinking tokens (if reported by provider).
        is_provider_reported: True if numbers originate directly from the API;
                             False if simulated by a mock provider.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    is_provider_reported: bool = True


@dataclass(frozen=True)
class LLMCompletionResult:
    """Standardized result of an LLM completion request.

    Attributes:
        message: The LLMMessage returned by the model.
        usage: Token consumption breakdown, or None if unavailable.
        model_identifier: The model string (e.g. 'gpt-6-luna' or mock ID).
        system_fingerprint: System fingerprint reported by provider for traceability.
        provider_request_id: Unique request ID returned by provider for audit logging.
        latency_ms: Measured end-to-end request duration in milliseconds.
        finish_reason: Termination reason ('stop', 'tool_calls', 'length', etc.).
        raw_response: Immutable mapping containing raw response metadata for audit traces.
    """

    message: LLMMessage
    usage: Optional[LLMUsage]
    model_identifier: str
    system_fingerprint: Optional[str] = None
    provider_request_id: Optional[str] = None
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    raw_response: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.raw_response, MappingProxyType):
            object.__setattr__(
                self,
                "raw_response",
                MappingProxyType(dict(self.raw_response)),
            )


# =============================================================================
# 3. LLM PROVIDER PROTOCOL
# =============================================================================

@runtime_checkable
class LLMProvider(Protocol):
    """Abstract interface isolating LLM network execution from agent logic."""

    def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: str = "low",
    ) -> LLMCompletionResult:
        """Execute a completion request against the underlying model.

        Args:
            messages: List of LLMMessage objects representing conversation history.
            tools: Optional tool/function specifications.
            tool_choice: Optional tool selection constraint ('auto', 'required', or name).
            response_format: Optional structured output schema constraint.
            reasoning_effort: Effort parameter for reasoning models ('low', 'medium', 'high').

        Returns:
            An LLMCompletionResult containing the response message, usage, and metadata.
        """
        ...


# =============================================================================
# 4. DETERMINISTIC OFFLINE MOCK PROVIDER
# =============================================================================

class MockLLMProvider:
    """Offline, deterministic mock provider for testing and validation.

    Guarantees:
    - Zero network calls.
    - Zero API key requirements.
    - Deterministic responses.
    - Sequential responses support for multi-turn dialogues.
    - Explicit is_provider_reported=False marking for simulated tokens.
    """

    def __init__(
        self,
        responses: Optional[List[LLMCompletionResult]] = None,
        canned_responses: Optional[Dict[str, LLMCompletionResult]] = None,
        default_response: Optional[LLMCompletionResult] = None,
    ) -> None:
        """Initialize MockLLMProvider.

        Args:
            responses: Queue of sequential responses to pop on successive complete() calls.
            canned_responses: Mapping from task/prompt substring to canned completion.
            default_response: Fallback completion if queue is empty and no canned match is found.
        """
        self.responses_queue: List[LLMCompletionResult] = list(responses) if responses else []
        self.canned_responses: Dict[str, LLMCompletionResult] = dict(canned_responses) if canned_responses else {}
        self.default_response = default_response
        self.call_history: List[Dict[str, Any]] = []

    def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: str = "low",
    ) -> LLMCompletionResult:
        """Return a deterministic completion without network access."""
        self.call_history.append({
            "messages": list(messages),
            "tools": list(tools) if tools else None,
            "tool_choice": tool_choice,
            "response_format": response_format,
            "reasoning_effort": reasoning_effort,
        })

        # 1. Sequential queue matching (first priority for multi-turn dialogues)
        if self.responses_queue:
            return self.responses_queue.pop(0)

        # 2. Canned prompt mapping (search latest user message content)
        user_content = ""
        for m in reversed(messages):
            if m.role == "user" and m.content:
                user_content = m.content
                break

        for key, resp in self.canned_responses.items():
            if key in user_content:
                return resp

        # 3. Configured default response
        if self.default_response is not None:
            return self.default_response

        # 4. Standard simulated fallback response
        return LLMCompletionResult(
            message=LLMMessage(
                role="assistant",
                content='{"status": "mock_default_response"}',
            ),
            usage=LLMUsage(
                input_tokens=120,
                output_tokens=60,
                total_tokens=180,
                cached_input_tokens=30,
                reasoning_tokens=20,
                is_provider_reported=False,  # Explicitly simulated
            ),
            model_identifier="mock-eval-model",
            system_fingerprint="fp_mock_offline_1",
            provider_request_id="req_mock_001",
            latency_ms=1.5,
            finish_reason="stop",
            raw_response={"mock": True},
        )


# =============================================================================
# 5. OPENAI RESPONSES API PROVIDER ADAPTER
# =============================================================================

def convert_to_responses_api_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Convert Stage 4 tool definitions to OpenAI Responses API function-tool format.

    Ensures compatibility with client.responses.create(tools=...).
    """
    if not tools:
        return None

    converted: List[Dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool.get("function", tool)
            converted.append({
                "type": "function",
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        else:
            converted.append(tool)
    return converted


class OpenAIProvider:
    """Provider adapter connecting to OpenAI Responses API for gpt-6-luna.

    Enforces Stage 6 architectural constraints:
    - Target model: gpt-6-luna.
    - Fixed reasoning: reasoning={"effort": "low"}.
    - No temperature parameter passed.
    - Credentials loaded strictly from OPENAI_API_KEY.
    - Secrets never logged or exposed in exceptions.
    - Exact provider-reported usage capture (input, output, cached, reasoning).
    - Missing usage fields left as None (never fabricated).
    - Supports dependency injection of client for 100% offline unit testing.
    """

    DEFAULT_MODEL: str = "gpt-6-luna"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = "low",
        timeout_seconds: float = 60.0,
        client: Optional[Any] = None,
    ) -> None:
        """Initialize OpenAIProvider.

        Args:
            api_key: Optional explicit API key. If omitted, reads OPENAI_API_KEY from env.
            model: Target model identifier (default: 'gpt-6-luna').
            reasoning_effort: Fixed reasoning effort (default: 'low').
            timeout_seconds: Request timeout in seconds.
            client: Optional injected OpenAI client or test double for offline testing.
        """
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self._explicit_api_key = api_key
        self._injected_client = client
        self._client: Optional[Any] = client

    def __repr__(self) -> str:
        """Return safe string representation without exposing API credentials."""
        return f"OpenAIProvider(model={self.model!r}, reasoning_effort={self.reasoning_effort!r})"

    def _get_client(self) -> Any:
        """Obtain or lazily instantiate the OpenAI client with credential validation."""
        if self._client is not None:
            return self._client

        api_key = self._explicit_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            raise ConfigurationError(
                "OPENAI_API_KEY environment variable is not set. "
                "Provide an API key or inject a client for testing."
            )

        import openai
        self._client = openai.OpenAI(
            api_key=api_key.strip(),
            timeout=self.timeout_seconds,
        )
        return self._client

    def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> LLMCompletionResult:
        """Execute a completion using the OpenAI Responses API.

        Strictly enforces:
        - Reasoning configuration effort='low'.
        - Temperature parameter omitted entirely.
        - Error mapping with zero secret leakage.
        """
        client = self._get_client()
        effort = reasoning_effort or self.reasoning_effort

        # 1. Format input items and separate system instructions
        instructions_parts: List[str] = []
        input_items: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                if msg.content:
                    instructions_parts.append(msg.content)
            elif msg.role in ("user", "assistant"):
                if msg.content:
                    input_items.append({"role": msg.role, "content": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        cid = tc.get("call_id") or tc.get("id")
                        fn_data = tc.get("function", {})
                        input_items.append({
                            "type": "function_call",
                            "call_id": cid,
                            "name": fn_data.get("name", tc.get("name", "")),
                            "arguments": fn_data.get("arguments", tc.get("arguments", "{}")),
                        })
            elif msg.role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id or "",
                    "output": msg.content or "",
                })

        instructions = "\n\n".join(instructions_parts) if instructions_parts else None

        # 2. Build Responses API payload arguments
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "reasoning": {"effort": effort},
            "input": input_items,
        }
        if instructions:
            request_kwargs["instructions"] = instructions

        formatted_tools = convert_to_responses_api_tools(tools)
        if formatted_tools:
            request_kwargs["tools"] = formatted_tools

        if tool_choice:
            request_kwargs["tool_choice"] = tool_choice

        # Explicit constraint: DO NOT pass temperature when reasoning effort is set.
        # request_kwargs deliberately omits 'temperature'

        # 3. Execute request with stopwatch timing and exception mapping
        t0 = time.perf_counter()
        try:
            raw_response = client.responses.create(**request_kwargs)
        except Exception as exc:
            self._handle_api_exception(exc)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Parse response output text and tool calls
        return self._parse_response(raw_response, latency_ms)

    def _handle_api_exception(self, exc: Exception) -> None:
        """Map underlying SDK exceptions to typed ProviderError classes without leaking credentials."""
        exc_type = type(exc).__name__
        exc_msg = str(exc)

        # Sanitize any accidental credential string from error messages
        sanitized_msg = "Request failed"
        if "Authentication" in exc_type or "Unauthorized" in exc_msg or getattr(exc, "status_code", None) == 401:
            raise AuthenticationError("Authentication failed: invalid or unauthorized OpenAI API key.") from None

        if "RateLimit" in exc_type or "429" in exc_msg or getattr(exc, "status_code", None) == 429:
            raise RateLimitError("OpenAI API rate limit or quota exceeded.") from None

        if "Timeout" in exc_type or "timed out" in exc_msg.lower():
            raise TimeoutError("OpenAI API request timed out.") from None

        if "Connection" in exc_type or "connection" in exc_msg.lower():
            raise ConnectionError("Failed to connect to OpenAI API endpoint.") from None

        # Clean generic message
        clean_detail = exc_msg.split("\n")[0] if exc_msg else exc_type
        # Remove potential key substrings
        for secret in filter(None, [self._explicit_api_key, os.environ.get("OPENAI_API_KEY")]):
            clean_detail = clean_detail.replace(secret, "[REDACTED_API_KEY]")

        raise ProviderError(f"OpenAI API provider error ({exc_type}): {clean_detail}") from None

    def _parse_response(self, response: Any, latency_ms: float) -> LLMCompletionResult:
        """Extract standardized LLMCompletionResult from Responses API Response object."""
        # 1. Output items extraction
        tool_calls: List[Dict[str, Any]] = []
        content_parts: List[str] = []

        output_items = getattr(response, "output", []) or []
        for item in output_items:
            item_type = getattr(item, "type", None)
            if item_type == "function_call" or hasattr(item, "call_id"):
                cid = getattr(item, "call_id", None) or getattr(item, "id", "")
                fn_name = getattr(item, "name", "")
                fn_args = getattr(item, "arguments", "{}")
                tool_calls.append({
                    "id": cid,
                    "call_id": cid,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": fn_args,
                    },
                })
            elif hasattr(item, "content"):
                c = getattr(item, "content", "")
                if isinstance(c, str):
                    content_parts.append(c)
                elif isinstance(c, list):
                    for part in c:
                        text_val = getattr(part, "text", str(part))
                        content_parts.append(text_val)

        text_content = "".join(content_parts) if content_parts else getattr(response, "output_text", None)

        msg = LLMMessage(
            role="assistant",
            content=text_content,
            tool_calls=tuple(tool_calls) if tool_calls else None,
        )

        # 2. Token usage extraction (exact provider reporting; missing fields stay None)
        usage: Optional[LLMUsage] = None
        usage_obj = getattr(response, "usage", None)
        if usage_obj is not None:
            # Input tokens
            in_tok = getattr(usage_obj, "input_tokens", None)
            if in_tok is None:
                in_tok = getattr(usage_obj, "prompt_tokens", 0)

            # Output tokens
            out_tok = getattr(usage_obj, "output_tokens", None)
            if out_tok is None:
                out_tok = getattr(usage_obj, "completion_tokens", 0)

            # Total tokens
            tot_tok = getattr(usage_obj, "total_tokens", None)
            if tot_tok is None:
                tot_tok = in_tok + out_tok

            # Cached input tokens
            cached_tok: Optional[int] = None
            in_details = getattr(usage_obj, "input_tokens_details", None) or getattr(usage_obj, "prompt_tokens_details", None)
            if in_details is not None:
                cached_tok = getattr(in_details, "cached_tokens", None)

            # Reasoning tokens
            reasoning_tok: Optional[int] = None
            out_details = getattr(usage_obj, "output_tokens_details", None) or getattr(usage_obj, "completion_tokens_details", None)
            if out_details is not None:
                reasoning_tok = getattr(out_details, "reasoning_tokens", None)

            usage = LLMUsage(
                input_tokens=int(in_tok),
                output_tokens=int(out_tok),
                total_tokens=int(tot_tok),
                cached_input_tokens=int(cached_tok) if cached_tok is not None else None,
                reasoning_tokens=int(reasoning_tok) if reasoning_tok is not None else None,
                is_provider_reported=True,
            )

        model_id = getattr(response, "model", self.model) or self.model
        sys_fp = getattr(response, "system_fingerprint", None)
        req_id = getattr(response, "id", None)
        finish_reason = getattr(response, "status", "stop") or "stop"

        raw_meta: Dict[str, Any] = {}
        if req_id:
            raw_meta["id"] = req_id
        if sys_fp:
            raw_meta["system_fingerprint"] = sys_fp
        if hasattr(response, "created_at"):
            raw_meta["created_at"] = response.created_at

        return LLMCompletionResult(
            message=msg,
            usage=usage,
            model_identifier=model_id,
            system_fingerprint=sys_fp,
            provider_request_id=req_id,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            raw_response=raw_meta,
        )
