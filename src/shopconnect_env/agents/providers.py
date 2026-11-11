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
        total_tokens: Total tokens reported directly by provider API, or None if omitted.
        cached_input_tokens: Billed input tokens served from provider cache (if available).
        reasoning_tokens: Internal thinking tokens (if reported by provider).
        is_provider_reported: True if numbers originate directly from the API;
                             False if simulated by a mock provider.
        cache_write_tokens: General cache creation tokens (if reported).
        cache_write_tokens_30m: 30-minute retention cache write tokens (if reported).
        cache_write_tokens_12h: 12-hour retention cache write tokens (if reported).
        derived_total_tokens: Derived sum (input_tokens + output_tokens) with explicit derived provenance,
                             used when the provider API omits a native total_tokens count.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    is_provider_reported: bool = True
    cache_write_tokens: Optional[int] = None
    cache_write_tokens_30m: Optional[int] = None
    cache_write_tokens_12h: Optional[int] = None
    derived_total_tokens: Optional[int] = None


def aggregate_usages(usages: List[Optional[LLMUsage]]) -> Optional[LLMUsage]:
    """Aggregate multiple LLMUsage records into a single cumulative total.

    Preserves unknown (None) fields when no record reports them.
    Guarantees that simulated mock usage (is_provider_reported=False) is never
    falsely aggregated into a provider-reported (True) record.
    """
    valid_usages = [u for u in usages if u is not None]
    if not valid_usages:
        return None

    input_tokens = sum(u.input_tokens for u in valid_usages)
    output_tokens = sum(u.output_tokens for u in valid_usages)

    has_total = any(u.total_tokens is not None for u in valid_usages)
    total_tokens = (
        sum(u.total_tokens for u in valid_usages if u.total_tokens is not None)
        if has_total
        else None
    )

    has_cached = any(u.cached_input_tokens is not None for u in valid_usages)
    cached_input_tokens = (
        sum(u.cached_input_tokens for u in valid_usages if u.cached_input_tokens is not None)
        if has_cached
        else None
    )

    has_reasoning = any(u.reasoning_tokens is not None for u in valid_usages)
    reasoning_tokens = (
        sum(u.reasoning_tokens for u in valid_usages if u.reasoning_tokens is not None)
        if has_reasoning
        else None
    )

    has_cw = any(u.cache_write_tokens is not None for u in valid_usages)
    cache_write_tokens = (
        sum(u.cache_write_tokens for u in valid_usages if u.cache_write_tokens is not None)
        if has_cw
        else None
    )

    has_cw_30m = any(u.cache_write_tokens_30m is not None for u in valid_usages)
    cache_write_tokens_30m = (
        sum(u.cache_write_tokens_30m for u in valid_usages if u.cache_write_tokens_30m is not None)
        if has_cw_30m
        else None
    )

    has_cw_12h = any(u.cache_write_tokens_12h is not None for u in valid_usages)
    cache_write_tokens_12h = (
        sum(u.cache_write_tokens_12h for u in valid_usages if u.cache_write_tokens_12h is not None)
        if has_cw_12h
        else None
    )

    is_provider_reported = all(u.is_provider_reported for u in valid_usages)

    has_derived = any(u.derived_total_tokens is not None for u in valid_usages)
    derived_total_tokens = (
        sum(
            (u.derived_total_tokens if u.derived_total_tokens is not None else (u.input_tokens + u.output_tokens))
            for u in valid_usages
        )
        if (has_derived or total_tokens is None)
        else None
    )

    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        reasoning_tokens=reasoning_tokens,
        is_provider_reported=is_provider_reported,
        cache_write_tokens=cache_write_tokens,
        cache_write_tokens_30m=cache_write_tokens_30m,
        cache_write_tokens_12h=cache_write_tokens_12h,
        derived_total_tokens=derived_total_tokens,
    )


def usage_to_dict(usage: Optional[LLMUsage]) -> Optional[Dict[str, Any]]:
    """Convert an LLMUsage instance to a serializable dictionary, or None if usage is None."""
    if usage is None:
        return None
    res = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "is_provider_reported": usage.is_provider_reported,
    }
    if usage.derived_total_tokens is not None:
        res["derived_total_tokens"] = usage.derived_total_tokens
    if usage.cache_write_tokens is not None:
        res["cache_write_tokens"] = usage.cache_write_tokens
    if usage.cache_write_tokens_30m is not None:
        res["cache_write_tokens_30m"] = usage.cache_write_tokens_30m
    if usage.cache_write_tokens_12h is not None:
        res["cache_write_tokens_12h"] = usage.cache_write_tokens_12h
    return res


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

    @property
    def sdk_version(self) -> Optional[str]:
        """Return SDK version string if applicable."""
        ...

    def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: str = "low",
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMCompletionResult:
        """Execute a completion request against the underlying model.

        Args:
            messages: List of LLMMessage objects representing conversation history.
            tools: Optional tool/function specifications.
            tool_choice: Optional tool selection constraint ('auto', 'required', or name).
            response_format: Optional structured output schema constraint.
            reasoning_effort: Effort parameter for reasoning models ('low', 'medium', 'high').
            max_output_tokens: Optional ceiling on generated output tokens.
            kwargs: Additional provider-specific parameters.

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

    @property
    def sdk_version(self) -> Optional[str]:
        """Return None for mock provider to avoid fabricating external SDK metadata."""
        return None

    def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: str = "low",
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMCompletionResult:
        """Return a deterministic completion without network access."""
        self.call_history.append({
            "messages": list(messages),
            "tools": list(tools) if tools else None,
            "tool_choice": tool_choice,
            "response_format": response_format,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            **kwargs,
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
    DEFAULT_OUTPUT_TOKEN_CEILING: int = 2048

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = "low",
        timeout_seconds: float = 60.0,
        client: Optional[Any] = None,
        max_output_tokens: Optional[int] = DEFAULT_OUTPUT_TOKEN_CEILING,
        max_retries: int = 0,
    ) -> None:
        """Initialize OpenAIProvider.

        Args:
            api_key: Optional explicit API key. If omitted, reads OPENAI_API_KEY from env.
            model: Target model identifier (default: 'gpt-6-luna').
            reasoning_effort: Fixed reasoning effort (default: 'low').
            timeout_seconds: Request timeout in seconds.
            client: Optional injected OpenAI client or test double for offline testing.
            max_output_tokens: Output token ceiling per request (default: 2048).
            max_retries: Maximum SDK retries (default: 0 to prevent uncontrolled retry loops).
        """
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self._explicit_api_key = api_key
        self._injected_client = client
        self._client: Optional[Any] = client
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries

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
            max_retries=self.max_retries,
        )
        return self._client

    @property
    def sdk_version(self) -> Optional[str]:
        """Return the installed version of the official OpenAI SDK."""
        try:
            import openai
            return getattr(openai, "__version__", None)
        except ImportError:
            return None

    def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> LLMCompletionResult:
        """Execute a completion using the OpenAI Responses API.

        Strictly enforces:
        - Reasoning configuration effort='low'.
        - Temperature parameter omitted entirely.
        - Error mapping with zero secret leakage.
        """
        client = self._get_client()
        effort = reasoning_effort or self.reasoning_effort
        effective_max_tokens = max_output_tokens if max_output_tokens is not None else self.max_output_tokens

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

        if effective_max_tokens is not None:
            request_kwargs["max_output_tokens"] = effective_max_tokens

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
                if reasoning_tok is None and isinstance(out_details, dict):
                    reasoning_tok = out_details.get("reasoning_tokens")

            # Cache write tokens (if reported)
            cw_tok: Optional[int] = None
            cw_30m: Optional[int] = None
            cw_12h: Optional[int] = None

            if in_details is not None:
                if cached_tok is None and isinstance(in_details, dict):
                    cached_tok = in_details.get("cached_tokens")

                cw_tok = getattr(in_details, "cache_write_tokens", None) or getattr(in_details, "cache_creation_tokens", None)
                if cw_tok is None and isinstance(in_details, dict):
                    cw_tok = in_details.get("cache_write_tokens") or in_details.get("cache_creation_tokens")

                cw_30m = getattr(in_details, "cache_write_tokens_30m", None)
                if cw_30m is None and isinstance(in_details, dict):
                    cw_30m = in_details.get("cache_write_tokens_30m")

                cw_12h = getattr(in_details, "cache_write_tokens_12h", None)
                if cw_12h is None and isinstance(in_details, dict):
                    cw_12h = in_details.get("cache_write_tokens_12h")

            if cw_tok is None:
                cw_tok = getattr(usage_obj, "cache_write_tokens", None) or getattr(usage_obj, "cache_creation_tokens", None)
            if cw_30m is None:
                cw_30m = getattr(usage_obj, "cache_write_tokens_30m", None)
            if cw_12h is None:
                cw_12h = getattr(usage_obj, "cache_write_tokens_12h", None)

            if cw_tok is None and (cw_30m is not None or cw_12h is not None):
                cw_tok = (int(cw_30m) if cw_30m is not None else 0) + (int(cw_12h) if cw_12h is not None else 0)

            usage = LLMUsage(
                input_tokens=int(in_tok),
                output_tokens=int(out_tok),
                total_tokens=int(tot_tok),
                cached_input_tokens=int(cached_tok) if cached_tok is not None else None,
                reasoning_tokens=int(reasoning_tok) if reasoning_tok is not None else None,
                is_provider_reported=True,
                cache_write_tokens=int(cw_tok) if cw_tok is not None else None,
                cache_write_tokens_30m=int(cw_30m) if cw_30m is not None else None,
                cache_write_tokens_12h=int(cw_12h) if cw_12h is not None else None,
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


# =============================================================================
# 6. GOOGLE GEMINI PROVIDER ADAPTER
# =============================================================================

def convert_to_gemini_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Any]]:
    """Convert Stage 4/5/6 tool specifications to google-genai Tool declarations.

    Ensures compatibility with client.models.generate_content(config=...).
    Does not enable grounding, code execution, web search, or other paid tools.
    """
    if not tools:
        return None
    from google.genai import types

    func_decls: List[Dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool.get("function", tool)
            func_decls.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        elif "name" in tool:
            func_decls.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            })

    if not func_decls:
        return None

    return [types.Tool(function_declarations=func_decls)]


class GeminiProvider:
    """Provider adapter connecting to Google GenAI SDK for gemini-3.8-flash.

    Enforces ShopConnect architectural constraints:
    - Target model: gemini-3.8-flash (default).
    - Thinking level: 'low' by default via types.ThinkingConfig(thinking_level=...).
    - Manual function calling: automatic_function_calling disabled so trusted Python
      and condition agents retain exclusive control over tool dispatch and policy validation.
    - Output token ceiling: 2048 default (types.GenerateContentConfig.max_output_tokens).
    - Temperature parameter omitted.
    - Credentials loaded strictly from GEMINI_API_KEY environment variable.
    - Secrets never logged, echoed, or exposed in exceptions.
    - Exact provider-reported usage capture (input, output, cached, thinking/reasoning).
    - Missing usage fields left as None (never fabricated).
    - Supports dependency injection of client for 100% offline unit testing.
    """

    DEFAULT_MODEL: str = "gemini-3.8-flash"
    DEFAULT_OUTPUT_TOKEN_CEILING: int = 2048
    DEFAULT_THINKING_LEVEL: str = "low"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        thinking_level: str = DEFAULT_THINKING_LEVEL,
        max_output_tokens: Optional[int] = DEFAULT_OUTPUT_TOKEN_CEILING,
        timeout_seconds: float = 60.0,
        client: Optional[Any] = None,
        max_retries: int = 0,
    ) -> None:
        """Initialize GeminiProvider.

        Args:
            api_key: Optional explicit API key. If omitted, reads GEMINI_API_KEY from env.
            model: Target model identifier (default: 'gemini-3.8-flash').
            thinking_level: Thinking effort level ('low', 'medium', 'high', default: 'low').
            max_output_tokens: Output token ceiling per request (default: 2048).
            timeout_seconds: Request timeout in seconds.
            client: Optional injected GenAI client or test double for offline testing.
            max_retries: Maximum SDK retries (default: 0 to enforce single-attempt execution).
        """
        self.model = model
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self._explicit_api_key = api_key
        self._injected_client = client
        self._client: Optional[Any] = client
        self.max_retries = max_retries

    def __repr__(self) -> str:
        """Return safe string representation without exposing API credentials."""
        return f"GeminiProvider(model={self.model!r}, thinking_level={self.thinking_level!r})"

    def _get_client(self) -> Any:
        """Obtain or lazily instantiate the GenAI client with credential validation."""
        if self._client is not None:
            return self._client

        api_key = self._explicit_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key or not api_key.strip():
            raise ConfigurationError(
                "GEMINI_API_KEY environment variable is not set. "
                "Provide an API key or inject a client for testing."
            )

        from google import genai
        self._client = genai.Client(api_key=api_key.strip())
        return self._client

    @property
    def sdk_version(self) -> Optional[str]:
        """Return the installed version of the official google-genai SDK."""
        try:
            from google import genai
            return getattr(genai, "__version__", None)
        except ImportError:
            return None

    def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMCompletionResult:
        """Execute a completion using the Google GenAI SDK.

        Strictly enforces:
        - Manual tool-call handling (automatic_function_calling disabled).
        - Thinking level configuration (effort='low').
        - Temperature parameter omitted.
        - Error mapping with zero secret leakage.
        """
        client = self._get_client()
        effort = (reasoning_effort or self.thinking_level).lower()
        effective_max_tokens = max_output_tokens if max_output_tokens is not None else self.max_output_tokens

        from google.genai import types

        # 1. Format contents and separate system instructions
        instructions_parts: List[str] = []
        contents: List[types.Content] = []

        for msg in messages:
            if msg.role == "system":
                if msg.content:
                    instructions_parts.append(msg.content)
            elif msg.role in ("user", "human"):
                parts: List[types.Part] = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                contents.append(types.Content(role="user", parts=parts))
            elif msg.role in ("assistant", "model"):
                parts = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        cid = tc.get("id") or tc.get("call_id")
                        fn_data = tc.get("function", tc)
                        fn_name = fn_data.get("name", tc.get("name", ""))
                        raw_args = fn_data.get("arguments", tc.get("arguments", {}))
                        if isinstance(raw_args, str):
                            try:
                                args_dict = json.loads(raw_args)
                            except Exception:
                                args_dict = {"raw_arguments": raw_args}
                        elif isinstance(raw_args, dict):
                            args_dict = dict(raw_args)
                        else:
                            args_dict = {}
                        parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    id=cid,
                                    name=fn_name,
                                    args=args_dict,
                                )
                            )
                        )
                contents.append(types.Content(role="model", parts=parts))
            elif msg.role == "tool":
                cid = msg.tool_call_id
                fn_name = msg.name or "tool_response"
                raw_out = msg.content or ""
                if isinstance(raw_out, str):
                    try:
                        resp_dict = json.loads(raw_out)
                        if not isinstance(resp_dict, dict):
                            resp_dict = {"output": raw_out}
                    except Exception:
                        resp_dict = {"output": raw_out}
                elif isinstance(raw_out, dict):
                    resp_dict = dict(raw_out)
                else:
                    resp_dict = {"output": str(raw_out)}

                fr_part = types.Part(
                    function_response=types.FunctionResponse(
                        id=cid,
                        name=fn_name,
                        response=resp_dict,
                    )
                )
                contents.append(types.Content(role="tool", parts=[fr_part]))

        instructions = "\n\n".join(instructions_parts) if instructions_parts else None

        # 2. Build GenerateContentConfig
        config_kwargs: Dict[str, Any] = {
            "max_output_tokens": effective_max_tokens,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if instructions:
            config_kwargs["system_instruction"] = instructions

        # Thinking config (low / medium / high)
        if effort in ("low", "minimal"):
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW,
            )
        elif effort == "medium":
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MEDIUM,
            )
        elif effort == "high":
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH,
            )

        gemini_tools = convert_to_gemini_tools(tools)
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools

        if tool_choice:
            tc_lower = tool_choice.lower()
            if tc_lower == "auto":
                config_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
                )
            elif tc_lower in ("required", "any"):
                config_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.ANY)
                )
            elif tc_lower == "none":
                config_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.NONE)
                )
            else:
                config_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=[tool_choice],
                    )
                )

        if response_format:
            fmt_type = response_format.get("type")
            if fmt_type in ("json_object", "application/json"):
                config_kwargs["response_mime_type"] = "application/json"
            if "schema" in response_format:
                config_kwargs["response_schema"] = response_format["schema"]

        gen_config = types.GenerateContentConfig(**config_kwargs)

        # 3. Execute request with stopwatch timing and exception mapping
        t0 = time.perf_counter()
        try:
            raw_response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=gen_config,
            )
        except Exception as exc:
            self._handle_api_exception(exc)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Parse response output text and tool calls
        return self._parse_response(raw_response, latency_ms)

    def _handle_api_exception(self, exc: Exception) -> None:
        """Map underlying SDK exceptions to typed ProviderError classes without leaking credentials."""
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        status_code = getattr(exc, "code", getattr(exc, "status_code", getattr(exc, "status", None)))

        clean_detail = exc_msg.split("\n")[0] if exc_msg else exc_type
        for secret in filter(None, [self._explicit_api_key, os.environ.get("GEMINI_API_KEY")]):
            clean_detail = clean_detail.replace(secret, "[REDACTED_API_KEY]")

        # 1. Authentication / Unauthorized (401, 403, API_KEY_INVALID, PERMISSION_DENIED)
        if (
            status_code in (401, 403)
            or "API_KEY_INVALID" in exc_msg
            or "PERMISSION_DENIED" in exc_msg
            or "Unauthorized" in exc_msg
            or "unauthenticated" in exc_msg.lower()
        ):
            raise AuthenticationError("Authentication failed: invalid or unauthorized Gemini API key.") from None

        # 2. Rate limit / Quota exceeded (429, RESOURCE_EXHAUSTED)
        if (
            status_code == 429
            or "RESOURCE_EXHAUSTED" in exc_msg
            or "429" in exc_msg
            or "rate limit" in exc_msg.lower()
            or "quota" in exc_msg.lower()
        ):
            raise RateLimitError("Gemini API rate limit or quota exceeded.") from None

        # 3. Timeout (408, 504, DEADLINE_EXCEEDED)
        if (
            status_code in (408, 504)
            or "DEADLINE_EXCEEDED" in exc_msg
            or "timed out" in exc_msg.lower()
            or "timeout" in exc_msg.lower()
        ):
            raise TimeoutError("Gemini API request timed out.") from None

        # 4. Connection failure
        if (
            "connection" in exc_msg.lower()
            or "unavailable" in exc_msg.lower()
            or "connect" in exc_msg.lower()
            or "httpx" in exc_type.lower()
        ):
            raise ConnectionError("Failed to connect to Gemini API endpoint.") from None

        # 5. Configuration / Model not found (404, NOT_FOUND)
        if status_code == 404 or "NOT_FOUND" in exc_msg:
            raise ConfigurationError(f"Gemini model '{self.model}' not found or unavailable.") from None

        # 6. Safety block
        if "safety" in exc_msg.lower() or "blocked" in exc_msg.lower():
            raise ProviderError(f"Gemini request blocked by safety policy: {clean_detail}") from None

        raise ProviderError(f"Gemini API provider error ({exc_type}): {clean_detail}") from None

    def _parse_response(self, response: Any, latency_ms: float) -> LLMCompletionResult:
        """Extract standardized LLMCompletionResult from Gemini GenerateContentResponse object."""
        tool_calls: List[Dict[str, Any]] = []
        content_parts: List[str] = []
        finish_reason = "stop"

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            cand = candidates[0]
            raw_fr = getattr(cand, "finish_reason", None)
            if raw_fr is not None:
                fr_str = str(raw_fr.value if hasattr(raw_fr, "value") else raw_fr)
                finish_reason = fr_str.lower()
                if "safety" in finish_reason or "blocked" in finish_reason:
                    raise ProviderError(f"Gemini generation was blocked by safety filters (finish_reason: {fr_str}).")

            content_obj = getattr(cand, "content", None)
            if content_obj:
                parts = getattr(content_obj, "parts", None) or []
                for part in parts:
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        cid = getattr(fc, "id", None)
                        fn_name = getattr(fc, "name", "")
                        raw_args = getattr(fc, "args", {})
                        if isinstance(raw_args, dict):
                            args_str = json.dumps(raw_args)
                        elif isinstance(raw_args, str):
                            args_str = raw_args
                        else:
                            args_str = "{}"
                        tool_calls.append({
                            "id": cid,
                            "call_id": cid,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": args_str,
                            },
                        })
                    else:
                        is_thought = getattr(part, "thought", False)
                        text_val = getattr(part, "text", None)
                        if text_val and not is_thought:
                            content_parts.append(text_val)

        text_content = "".join(content_parts) if content_parts else getattr(response, "text", None)
        if not text_content and not tool_calls and candidates:
            text_content = getattr(response, "text", None)

        msg = LLMMessage(
            role="assistant",
            content=text_content,
            tool_calls=tuple(tool_calls) if tool_calls else None,
        )

        # 2. Token usage extraction
        usage: Optional[LLMUsage] = None
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta is not None:
            in_tok = getattr(usage_meta, "prompt_token_count", None) or 0
            out_tok = getattr(usage_meta, "candidates_token_count", None) or 0
            tot_tok = getattr(usage_meta, "total_token_count", None)

            cached_tok = getattr(usage_meta, "cached_content_token_count", None)
            reasoning_tok = getattr(usage_meta, "thoughts_token_count", None)

            # Strict provenance: If Gemini does not return total_token_count,
            # preserve LLMUsage.total_tokens as None. Do not silently label a derived
            # input + output sum as provider-reported usage. Expose derived_total_tokens separately.
            total_tokens_val = int(tot_tok) if tot_tok is not None else None
            derived_total = int(in_tok + out_tok)

            usage = LLMUsage(
                input_tokens=int(in_tok),
                output_tokens=int(out_tok),
                total_tokens=total_tokens_val,
                cached_input_tokens=int(cached_tok) if cached_tok is not None else None,
                reasoning_tokens=int(reasoning_tok) if reasoning_tok is not None else None,
                is_provider_reported=True,
                derived_total_tokens=derived_total,
            )

        model_id = getattr(response, "model_version", None) or self.model
        req_id = getattr(response, "response_id", None)

        raw_meta: Dict[str, Any] = {}
        if req_id:
            raw_meta["response_id"] = req_id
        if hasattr(response, "create_time") and response.create_time:
            raw_meta["create_time"] = str(response.create_time)

        return LLMCompletionResult(
            message=msg,
            usage=usage,
            model_identifier=model_id,
            system_fingerprint=None,  # Gemini API does not expose system fingerprint; None documented
            provider_request_id=req_id,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            raw_response=raw_meta,
        )
