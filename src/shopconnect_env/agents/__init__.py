"""Agent interfaces, response containers, providers, and mock implementations.

Exports BaseAgent, AgentResponse, ObjectInspector, MockAgent, LLMProvider,
MockLLMProvider, OpenAIProvider, and associated exceptions/contracts.
"""

from shopconnect_env.agents.base import (
    AgentResponse,
    BaseAgent,
    ObjectInspector,
)
from shopconnect_env.agents.mock_agent import MockAgent
from shopconnect_env.agents.providers import (
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    LLMCompletionResult,
    LLMMessage,
    LLMProvider,
    LLMUsage,
    MalformedResponseError,
    MockLLMProvider,
    OpenAIProvider,
    ProviderError,
    RateLimitError,
    TimeoutError,
    convert_to_responses_api_tools,
)

__all__ = [
    "AgentResponse",
    "AuthenticationError",
    "BaseAgent",
    "ConfigurationError",
    "ConnectionError",
    "LLMCompletionResult",
    "LLMMessage",
    "LLMProvider",
    "LLMUsage",
    "MalformedResponseError",
    "MockAgent",
    "MockLLMProvider",
    "ObjectInspector",
    "OpenAIProvider",
    "ProviderError",
    "RateLimitError",
    "TimeoutError",
    "convert_to_responses_api_tools",
]
