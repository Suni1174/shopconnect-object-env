"""Evaluation framework, benchmark fixtures, and metrics tracking for Stage 6.

Exports:
- BenchmarkTask
- ExpectedOutcome
- FixtureValidationError
- PlanFixture
- TaskValidationError
- load_benchmark_tasks
- load_plan_fixtures
"""

import importlib
from typing import Any

from shopconnect_env.evaluation.metrics import (
    AggregateMetrics,
    PricingConfig,
    RunMetrics,
    calculate_cost,
    calculate_percentile,
    compute_aggregate_metrics,
    create_gemini_38_flash_pricing,
    create_gpt6_luna_pricing,
    create_groq_free_pricing,
    create_ollama_local_pricing,
    create_openrouter_free_pricing,
)
from shopconnect_env.evaluation.runner import (
    ExperimentRunner,
    OutcomeEvaluator,
    PlanDiffResult,
    PlanDiffer,
    RunRecord,
    format_conditions_label,
    run_offline_smoke_test,
)
from shopconnect_env.evaluation.tasks import (
    BenchmarkTask,
    ExpectedOutcome,
    FixtureValidationError,
    PlanFixture,
    TaskValidationError,
    load_benchmark_tasks,
    load_plan_fixtures,
)

# Lazy re-exports for CLI pilot modules (prevents runpy RuntimeWarning on python -m)
_LAZY_EXPORTS = {
    # gemini_pilot
    "CONNECTION_TEST_PROMPT": ("shopconnect_env.evaluation.gemini_pilot", "CONNECTION_TEST_PROMPT"),
    "EXPECTED_CONNECTION_TEST_TOKEN": ("shopconnect_env.evaluation.gemini_pilot", "EXPECTED_CONNECTION_TEST_TOKEN"),
    "GeminiPilotRequestGuardrailProvider": ("shopconnect_env.evaluation.gemini_pilot", "GeminiPilotRequestGuardrailProvider"),
    "MAX_REQUESTS_PER_CONDITION_C": ("shopconnect_env.evaluation.gemini_pilot", "MAX_REQUESTS_PER_CONDITION_C"),
    "MAX_THEORETICAL_TOTAL_REQUESTS": ("shopconnect_env.evaluation.gemini_pilot", "MAX_THEORETICAL_TOTAL_REQUESTS"),
    "SequentialRateLimiter": ("shopconnect_env.evaluation.gemini_pilot", "SequentialRateLimiter"),
    "create_gemini_pilot_mock_provider": ("shopconnect_env.evaluation.gemini_pilot", "create_gemini_pilot_mock_provider"),
    "run_gemini_connection_test": ("shopconnect_env.evaluation.gemini_pilot", "run_gemini_connection_test"),
    "run_gemini_pilot_dry_run": ("shopconnect_env.evaluation.gemini_pilot", "run_gemini_pilot_dry_run"),
    "run_gemini_pilot_preflight": ("shopconnect_env.evaluation.gemini_pilot", "run_gemini_pilot_preflight"),
    "run_live_gemini_pilot": ("shopconnect_env.evaluation.gemini_pilot", "run_live_gemini_pilot"),
    # groq_pilot
    "INSPECT_PLAN_TOOL_CHOICE": ("shopconnect_env.evaluation.groq_pilot", "INSPECT_PLAN_TOOL_CHOICE"),
    "INSPECT_PLAN_TOOL_DEFINITION": ("shopconnect_env.evaluation.groq_pilot", "INSPECT_PLAN_TOOL_DEFINITION"),
    "GroqPilotRequestGuardrailProvider": ("shopconnect_env.evaluation.groq_pilot", "GroqPilotRequestGuardrailProvider"),
    "create_groq_pilot_mock_provider": ("shopconnect_env.evaluation.groq_pilot", "create_groq_pilot_mock_provider"),
    "run_groq_connection_test": ("shopconnect_env.evaluation.groq_pilot", "run_groq_connection_test"),
    "run_groq_json_smoke_test": ("shopconnect_env.evaluation.groq_pilot", "run_groq_json_smoke_test"),
    "run_groq_pilot_dry_run": ("shopconnect_env.evaluation.groq_pilot", "run_groq_pilot_dry_run"),
    "run_groq_pilot_preflight": ("shopconnect_env.evaluation.groq_pilot", "run_groq_pilot_preflight"),
    "run_groq_tool_calling_smoke_test": ("shopconnect_env.evaluation.groq_pilot", "run_groq_tool_calling_smoke_test"),
    "run_live_groq_pilot": ("shopconnect_env.evaluation.groq_pilot", "run_live_groq_pilot"),
    # ollama_pilot
    "OllamaPilotRequestGuardrailProvider": ("shopconnect_env.evaluation.ollama_pilot", "OllamaPilotRequestGuardrailProvider"),
    "check_ollama_server_status": ("shopconnect_env.evaluation.ollama_pilot", "check_ollama_server_status"),
    "create_ollama_pilot_mock_provider": ("shopconnect_env.evaluation.ollama_pilot", "create_ollama_pilot_mock_provider"),
    "run_live_ollama_pilot": ("shopconnect_env.evaluation.ollama_pilot", "run_live_ollama_pilot"),
    "run_ollama_connection_test": ("shopconnect_env.evaluation.ollama_pilot", "run_ollama_connection_test"),
    "run_ollama_pilot_dry_run": ("shopconnect_env.evaluation.ollama_pilot", "run_ollama_pilot_dry_run"),
    "run_ollama_pilot_preflight": ("shopconnect_env.evaluation.ollama_pilot", "run_ollama_pilot_preflight"),
    # openrouter_pilot
    "OpenRouterPilotRequestGuardrailProvider": ("shopconnect_env.evaluation.openrouter_pilot", "OpenRouterPilotRequestGuardrailProvider"),
    "create_openrouter_pilot_mock_provider": ("shopconnect_env.evaluation.openrouter_pilot", "create_openrouter_pilot_mock_provider"),
    "run_live_openrouter_pilot": ("shopconnect_env.evaluation.openrouter_pilot", "run_live_openrouter_pilot"),
    "run_openrouter_connection_test": ("shopconnect_env.evaluation.openrouter_pilot", "run_openrouter_connection_test"),
    "run_openrouter_pilot_dry_run": ("shopconnect_env.evaluation.openrouter_pilot", "run_openrouter_pilot_dry_run"),
    "run_openrouter_pilot_preflight": ("shopconnect_env.evaluation.openrouter_pilot", "run_openrouter_pilot_preflight"),
    # pilot
    "DEFAULT_OUTPUT_TOKEN_CEILING": ("shopconnect_env.evaluation.pilot", "DEFAULT_OUTPUT_TOKEN_CEILING"),
    "MAX_PILOT_RUNS": ("shopconnect_env.evaluation.pilot", "MAX_PILOT_RUNS"),
    "MAX_TOTAL_PROVIDER_REQUESTS": ("shopconnect_env.evaluation.pilot", "MAX_TOTAL_PROVIDER_REQUESTS"),
    "PILOT_CONDITIONS": ("shopconnect_env.evaluation.pilot", "PILOT_CONDITIONS"),
    "PILOT_MODEL": ("shopconnect_env.evaluation.pilot", "PILOT_MODEL"),
    "PILOT_TAG": ("shopconnect_env.evaluation.pilot", "PILOT_TAG"),
    "PILOT_TASK_IDS": ("shopconnect_env.evaluation.pilot", "PILOT_TASK_IDS"),
    "PilotRequestGuardrailProvider": ("shopconnect_env.evaluation.pilot", "PilotRequestGuardrailProvider"),
    "create_pilot_mock_provider": ("shopconnect_env.evaluation.pilot", "create_pilot_mock_provider"),
    "run_live_pilot": ("shopconnect_env.evaluation.pilot", "run_live_pilot"),
    "run_pilot_dry_run": ("shopconnect_env.evaluation.pilot", "run_pilot_dry_run"),
    "run_pilot_preflight": ("shopconnect_env.evaluation.pilot", "run_pilot_preflight"),
}


def __getattr__(name: str) -> Any:
    """Lazily import and resolve pilot-specific CLI symbols on demand (PEP 562)."""
    if name in _LAZY_EXPORTS:
        mod_name, attr_name = _LAZY_EXPORTS[name]
        mod = importlib.import_module(mod_name)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AggregateMetrics",
    "BenchmarkTask",
    "CONNECTION_TEST_PROMPT",
    "DEFAULT_OUTPUT_TOKEN_CEILING",
    "EXPECTED_CONNECTION_TEST_TOKEN",
    "ExpectedOutcome",
    "ExperimentRunner",
    "FixtureValidationError",
    "GeminiPilotRequestGuardrailProvider",
    "GroqPilotRequestGuardrailProvider",
    "MAX_PILOT_RUNS",
    "MAX_REQUESTS_PER_CONDITION_C",
    "MAX_THEORETICAL_TOTAL_REQUESTS",
    "MAX_TOTAL_PROVIDER_REQUESTS",
    "OllamaPilotRequestGuardrailProvider",
    "OpenRouterPilotRequestGuardrailProvider",
    "OutcomeEvaluator",
    "PILOT_CONDITIONS",
    "PILOT_MODEL",
    "PILOT_TAG",
    "PILOT_TASK_IDS",
    "PilotRequestGuardrailProvider",
    "PlanDiffResult",
    "INSPECT_PLAN_TOOL_CHOICE",
    "INSPECT_PLAN_TOOL_DEFINITION",
    "PlanDiffer",
    "PlanFixture",
    "PricingConfig",
    "RunMetrics",
    "RunRecord",
    "SequentialRateLimiter",
    "TaskValidationError",
    "calculate_cost",
    "calculate_percentile",
    "check_ollama_server_status",
    "compute_aggregate_metrics",
    "create_gemini_38_flash_pricing",
    "create_gemini_pilot_mock_provider",
    "create_gpt6_luna_pricing",
    "create_groq_free_pricing",
    "create_groq_pilot_mock_provider",
    "create_ollama_local_pricing",
    "create_ollama_pilot_mock_provider",
    "create_openrouter_free_pricing",
    "create_openrouter_pilot_mock_provider",
    "create_pilot_mock_provider",
    "load_benchmark_tasks",
    "load_plan_fixtures",
    "run_gemini_connection_test",
    "run_gemini_pilot_dry_run",
    "run_gemini_pilot_preflight",
    "run_groq_connection_test",
    "run_groq_json_smoke_test",
    "run_groq_pilot_dry_run",
    "run_groq_pilot_preflight",
    "run_groq_tool_calling_smoke_test",
    "run_live_gemini_pilot",
    "run_live_groq_pilot",
    "run_live_ollama_pilot",
    "run_live_openrouter_pilot",
    "run_live_pilot",
    "run_offline_smoke_test",
    "run_ollama_connection_test",
    "run_ollama_pilot_dry_run",
    "run_ollama_pilot_preflight",
    "run_openrouter_connection_test",
    "run_openrouter_pilot_dry_run",
    "run_openrouter_pilot_preflight",
    "run_pilot_dry_run",
    "run_pilot_preflight",
]
