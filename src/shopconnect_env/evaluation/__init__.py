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

from shopconnect_env.evaluation.gemini_pilot import (
    CONNECTION_TEST_PROMPT,
    EXPECTED_CONNECTION_TEST_TOKEN,
    GeminiPilotRequestGuardrailProvider,
    MAX_REQUESTS_PER_CONDITION_C,
    MAX_THEORETICAL_TOTAL_REQUESTS,
    SequentialRateLimiter,
    create_gemini_pilot_mock_provider,
    run_gemini_connection_test,
    run_gemini_pilot_dry_run,
    run_gemini_pilot_preflight,
    run_live_gemini_pilot,
)
from shopconnect_env.evaluation.metrics import (
    AggregateMetrics,
    PricingConfig,
    RunMetrics,
    calculate_cost,
    calculate_percentile,
    compute_aggregate_metrics,
    create_gemini_38_flash_pricing,
    create_gpt6_luna_pricing,
)
from shopconnect_env.evaluation.pilot import (
    DEFAULT_OUTPUT_TOKEN_CEILING,
    MAX_PILOT_RUNS,
    MAX_TOTAL_PROVIDER_REQUESTS,
    PILOT_CONDITIONS,
    PILOT_MODEL,
    PILOT_TAG,
    PILOT_TASK_IDS,
    PilotRequestGuardrailProvider,
    create_pilot_mock_provider,
    run_live_pilot,
    run_pilot_dry_run,
    run_pilot_preflight,
)
from shopconnect_env.evaluation.runner import (
    ExperimentRunner,
    OutcomeEvaluator,
    PlanDiffResult,
    PlanDiffer,
    RunRecord,
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
    "MAX_PILOT_RUNS",
    "MAX_REQUESTS_PER_CONDITION_C",
    "MAX_THEORETICAL_TOTAL_REQUESTS",
    "MAX_TOTAL_PROVIDER_REQUESTS",
    "OutcomeEvaluator",
    "PILOT_CONDITIONS",
    "PILOT_MODEL",
    "PILOT_TAG",
    "PILOT_TASK_IDS",
    "PilotRequestGuardrailProvider",
    "PlanDiffResult",
    "PlanDiffer",
    "PlanFixture",
    "PricingConfig",
    "RunMetrics",
    "RunRecord",
    "SequentialRateLimiter",
    "TaskValidationError",
    "calculate_cost",
    "calculate_percentile",
    "compute_aggregate_metrics",
    "create_gemini_38_flash_pricing",
    "create_gemini_pilot_mock_provider",
    "create_gpt6_luna_pricing",
    "create_pilot_mock_provider",
    "load_benchmark_tasks",
    "load_plan_fixtures",
    "run_gemini_connection_test",
    "run_gemini_pilot_dry_run",
    "run_gemini_pilot_preflight",
    "run_live_gemini_pilot",
    "run_live_pilot",
    "run_offline_smoke_test",
    "run_pilot_dry_run",
    "run_pilot_preflight",
]
