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

from shopconnect_env.evaluation.metrics import (
    AggregateMetrics,
    PricingConfig,
    RunMetrics,
    calculate_cost,
    calculate_percentile,
    compute_aggregate_metrics,
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
    "DEFAULT_OUTPUT_TOKEN_CEILING",
    "ExpectedOutcome",
    "ExperimentRunner",
    "FixtureValidationError",
    "MAX_PILOT_RUNS",
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
    "TaskValidationError",
    "calculate_cost",
    "calculate_percentile",
    "compute_aggregate_metrics",
    "create_gpt6_luna_pricing",
    "create_pilot_mock_provider",
    "load_benchmark_tasks",
    "load_plan_fixtures",
    "run_live_pilot",
    "run_offline_smoke_test",
    "run_pilot_dry_run",
    "run_pilot_preflight",
]
