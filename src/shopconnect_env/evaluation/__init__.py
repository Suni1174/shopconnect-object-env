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
    "ExpectedOutcome",
    "ExperimentRunner",
    "FixtureValidationError",
    "OutcomeEvaluator",
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
    "load_benchmark_tasks",
    "load_plan_fixtures",
    "run_offline_smoke_test",
]
