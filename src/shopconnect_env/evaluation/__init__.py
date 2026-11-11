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
    "BenchmarkTask",
    "ExpectedOutcome",
    "FixtureValidationError",
    "PlanFixture",
    "TaskValidationError",
    "load_benchmark_tasks",
    "load_plan_fixtures",
]
