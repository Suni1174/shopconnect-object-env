"""Controlled real OpenRouter API pilot execution harness and guardrails.

Stage 6 Phase 6.5:
- Safe, bounded execution of exactly 3 frozen benchmark tasks across 3 conditions (9 runs).
- Selected tasks:
  * 1 COMMIT: task_01_commit_replace_wed_dinner (meal_replacement)
  * 1 REJECT: task_14_reject_allergen_peanut_toast (allergen_conflict)
  * 1 CLARIFY: task_22_clarify_missing_meal_slot (ambiguous_request)
- Strict maximum 20 total provider requests across the full pilot.
- Sequential execution only (zero parallelism).
- SDK retries disabled / single-attempt execution.
- Output token ceiling enforced (max_output_tokens=2048).
- Model: qwen/qwen3.8-27b:free (never drop :free; never use openrouter/free).
- Dual confirmation flags (--live AND --confirm-live) strictly required for live API calls.
- Offline preflight validation and simulated dry-run modes with zero network access.
- Dedicated artifacts under artifacts/phase6_5_openrouter_pilot/<pilot_run_id>/.
- Distinctly tagged: REAL_API_PILOT (OPENROUTER) — LIMITED_SAMPLE — NOT_FULL_EXPERIMENT.
- Pricing: $0.00 base rates for the documented free variant.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

# Ensure src/ is in sys.path for direct CLI script execution
_src_root = str(Path(__file__).resolve().parent.parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from shopconnect_env.agents.providers import (
    LLMCompletionResult,
    LLMMessage,
    LLMProvider,
    LLMUsage,
    MockLLMProvider,
    OpenRouterProvider,
    usage_to_dict,
)
from shopconnect_env.domain.models import load_catalog_from_json
from shopconnect_env.evaluation.metrics import (
    PricingConfig,
    RunMetrics,
    compute_aggregate_metrics,
    create_openrouter_free_pricing,
)
from shopconnect_env.evaluation.runner import (
    ExperimentRunner,
    PlanFixture,
    RunRecord,
    compute_sha256,
    load_plan_fixtures,
)
from shopconnect_env.evaluation.tasks import (
    BenchmarkTask,
    ExpectedOutcome,
    load_benchmark_tasks,
)


# =============================================================================
# 1. PILOT CONSTANTS & SPECIFICATIONS
# =============================================================================

PILOT_TASK_IDS: Tuple[str, ...] = (
    "task_01_commit_replace_wed_dinner",
    "task_14_reject_allergen_peanut_toast",
    "task_22_clarify_missing_meal_slot",
)

PILOT_CONDITIONS: Tuple[str, ...] = ("A", "B", "C")

MAX_PILOT_RUNS: int = 9  # 3 tasks * 3 conditions
PILOT_TASKS_COUNT: int = len(PILOT_TASK_IDS)  # 3

# Per-condition request budgets for OpenRouter pilot
MAX_REQUESTS_PER_CONDITION_A: int = 1
MAX_REQUESTS_PER_CONDITION_B: int = 1
MAX_REQUESTS_PER_CONDITION_C: int = 4
MAX_INSPECTIONS_PER_CONDITION_C: int = 5

# Maximum possible total requests: 3 tasks × (1 A + 1 B + 4 C) = 18 requests
MAX_THEORETICAL_TOTAL_REQUESTS: int = PILOT_TASKS_COUNT * (
    MAX_REQUESTS_PER_CONDITION_A + MAX_REQUESTS_PER_CONDITION_B + MAX_REQUESTS_PER_CONDITION_C
)

# Global hard ceiling adapted to free variant quotas (20 RPM, 50 RPD documented; ceiling 20)
MAX_TOTAL_PROVIDER_REQUESTS: int = 20
GLOBAL_REQUEST_CEILING: int = MAX_TOTAL_PROVIDER_REQUESTS
DEFAULT_OUTPUT_TOKEN_CEILING: int = 2048
PILOT_MODEL: str = "qwen/qwen3.8-27b:free"
PILOT_TAG: str = "REAL_API_PILOT (OPENROUTER) — LIMITED_SAMPLE — NOT_FULL_EXPERIMENT"

# Standalone One-Request Connection Test Specifications
CONNECTION_TEST_PROMPT: str = "Reply with exactly: SHOPCONNECT_OPENROUTER_CONNECTION_OK"
EXPECTED_CONNECTION_TEST_TOKEN: str = "SHOPCONNECT_OPENROUTER_CONNECTION_OK"
CONNECTION_TEST_OUTPUT_TOKEN_CEILING: int = 64
DEFAULT_CONNECTION_TEST_ARTIFACT_DIR: str = "artifacts/phase6_5_openrouter_connection_test"

# Safe Rate Limiting Target
CONFIGURED_SAFE_TARGET_RPM: int = 3
MIN_REQUEST_INTERVAL_SECONDS: float = 20.0  # 60s / 3 = 20.0s minimum interval between request starts


# =============================================================================
# 2. SEQUENTIAL RATE LIMITER
# =============================================================================

class SequentialRateLimiter:
    """Enforces a safe sequential interval between provider requests."""

    def __init__(
        self,
        min_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS,
        clock: Optional[Callable[[], float]] = None,
        sleeper: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.min_interval_seconds = min_interval_seconds
        self.clock = clock or time.monotonic
        self.sleeper = sleeper or time.sleep
        self.last_request_start: Optional[float] = None
        self.total_sleep_seconds: float = 0.0
        self.sleep_events: List[float] = []

    def wait_for_slot(self) -> float:
        """Wait until at least min_interval_seconds has elapsed since the last request start."""
        if self.min_interval_seconds <= 0.0:
            self.last_request_start = self.clock()
            return 0.0

        now = self.clock()
        slept = 0.0
        if self.last_request_start is not None:
            elapsed = now - self.last_request_start
            wait_time = self.min_interval_seconds - elapsed
            if wait_time > 0:
                self.sleeper(wait_time)
                slept = wait_time
                now = self.clock()

        self.last_request_start = now
        self.total_sleep_seconds += slept
        if slept > 0.0:
            self.sleep_events.append(slept)
        return slept


# =============================================================================
# 3. PILOT REQUEST GUARDRAIL PROVIDER
# =============================================================================

class OpenRouterPilotRequestGuardrailProvider:
    """Decorates an LLMProvider to enforce strict invocation, token, and rate limits."""

    MAX_TOTAL_PROVIDER_REQUESTS: int = MAX_TOTAL_PROVIDER_REQUESTS

    def __init__(
        self,
        provider: LLMProvider,
        max_requests: int = MAX_TOTAL_PROVIDER_REQUESTS,
        rate_limiter: Optional[SequentialRateLimiter] = None,
        min_interval_seconds: float = 0.0,
        clock: Optional[Callable[[], float]] = None,
        sleeper: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.provider = provider
        self.max_requests = max_requests
        if rate_limiter is not None:
            self.rate_limiter: Optional[SequentialRateLimiter] = rate_limiter
        elif min_interval_seconds > 0.0:
            self.rate_limiter = SequentialRateLimiter(
                min_interval_seconds=min_interval_seconds,
                clock=clock,
                sleeper=sleeper,
            )
        else:
            self.rate_limiter = None

        self.total_requests: int = 0
        self.request_history: List[Dict[str, Any]] = []
        self.provider_request_ids: List[str] = []

        # Cumulative token counters
        self.cumulative_input_tokens: int = 0
        self.cumulative_output_tokens: int = 0
        self.cumulative_cached_input_tokens: int = 0
        self.cumulative_reasoning_tokens: int = 0
        self.cumulative_total_tokens: Optional[int] = None
        self.cumulative_derived_total_tokens: int = 0
        self.cumulative_sleep_seconds: float = 0.0

    @property
    def sdk_version(self) -> Optional[str]:
        return getattr(self.provider, "sdk_version", None)

    def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: str = "low",
        max_output_tokens: Optional[int] = None,
    ) -> LLMCompletionResult:
        if self.total_requests >= self.max_requests:
            raise RuntimeError(
                f"OpenRouter pilot execution safety ceiling reached: ceiling of {self.max_requests} "
                "total provider requests exceeded. Cleanly aborting pilot execution to "
                "prevent unexpected billing or runaways."
            )

        slept = 0.0
        if self.rate_limiter is not None:
            slept = self.rate_limiter.wait_for_slot()
            self.cumulative_sleep_seconds += slept

        t0 = time.perf_counter()
        try:
            result = self.provider.complete(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
            )
        finally:
            self.total_requests += 1

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if result.provider_request_id:
            self.provider_request_ids.append(result.provider_request_id)

        if result.usage is not None:
            self.cumulative_input_tokens += result.usage.input_tokens
            self.cumulative_output_tokens += result.usage.output_tokens
            if result.usage.cached_input_tokens:
                self.cumulative_cached_input_tokens += result.usage.cached_input_tokens
            if result.usage.reasoning_tokens:
                self.cumulative_reasoning_tokens += result.usage.reasoning_tokens
            if result.usage.total_tokens is not None:
                if self.cumulative_total_tokens is None:
                    self.cumulative_total_tokens = 0
                self.cumulative_total_tokens += result.usage.total_tokens
            derived = (
                result.usage.derived_total_tokens
                if result.usage.derived_total_tokens is not None
                else (result.usage.input_tokens + result.usage.output_tokens)
            )
            self.cumulative_derived_total_tokens += derived

        self.request_history.append({
            "request_index": self.total_requests,
            "provider_request_id": result.provider_request_id,
            "system_fingerprint": result.system_fingerprint,
            "latency_ms": latency_ms,
            "slept_seconds": slept,
            "usage": usage_to_dict(result.usage),
        })

        return result


# =============================================================================
# 4. PREFLIGHT VALIDATION (ZERO NETWORK ACTIVITY)
# =============================================================================

def run_openrouter_pilot_preflight(
    catalog_path: Union[str, Path] = "data/catalog_seed.json",
    fixtures_path: Union[str, Path] = "data/plan_fixtures.json",
    tasks_path: Union[str, Path] = "data/tasks.jsonl",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute preflight configuration and data integrity validation for OpenRouter."""
    catalog_p = Path(catalog_path)
    fixtures_p = Path(fixtures_path)
    tasks_p = Path(tasks_path)

    errors: List[str] = []

    # API key check (never log secrets)
    configured_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    key_present = bool(configured_key and configured_key.strip())

    catalog = {}
    ingredients = {}
    if not catalog_p.exists():
        errors.append(f"Catalog file not found: {catalog_p}")
    else:
        try:
            ingredients, catalog = load_catalog_from_json(catalog_p)
            if len(catalog) < 15:
                errors.append(f"Catalog contains only {len(catalog)} recipes; expected at least 15.")
        except Exception as e:
            errors.append(f"Failed to load catalog: {e}")

    fixtures = {}
    if not fixtures_p.exists():
        errors.append(f"Plan fixtures file not found: {fixtures_p}")
    elif catalog:
        try:
            fixtures = load_plan_fixtures(fixtures_p, catalog, ingredients)
            if "fixture_standard" not in fixtures:
                errors.append("Standard fixture 'fixture_standard' missing from fixtures.")
        except Exception as e:
            errors.append(f"Failed to load plan fixtures: {e}")

    pilot_tasks: List[BenchmarkTask] = []
    if not tasks_p.exists():
        errors.append(f"Tasks file not found: {tasks_p}")
    elif fixtures and catalog:
        try:
            all_tasks = load_benchmark_tasks(tasks_p, fixtures, catalog)
            task_map = {t.task_id: t for t in all_tasks}
            missing = [tid for tid in PILOT_TASK_IDS if tid not in task_map]
            if missing:
                errors.append(f"Required pilot task(s) missing from {tasks_p}: {missing}")
            else:
                pilot_tasks = [task_map[tid] for tid in PILOT_TASK_IDS]
                expected_matrix = {
                    "task_01_commit_replace_wed_dinner": ExpectedOutcome.COMMIT,
                    "task_14_reject_allergen_peanut_toast": ExpectedOutcome.REJECT,
                    "task_22_clarify_missing_meal_slot": ExpectedOutcome.CLARIFY,
                }
                for t in pilot_tasks:
                    exp = expected_matrix[t.task_id]
                    if t.expected_outcome != exp:
                        errors.append(
                            f"Task {t.task_id} has outcome {t.expected_outcome.value}; "
                            f"expected {exp.value} for pilot."
                        )
        except Exception as e:
            errors.append(f"Failed to load benchmark tasks: {e}")

    try:
        pricing = create_openrouter_free_pricing()
    except Exception as e:
        pricing = None
        errors.append(f"Invalid pricing configuration: {e}")

    return {
        "status": "passed" if not errors else "failed",
        "api_key_configured": key_present,
        "pilot_model": PILOT_MODEL,
        "planned_runs_count": MAX_PILOT_RUNS,
        "max_requests_per_condition_a": MAX_REQUESTS_PER_CONDITION_A,
        "max_requests_per_condition_b": MAX_REQUESTS_PER_CONDITION_B,
        "max_requests_per_condition_c": MAX_REQUESTS_PER_CONDITION_C,
        "max_inspections_per_condition_c": MAX_INSPECTIONS_PER_CONDITION_C,
        "max_theoretical_requests": MAX_THEORETICAL_TOTAL_REQUESTS,
        "max_possible_total_requests": MAX_THEORETICAL_TOTAL_REQUESTS,
        "request_budget_formula": (
            f"{len(PILOT_TASK_IDS)} tasks × ({MAX_REQUESTS_PER_CONDITION_A} A + "
            f"{MAX_REQUESTS_PER_CONDITION_B} B + {MAX_REQUESTS_PER_CONDITION_C} C) = "
            f"{MAX_THEORETICAL_TOTAL_REQUESTS} requests"
        ),
        "global_request_ceiling": MAX_TOTAL_PROVIDER_REQUESTS,
        "max_provider_requests_ceiling": MAX_TOTAL_PROVIDER_REQUESTS,
        "safety_margin_requests": MAX_TOTAL_PROVIDER_REQUESTS - MAX_THEORETICAL_TOTAL_REQUESTS,
        "pricing_complete": pricing.is_complete() if pricing else False,
        "billing_tier": "free",
        "tasks_loaded": len(pilot_tasks),
        "errors": errors,
    }


# =============================================================================
# 5. DETERMINISTIC MOCK FOR DRY-RUN TESTING
# =============================================================================

def create_openrouter_pilot_mock_provider() -> MockLLMProvider:
    """Create offline mock provider configured specifically for OpenRouter pilot dry run."""
    canned: Dict[str, LLMCompletionResult] = {
        "pasta": LLMCompletionResult(
            message=LLMMessage(
                role="assistant",
                content='{"action": "replace_meal", "day": "Wednesday", "slot": "Dinner", "recipe": "Vegetable Pasta Primavera"}',
            ),
            usage=LLMUsage(
                input_tokens=150,
                output_tokens=48,
                total_tokens=198,
                cached_input_tokens=35,
                reasoning_tokens=18,
                is_provider_reported=False,
            ),
            model_identifier="mock-qwen-3.8-27b:free",
            system_fingerprint=None,
            provider_request_id="openrouter_mock_req_01",
            latency_ms=15.0,
            finish_reason="stop",
        ),
        "peanut": LLMCompletionResult(
            message=LLMMessage(
                role="assistant",
                content="I cannot schedule peanut butter toast because the profile has a severe peanut allergy constraint.",
            ),
            usage=LLMUsage(
                input_tokens=140,
                output_tokens=30,
                total_tokens=170,
                is_provider_reported=False,
            ),
            model_identifier="mock-qwen-3.8-27b:free",
            system_fingerprint=None,
            provider_request_id="openrouter_mock_req_02",
            latency_ms=12.0,
            finish_reason="stop",
        ),
        "unclear": LLMCompletionResult(
            message=LLMMessage(
                role="assistant",
                content='{"clarification_request": "Which meal slot would you like to assign this recipe to?"}',
            ),
            usage=LLMUsage(
                input_tokens=120,
                output_tokens=25,
                total_tokens=145,
                is_provider_reported=False,
            ),
            model_identifier="mock-qwen-3.8-27b:free",
            system_fingerprint=None,
            provider_request_id="openrouter_mock_req_03",
            latency_ms=10.0,
            finish_reason="stop",
        ),
    }

    default_resp = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            content='{"status": "openrouter_dry_run_simulated_completion"}',
        ),
        usage=LLMUsage(
            input_tokens=105,
            output_tokens=20,
            total_tokens=125,
            is_provider_reported=False,
        ),
        model_identifier="mock-qwen-3.8-27b:free",
        system_fingerprint=None,
        provider_request_id="openrouter_mock_req_default",
        latency_ms=8.0,
        finish_reason="stop",
    )

    return MockLLMProvider(canned_responses=canned, default_response=default_resp)


# =============================================================================
# 6. DRY RUN RUNNER (ZERO NETWORK CALLS)
# =============================================================================

def run_openrouter_pilot_dry_run(
    catalog_path: Union[str, Path] = "data/catalog_seed.json",
    fixtures_path: Union[str, Path] = "data/plan_fixtures.json",
    tasks_path: Union[str, Path] = "data/tasks.jsonl",
    output_dir: Optional[Union[str, Path]] = None,
    provider: Optional[LLMProvider] = None,
    rate_limiter: Optional[SequentialRateLimiter] = None,
    max_requests: int = MAX_TOTAL_PROVIDER_REQUESTS,
) -> Dict[str, Any]:
    """Execute simulated 9-run pilot offline with deterministic mock data."""
    ingredients, catalog = load_catalog_from_json(catalog_path)
    fixtures = load_plan_fixtures(fixtures_path, catalog, ingredients)
    all_tasks = load_benchmark_tasks(tasks_path, fixtures, catalog)

    task_map = {t.task_id: t for t in all_tasks}
    pilot_tasks = [task_map[tid] for tid in PILOT_TASK_IDS]

    pricing = create_openrouter_free_pricing()

    underlying_provider = provider or create_openrouter_pilot_mock_provider()
    guardrail = OpenRouterPilotRequestGuardrailProvider(
        provider=underlying_provider,
        max_requests=max_requests,
        rate_limiter=rate_limiter,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"openrouter_dry_run_{timestamp}"

    runner = ExperimentRunner(
        fixtures=fixtures,
        tasks=pilot_tasks,
        catalog=catalog,
        ingredients=ingredients,
        pricing=pricing,
        default_provider=guardrail,
        experiment_run_id=run_id,
        experiment_config_version="phase6.5-openrouter-pilot-dry-run",
        provider_mode="simulated",
        model_name="mock-qwen-3.8-27b:free",
        condition_c_max_model_requests=MAX_REQUESTS_PER_CONDITION_C,
        condition_c_max_inspections=MAX_INSPECTIONS_PER_CONDITION_C,
    )

    records: List[RunRecord] = []
    incomplete_reason: Optional[str] = None
    for task in pilot_tasks:
        for condition in PILOT_CONDITIONS:
            if guardrail.total_requests >= max_requests:
                incomplete_reason = f"Request ceiling of {max_requests} reached before starting {task.task_id} condition {condition}."
                break
            try:
                rec = runner.run_task(task=task, condition_id=condition, provider=guardrail)
                records.append(rec)
            except RuntimeError as exc:
                if "safety ceiling reached" in str(exc).lower():
                    incomplete_reason = str(exc)
                    break
                raise
        if incomplete_reason is not None:
            break

    if output_dir is None:
        target_dir = Path("artifacts") / "phase6_5_openrouter_pilot" / run_id
    else:
        target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    is_complete = len(records) == MAX_PILOT_RUNS and incomplete_reason is None

    # Export artifacts
    runner.export_artifacts(records, target_dir)
    _patch_pilot_analysis_markdown(
        target_dir / "analysis.md",
        run_id=run_id,
        is_live=False,
        guardrail=guardrail,
        max_requests=max_requests,
        incomplete_reason=incomplete_reason,
    )

    return {
        "status": "completed" if is_complete else "incomplete",
        "incomplete_reason": incomplete_reason,
        "run_id": run_id,
        "runs_completed": len(records),
        "runs_planned": MAX_PILOT_RUNS,
        "total_provider_requests": guardrail.total_requests,
        "max_theoretical_requests": MAX_THEORETICAL_TOTAL_REQUESTS,
        "max_requests_per_condition_c": MAX_REQUESTS_PER_CONDITION_C,
        "max_inspections_per_condition_c": MAX_INSPECTIONS_PER_CONDITION_C,
        "max_requests_ceiling": max_requests,
        "global_request_ceiling": MAX_TOTAL_PROVIDER_REQUESTS,
        "ceiling_reached": not is_complete and (guardrail.total_requests >= max_requests or "ceiling" in (incomplete_reason or "").lower()),
        "network_calls_made": 0,
        "output_directory": str(target_dir),
        "artifacts": {
            "raw_runs": str(target_dir / "raw_runs.jsonl"),
            "comparison_csv": str(target_dir / "comparison.csv"),
            "analysis_markdown": str(target_dir / "analysis.md"),
        },
    }


# =============================================================================
# 7. LIVE PILOT RUNNER (REQUIRES DUAL CONFIRMATION)
# =============================================================================

def run_live_openrouter_pilot(
    api_key: Optional[str] = None,
    catalog_path: Union[str, Path] = "data/catalog_seed.json",
    fixtures_path: Union[str, Path] = "data/plan_fixtures.json",
    tasks_path: Union[str, Path] = "data/tasks.jsonl",
    output_dir: Optional[Union[str, Path]] = None,
    live: bool = False,
    confirm_live: bool = False,
    max_requests: int = MAX_TOTAL_PROVIDER_REQUESTS,
    min_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS,
    client: Optional[Any] = None,
    clock: Optional[Callable[[], float]] = None,
    sleeper: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Execute live OpenRouter pilot with strict dual-flag confirmation and request ceilings."""
    if not (live and confirm_live):
        raise RuntimeError(
            "Live OpenRouter pilot execution requires explicit dual confirmation: "
            "both --live and --confirm-live flags must be supplied. "
            "Cleanly aborting to prevent unintended API execution."
        )

    # 1. Preflight validation
    preflight = run_openrouter_pilot_preflight(
        catalog_path=catalog_path,
        fixtures_path=fixtures_path,
        tasks_path=tasks_path,
        api_key=api_key,
    )
    if preflight["status"] != "passed":
        raise RuntimeError(f"Preflight validation failed: {preflight['errors']}")
    if not preflight["api_key_configured"]:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not configured.")

    ingredients, catalog = load_catalog_from_json(catalog_path)
    fixtures = load_plan_fixtures(fixtures_path, catalog, ingredients)
    all_tasks = load_benchmark_tasks(tasks_path, fixtures, catalog)

    task_map = {t.task_id: t for t in all_tasks}
    pilot_tasks = [task_map[tid] for tid in PILOT_TASK_IDS]

    pricing = create_openrouter_free_pricing()

    base_provider = OpenRouterProvider(
        api_key=api_key,
        model=PILOT_MODEL,
        timeout_seconds=60.0,
        client=client,
        max_output_tokens=DEFAULT_OUTPUT_TOKEN_CEILING,
        max_retries=0,
    )

    rate_limiter = SequentialRateLimiter(
        min_interval_seconds=min_interval_seconds,
        clock=clock,
        sleeper=sleeper,
    )

    guardrail = OpenRouterPilotRequestGuardrailProvider(
        provider=base_provider,
        max_requests=max_requests,
        rate_limiter=rate_limiter,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"openrouter_live_pilot_{timestamp}"

    runner = ExperimentRunner(
        fixtures=fixtures,
        tasks=pilot_tasks,
        catalog=catalog,
        ingredients=ingredients,
        pricing=pricing,
        default_provider=guardrail,
        experiment_run_id=run_id,
        experiment_config_version="phase6.5-openrouter-live-pilot",
        provider_mode="live",
        model_name=PILOT_MODEL,
        condition_c_max_model_requests=MAX_REQUESTS_PER_CONDITION_C,
        condition_c_max_inspections=MAX_INSPECTIONS_PER_CONDITION_C,
    )

    records: List[RunRecord] = []
    incomplete_reason: Optional[str] = None
    for task in pilot_tasks:
        for condition in PILOT_CONDITIONS:
            if guardrail.total_requests >= max_requests:
                incomplete_reason = f"Request ceiling of {max_requests} reached before starting {task.task_id} condition {condition}."
                break
            try:
                rec = runner.run_task(task=task, condition_id=condition, provider=guardrail)
                records.append(rec)
            except RuntimeError as exc:
                if "safety ceiling reached" in str(exc).lower():
                    incomplete_reason = str(exc)
                    break
                raise
        if incomplete_reason is not None:
            break

    if output_dir is None:
        target_dir = Path("artifacts") / "phase6_5_openrouter_pilot" / run_id
    else:
        target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    is_complete = len(records) == MAX_PILOT_RUNS and incomplete_reason is None

    # Export artifacts
    runner.export_artifacts(records, target_dir)
    _patch_pilot_analysis_markdown(
        target_dir / "analysis.md",
        run_id=run_id,
        is_live=True,
        guardrail=guardrail,
        max_requests=max_requests,
        incomplete_reason=incomplete_reason,
    )

    return {
        "status": "completed" if is_complete else "incomplete",
        "incomplete_reason": incomplete_reason,
        "run_id": run_id,
        "runs_completed": len(records),
        "runs_planned": MAX_PILOT_RUNS,
        "total_provider_requests": guardrail.total_requests,
        "max_theoretical_requests": MAX_THEORETICAL_TOTAL_REQUESTS,
        "max_requests_per_condition_c": MAX_REQUESTS_PER_CONDITION_C,
        "max_inspections_per_condition_c": MAX_INSPECTIONS_PER_CONDITION_C,
        "max_requests_ceiling": max_requests,
        "global_request_ceiling": MAX_TOTAL_PROVIDER_REQUESTS,
        "ceiling_reached": not is_complete and (guardrail.total_requests >= max_requests or "ceiling" in (incomplete_reason or "").lower()),
        "cumulative_tokens": {
            "input": guardrail.cumulative_input_tokens,
            "output": guardrail.cumulative_output_tokens,
            "cached": guardrail.cumulative_cached_input_tokens,
            "reasoning": guardrail.cumulative_reasoning_tokens,
            "total": guardrail.cumulative_total_tokens,
            "derived_total": guardrail.cumulative_derived_total_tokens,
        },
        "output_directory": str(target_dir),
        "artifacts": {
            "raw_runs": str(target_dir / "raw_runs.jsonl"),
            "comparison_csv": str(target_dir / "comparison.csv"),
            "analysis_markdown": str(target_dir / "analysis.md"),
        },
    }


# =============================================================================
# 8. STANDALONE ONE-REQUEST CONNECTION TEST
# =============================================================================

def run_openrouter_connection_test(
    api_key: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    connection_test: bool = False,
    confirm_connection_test: bool = False,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a single-request live connectivity test against OpenRouter API."""
    if not (connection_test and confirm_connection_test):
        raise RuntimeError(
            "Standalone OpenRouter connection test requires explicit dual confirmation: "
            "both --connection-test and --confirm-connection-test flags must be supplied. "
            "Neither flag alone may make a network request."
        )

    configured_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    key_configured = bool(configured_key and configured_key.strip())

    ts_utc = datetime.now(timezone.utc).isoformat()
    test_run_id = f"openrouter_conn_test_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    artifact_dir = Path(output_dir) if output_dir else Path(DEFAULT_CONNECTION_TEST_ARTIFACT_DIR) / test_run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifact_dir / "connection_test_result.json"

    if not key_configured:
        payload = {
            "status": "failed",
            "success": False,
            "error": "OPENROUTER_API_KEY environment variable is not configured.",
            "error_type": "MissingAPIKeyError",
            "model_requested": PILOT_MODEL,
            "actual_model_identifier": None,
            "provider_request_id": None,
            "system_fingerprint": None,
            "latency_ms": 0.0,
            "timestamp_utc": ts_utc,
            "prompt_sent": CONNECTION_TEST_PROMPT,
            "response_text": None,
            "token_usage": None,
            "usage_provenance": None,
            "billing_tier": "free",
            "artifact_file": str(result_path),
            "requests_made": 0,
            "max_requests_allowed": 1,
        }
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    base_provider = OpenRouterProvider(
        api_key=api_key,
        model=PILOT_MODEL,
        timeout_seconds=30.0,
        client=client,
        max_output_tokens=CONNECTION_TEST_OUTPUT_TOKEN_CEILING,
        max_retries=0,
    )
    guardrail = OpenRouterPilotRequestGuardrailProvider(
        provider=base_provider,
        max_requests=1,
    )

    t_start = time.perf_counter()
    try:
        completion: LLMCompletionResult = guardrail.complete(
            messages=[LLMMessage(role="user", content=CONNECTION_TEST_PROMPT)],
            tools=None,
            tool_choice=None,
        )
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        response_text = (completion.message.content or "").strip()

        contains_expected_token = EXPECTED_CONNECTION_TEST_TOKEN in response_text
        if contains_expected_token:
            status = "passed"
            success = True
            error = None
            error_type = None
        else:
            status = "failed"
            success = False
            error = (
                f"Response validation failed: expected confirmation token '{EXPECTED_CONNECTION_TEST_TOKEN}' "
                f"was not found in response text: {response_text[:200]!r}"
            )
            error_type = "ResponseValidationError"

        usage = completion.usage
        token_usage_dict = {
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
            "derived_total_tokens": usage.derived_total_tokens if usage else None,
            "cached_input_tokens": usage.cached_input_tokens if usage else None,
            "reasoning_tokens": usage.reasoning_tokens if usage else None,
            "is_provider_reported": usage.is_provider_reported if usage else False,
        }
        provenance = "provider_reported" if (usage and usage.is_provider_reported) else "simulated"

        payload = {
            "status": status,
            "success": success,
            "model_requested": PILOT_MODEL,
            "actual_model_identifier": completion.model_identifier,
            "provider_request_id": completion.provider_request_id,
            "system_fingerprint": completion.system_fingerprint,
            "latency_ms": latency_ms,
            "timestamp_utc": ts_utc,
            "prompt_sent": CONNECTION_TEST_PROMPT,
            "response_text": response_text,
            "token_usage": token_usage_dict,
            "usage_provenance": provenance,
            "billing_tier": "free",
            "artifact_file": str(result_path),
            "requests_made": guardrail.total_requests,
            "max_requests_allowed": 1,
            "error": error,
            "error_type": error_type,
        }

    except Exception as exc:
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        exc_type_name = type(exc).__name__
        exc_msg = str(exc)
        for secret in filter(None, [api_key, os.environ.get("OPENROUTER_API_KEY")]):
            exc_msg = exc_msg.replace(secret, "[REDACTED_API_KEY]")

        payload = {
            "status": "failed",
            "success": False,
            "error": exc_msg,
            "error_type": exc_type_name,
            "model_requested": PILOT_MODEL,
            "actual_model_identifier": None,
            "provider_request_id": None,
            "system_fingerprint": None,
            "latency_ms": latency_ms,
            "timestamp_utc": ts_utc,
            "prompt_sent": CONNECTION_TEST_PROMPT,
            "response_text": None,
            "token_usage": None,
            "usage_provenance": None,
            "billing_tier": "free",
            "artifact_file": str(result_path),
            "requests_made": guardrail.total_requests,
            "max_requests_allowed": 1,
        }

    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# =============================================================================
# 9. ARTIFACT WRITERS
# =============================================================================

def _patch_pilot_analysis_markdown(
    analysis_path: Path,
    run_id: str,
    is_live: bool,
    guardrail: OpenRouterPilotRequestGuardrailProvider,
    max_requests: int = MAX_TOTAL_PROVIDER_REQUESTS,
    incomplete_reason: Optional[str] = None,
) -> None:
    existing = analysis_path.read_text(encoding="utf-8") if analysis_path.exists() else ""
    runs_completed = len(guardrail.request_history)
    banner = f"""# {PILOT_TAG}
> [!IMPORTANT]
> **LIMITED SAMPLE SIZE NOTICE**: This report represents a controlled preliminary pilot
> consisting of 3 tasks across Conditions A, B, and C (9 runs).
> It is **NOT** the full 25-task benchmark.

- **Pilot Run ID:** `{run_id}`
- **Execution Mode:** `{'LIVE OPENROUTER API' if is_live else 'OFFLINE SIMULATED TEST DOUBLE'}`
- **Model Identifier:** `{PILOT_MODEL}`
- **Total Provider Requests Used:** `{guardrail.total_requests}`
- **Request Ceiling:** `{max_requests}`
"""
    if incomplete_reason:
        banner += f"\n> [!WARNING]\n> **INCOMPLETE PILOT**: {incomplete_reason}\n\n"
    banner += "\n---\n\n"
    analysis_path.write_text(banner + existing, encoding="utf-8")


# =============================================================================
# 10. CLI ENTRYPOINT
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ShopConnect OpenRouter Pilot Execution Harness")
    parser.add_argument("--preflight", action="store_true", help="Run offline preflight validation")
    parser.add_argument("--dry-run", action="store_true", help="Run offline simulated dry-run (9 runs)")
    parser.add_argument("--live", action="store_true", help="Enable live OpenRouter API pilot")
    parser.add_argument("--confirm-live", action="store_true", help="Confirm live OpenRouter API pilot")
    parser.add_argument("--connection-test", action="store_true", help="Enable one-request connection test")
    parser.add_argument("--confirm-connection-test", action="store_true", help="Confirm one-request connection test")
    parser.add_argument("--catalog", default="data/catalog_seed.json")
    parser.add_argument("--fixtures", default="data/plan_fixtures.json")
    parser.add_argument("--tasks", default="data/tasks.jsonl")
    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args(argv)

    if args.connection_test or args.confirm_connection_test:
        if not (args.connection_test and args.confirm_connection_test):
            print("ERROR: Connection test requires both --connection-test AND --confirm-connection-test.")
            return 1
        res = run_openrouter_connection_test(output_dir=args.output_dir, connection_test=True, confirm_connection_test=True)
        print(json.dumps(res, indent=2))
        return 0 if res.get("success") else 1

    if args.preflight:
        pf = run_openrouter_pilot_preflight(args.catalog, args.fixtures, args.tasks)
        print(json.dumps(pf, indent=2))
        return 0 if pf["status"] == "passed" else 1

    if args.dry_run:
        res = run_openrouter_pilot_dry_run(args.catalog, args.fixtures, args.tasks, args.output_dir)
        print(json.dumps(res, indent=2))
        return 0 if res["status"] == "completed" else 1

    if args.live or args.confirm_live:
        if not (args.live and args.confirm_live):
            print("ERROR: Live pilot requires both --live AND --confirm-live.")
            return 1
        res = run_live_openrouter_pilot(
            catalog_path=args.catalog,
            fixtures_path=args.fixtures,
            tasks_path=args.tasks,
            output_dir=args.output_dir,
            live=True,
            confirm_live=True,
        )
        print(json.dumps(res, indent=2))
        return 0 if res["status"] == "completed" else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
