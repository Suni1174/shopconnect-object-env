"""Controlled real Google Gemini API pilot execution harness and guardrails.

Stage 6 Phase 6.5:
- Safe, bounded execution of exactly 3 frozen benchmark tasks across 3 conditions (9 runs).
- Selected tasks:
  * 1 COMMIT: task_01_commit_replace_wed_dinner (meal_replacement)
  * 1 REJECT: task_14_reject_allergen_peanut_toast (allergen_conflict)
  * 1 CLARIFY: task_22_clarify_missing_meal_slot (ambiguous_request)
- Strict maximum 24 total provider requests across the full pilot.
- Sequential execution only (zero parallelism).
- SDK retries disabled / single-attempt execution.
- Output token ceiling enforced (max_output_tokens=2048).
- Model: gemini-3.8-flash with low thinking level and no temperature.
- Dual confirmation flags (--live AND --confirm-live) strictly required for live API calls.
- Offline preflight validation and simulated dry-run modes with zero network access.
- Dedicated artifacts under artifacts/phase6_5_gemini_pilot/<pilot_run_id>/.
- Distinctly tagged: REAL_API_PILOT (GEMINI) — LIMITED_SAMPLE — NOT_FULL_EXPERIMENT.
- Differentiates free tier ($0.00) from paid tier and unknown billing tier (cost evaluates to None).
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
    GeminiProvider,
    LLMCompletionResult,
    LLMMessage,
    LLMProvider,
    LLMUsage,
    MockLLMProvider,
    usage_to_dict,
)
from shopconnect_env.domain.models import load_catalog_from_json
from shopconnect_env.evaluation.metrics import (
    PricingConfig,
    RunMetrics,
    compute_aggregate_metrics,
    create_gemini_38_flash_pricing,
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

# Per-condition request budgets for Gemini pilot
MAX_REQUESTS_PER_CONDITION_A: int = 1
MAX_REQUESTS_PER_CONDITION_B: int = 1
MAX_REQUESTS_PER_CONDITION_C: int = 4  # Gemini-pilot-specific maximum of 4 model requests per Condition C task
MAX_INSPECTIONS_PER_CONDITION_C: int = 5  # Preserved maximum of 5 actual inspections, bounded by smaller request budget

# Maximum possible total requests: 3 tasks × (1 A + 1 B + 4 C) = 18 requests
MAX_THEORETICAL_TOTAL_REQUESTS: int = PILOT_TASKS_COUNT * (
    MAX_REQUESTS_PER_CONDITION_A + MAX_REQUESTS_PER_CONDITION_B + MAX_REQUESTS_PER_CONDITION_C
)

# Global hard ceiling adapted to Google AI Studio free-tier quotas (RPD=20, 2-request safety margin)
MAX_TOTAL_PROVIDER_REQUESTS: int = 20
GLOBAL_REQUEST_CEILING: int = MAX_TOTAL_PROVIDER_REQUESTS
DEFAULT_OUTPUT_TOKEN_CEILING: int = 2048
PILOT_MODEL: str = "gemini-3.8-flash"
PILOT_TAG: str = "REAL_API_PILOT (GEMINI) — LIMITED_SAMPLE — NOT_FULL_EXPERIMENT"

# Google AI Studio Free-Tier Quotas & Safe Rate Limiting Target
QUOTA_RPM_LIMIT: int = 5
QUOTA_TPM_LIMIT: int = 250_000
QUOTA_RPD_LIMIT: int = 20
CONFIGURED_SAFE_TARGET_RPM: int = 3
MIN_REQUEST_INTERVAL_SECONDS: float = 20.0  # 60s / 3 = 20.0s minimum interval between request starts


# =============================================================================
# 2. SEQUENTIAL RATE LIMITER
# =============================================================================

class SequentialRateLimiter:
    """Enforces a safe sequential interval between provider requests.

    Guarantees:
    - Maintains at least min_interval_seconds (default 20.0s for 3 RPM target)
      between consecutive request starts to respect the 5 RPM free-tier quota.
    - Uses monotonic timing via clock callable (defaults to time.monotonic).
    - Allows clock and sleeper dependency injection for fast, deterministic unit testing.
    - Zero sleep when min_interval_seconds <= 0.0 (e.g. mock-only dry runs).
    """

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
        """Wait until at least min_interval_seconds has elapsed since the last request start.

        Returns:
            The duration in seconds that was slept (0.0 if no sleep was required).
        """
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

class GeminiPilotRequestGuardrailProvider:
    """Decorates an LLMProvider to enforce strict invocation, token, and rate limits.

    Guarantees:
    - Counts every single provider request across all pilot runs.
    - Halts execution by raising RuntimeError if 20-request ceiling is reached.
    - Enforces sequential rate limiting (minimum 20s interval / 3 RPM target) when configured.
    - Aggregates provider-reported tokens, derived totals, and latencies.
    - Captures request IDs for audit trails.
    """

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
        """Delegate SDK version retrieval to wrapped provider."""
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
        """Execute request under strict count and rate limit guardrails."""
        if self.total_requests >= self.max_requests:
            raise RuntimeError(
                f"Gemini pilot execution safety ceiling reached: ceiling of {self.max_requests} "
                "total provider requests exceeded. Cleanly aborting pilot execution to "
                "prevent unexpected billing or runaways."
            )

        slept = 0.0
        if self.rate_limiter is not None:
            slept = self.rate_limiter.wait_for_slot()
            self.cumulative_sleep_seconds += slept

        t0 = time.perf_counter()
        result = self.provider.complete(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        self.total_requests += 1

        # Track identifiers
        if result.provider_request_id:
            self.provider_request_ids.append(result.provider_request_id)

        # Track usage
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

        self.request_history.append({
            "request_index": self.total_requests,
            "provider_request_id": result.provider_request_id,
            "system_fingerprint": result.system_fingerprint,
            "latency_ms": latency_ms,
            "usage": usage_to_dict(result.usage),
        })

        return result


# =============================================================================
# 3. PREFLIGHT VALIDATION (ZERO NETWORK ACTIVITY)
# =============================================================================

def run_gemini_pilot_preflight(
    catalog_path: Union[str, Path] = "data/catalog_seed.json",
    fixtures_path: Union[str, Path] = "data/plan_fixtures.json",
    tasks_path: Union[str, Path] = "data/tasks.jsonl",
    api_key: Optional[str] = None,
    billing_tier: str = "unknown",
) -> Dict[str, Any]:
    """Execute preflight configuration and data integrity validation for Gemini.

    Guarantees:
    - Exactly ZERO network calls.
    - API key string is NEVER logged, echoed, or included in return dictionary.
    - Verifies presence of catalog, fixtures, and exact 3 pilot benchmark tasks.
    - Validates Gemini 3.8 Flash pricing configuration.
    """
    catalog_p = Path(catalog_path)
    fixtures_p = Path(fixtures_path)
    tasks_p = Path(tasks_path)

    errors: List[str] = []

    # 1. API Key Configuration Check (Without Exposing Secrets)
    configured_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    key_present = bool(configured_key and configured_key.strip())

    # 2. Catalog & Ingredients Validation
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

    # 3. Fixtures Validation
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

    # 4. Tasks Validation (Strict Pilot Subset)
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

                # Verify expected outcomes match pilot matrix
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

    # 5. Pricing Configuration Check
    try:
        pricing = create_gemini_38_flash_pricing(billing_tier=billing_tier)
    except Exception as e:
        pricing = None
        errors.append(f"Invalid pricing configuration: {e}")

    return {
        "status": "passed" if not errors else "failed",
        "api_key_configured": key_present,
        "api_key_status": "present (masked)" if key_present else "missing",
        "provider": "Google",
        "catalog_recipe_count": len(catalog),
        "ingredient_count": len(ingredients),
        "fixtures_count": len(fixtures),
        "pilot_tasks_count": len(pilot_tasks),
        "pilot_task_ids": list(PILOT_TASK_IDS),
        "pilot_conditions": list(PILOT_CONDITIONS),
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
        "output_token_ceiling": DEFAULT_OUTPUT_TOKEN_CEILING,
        "model": PILOT_MODEL,
        "thinking_level": "low",
        "billing_tier": billing_tier,
        "billing_tier_is_user_assumption": True if billing_tier != "unknown" else False,
        "billing_tier_note": (
            "User-provided reporting assumption, NOT an account-billing verification. "
            "Never inferred from model name or availability."
        ),
        "rpm_limit": QUOTA_RPM_LIMIT,
        "tpm_limit": QUOTA_TPM_LIMIT,
        "rpd_limit": QUOTA_RPD_LIMIT,
        "configured_safe_target": f"{CONFIGURED_SAFE_TARGET_RPM} RPM",
        "configured_safe_target_rpm": CONFIGURED_SAFE_TARGET_RPM,
        "min_request_interval_seconds": MIN_REQUEST_INTERVAL_SECONDS,
        "max_provider_requests": MAX_TOTAL_PROVIDER_REQUESTS,
        "pricing_version": pricing.version if pricing else None,
        "pricing_complete": pricing.is_complete() if pricing else False,
        "input_rate_per_m": str(pricing.input_cost_per_million) if pricing and pricing.input_cost_per_million is not None else "unknown",
        "cached_input_rate_per_m": str(pricing.cached_input_cost_per_million) if pricing and pricing.cached_input_cost_per_million is not None else "unknown",
        "output_rate_per_m": str(pricing.output_cost_per_million) if pricing and pricing.output_cost_per_million is not None else "unknown",
        "privacy_notice": pricing.privacy_notice if pricing else None,
        "network_calls_made": 0,
        "errors": errors,
    }


# =============================================================================
# 4. SIMULATED DRY-RUN (ZERO NETWORK ACTIVITY)
# =============================================================================

def create_gemini_pilot_mock_provider() -> MockLLMProvider:
    """Create a deterministic MockLLMProvider with responses for the 3 Gemini pilot tasks."""
    canned: Dict[str, LLMCompletionResult] = {
        # Task 01: replace wed dinner with rustic tomato and spinach whole wheat pasta (4 servings)
        "Wednesday": LLMCompletionResult(
            message=LLMMessage(
                role="assistant",
                content=(
                    '[{"day": "Monday", "meal_type": "breakfast", "status": "assigned", "recipe_id": "rec_berry_oatmeal", "servings": 2}, '
                    '{"day": "Monday", "meal_type": "lunch", "status": "assigned", "recipe_id": "rec_chicken_salad_wrap", "servings": 2}, '
                    '{"day": "Monday", "meal_type": "dinner", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Tuesday", "meal_type": "breakfast", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Tuesday", "meal_type": "lunch", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Tuesday", "meal_type": "dinner", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Wednesday", "meal_type": "breakfast", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Wednesday", "meal_type": "lunch", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Wednesday", "meal_type": "dinner", "status": "assigned", "recipe_id": "rec_tomato_spinach_pasta", "servings": 4}, '
                    '{"day": "Thursday", "meal_type": "breakfast", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Thursday", "meal_type": "lunch", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Thursday", "meal_type": "dinner", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Friday", "meal_type": "breakfast", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Friday", "meal_type": "lunch", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Friday", "meal_type": "dinner", "status": "assigned", "recipe_id": "rec_baked_chicken_rice", "servings": 2}, '
                    '{"day": "Saturday", "meal_type": "breakfast", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Saturday", "meal_type": "lunch", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Saturday", "meal_type": "dinner", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Sunday", "meal_type": "breakfast", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Sunday", "meal_type": "lunch", "status": "unfilled", "recipe_id": null, "servings": 0}, '
                    '{"day": "Sunday", "meal_type": "dinner", "status": "unfilled", "recipe_id": null, "servings": 0}]'
                ),
                tool_calls=(
                    {
                        "id": "gemini_call_replace_01",
                        "call_id": "gemini_call_replace_01",
                        "type": "function",
                        "function": {
                            "name": "replace_meal",
                            "arguments": '{"day": "Wednesday", "meal_type": "dinner", "recipe_id": "rec_tomato_spinach_pasta", "servings": 4}',
                        },
                    },
                ),
            ),
            usage=LLMUsage(
                input_tokens=160,
                output_tokens=50,
                total_tokens=210,
                cached_input_tokens=40,
                reasoning_tokens=20,
                is_provider_reported=False,
            ),
            model_identifier="mock-gemini-3.8-flash",
            system_fingerprint=None,
            provider_request_id="gemini_req_dry_run_01",
            latency_ms=14.0,
            finish_reason="stop",
        ),
        # Task 14: allergen peanut toast
        "peanut": LLMCompletionResult(
            message=LLMMessage(
                role="assistant",
                content="I cannot schedule peanut butter toast because the profile has a severe peanut allergy constraint.",
            ),
            usage=LLMUsage(
                input_tokens=145,
                output_tokens=32,
                total_tokens=177,
                is_provider_reported=False,
            ),
            model_identifier="mock-gemini-3.8-flash",
            system_fingerprint=None,
            provider_request_id="gemini_req_dry_run_02",
            latency_ms=11.0,
            finish_reason="stop",
        ),
        # Task 22: clarify missing slot
        "unclear": LLMCompletionResult(
            message=LLMMessage(
                role="assistant",
                content='{"clarification_request": "Which meal slot would you like to assign this recipe to?"}',
            ),
            usage=LLMUsage(
                input_tokens=125,
                output_tokens=28,
                total_tokens=153,
                is_provider_reported=False,
            ),
            model_identifier="mock-gemini-3.8-flash",
            system_fingerprint=None,
            provider_request_id="gemini_req_dry_run_03",
            latency_ms=9.5,
            finish_reason="stop",
        ),
    }

    # Default fallback
    default_resp = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            content='{"status": "gemini_dry_run_simulated_completion"}',
        ),
        usage=LLMUsage(
            input_tokens=110,
            output_tokens=22,
            total_tokens=132,
            is_provider_reported=False,
        ),
        model_identifier="mock-gemini-3.8-flash",
        system_fingerprint=None,
        provider_request_id="gemini_req_dry_run_default",
        latency_ms=8.5,
        finish_reason="stop",
    )

    return MockLLMProvider(canned_responses=canned, default_response=default_resp)


def run_gemini_pilot_dry_run(
    catalog_path: Union[str, Path] = "data/catalog_seed.json",
    fixtures_path: Union[str, Path] = "data/plan_fixtures.json",
    tasks_path: Union[str, Path] = "data/tasks.jsonl",
    output_dir: Optional[Union[str, Path]] = None,
    provider: Optional[LLMProvider] = None,
    billing_tier: str = "unknown",
    rate_limiter: Optional[SequentialRateLimiter] = None,
    max_requests: int = MAX_TOTAL_PROVIDER_REQUESTS,
) -> Dict[str, Any]:
    """Execute complete simulated Gemini pilot evaluation without network access.

    Performs preflight checks, loads frozen artifacts, and runs all 9 pilot
    runs sequentially against a deterministic test double.
    Does not sleep by default in dry-run mode (rate_limiter is None/0.0s).
    """
    preflight = run_gemini_pilot_preflight(
        catalog_path=catalog_path,
        fixtures_path=fixtures_path,
        tasks_path=tasks_path,
        billing_tier=billing_tier,
    )
    if preflight["status"] != "passed":
        raise ValueError(f"Preflight validation failed: {preflight['errors']}")

    ingredients, catalog = load_catalog_from_json(catalog_path)
    fixtures = load_plan_fixtures(fixtures_path, catalog, ingredients)
    all_tasks = load_benchmark_tasks(tasks_path, fixtures, catalog)
    task_map = {t.task_id: t for t in all_tasks}
    pilot_tasks = [task_map[tid] for tid in PILOT_TASK_IDS]

    simulated_provider = provider or create_gemini_pilot_mock_provider()
    guardrail = GeminiPilotRequestGuardrailProvider(
        provider=simulated_provider,
        max_requests=max_requests,
        rate_limiter=rate_limiter,
    )

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"gemini_pilot_dry_run_{ts_str}"
    target_dir = Path(output_dir) if output_dir else Path(f"artifacts/phase6_5_gemini_pilot/{run_id}")
    target_dir.mkdir(parents=True, exist_ok=True)

    pricing = create_gemini_38_flash_pricing(billing_tier=billing_tier)
    runner = ExperimentRunner(
        fixtures=fixtures,
        tasks=pilot_tasks,
        catalog=catalog,
        ingredients=ingredients,
        pricing=pricing,
        default_provider=guardrail,
        experiment_run_id=run_id,
        experiment_config_version="phase6.5-gemini-pilot-dry-run",
        provider_mode="simulated",
        model_name="mock-gemini-3.8-flash",
        condition_c_max_model_requests=MAX_REQUESTS_PER_CONDITION_C,
        condition_c_max_inspections=MAX_INSPECTIONS_PER_CONDITION_C,
    )

    records: List[RunRecord] = []
    incomplete_reason: Optional[str] = None
    for task in pilot_tasks:
        for cond in PILOT_CONDITIONS:
            if guardrail.total_requests >= guardrail.max_requests:
                incomplete_reason = (
                    f"Safety ceiling of {guardrail.max_requests} total provider requests reached "
                    f"before executing task {task.task_id} on Condition {cond}."
                )
                break
            try:
                rec = runner.run_task(task=task, condition_id=cond, provider=guardrail)
                records.append(rec)
                if rec.actual_outcome == "ERROR":
                    err_msg = rec.error_detail or "Unknown execution error"
                    if "safety ceiling" in err_msg.lower() or "requests exceeded" in err_msg.lower():
                        incomplete_reason = err_msg
                    else:
                        incomplete_reason = f"Run failure on task {task.task_id} (Condition {cond}): {err_msg}"
                    break
            except RuntimeError as exc:
                if "safety ceiling reached" in str(exc) or "requests exceeded" in str(exc):
                    incomplete_reason = str(exc)
                    break
                incomplete_reason = f"RuntimeError during task {task.task_id} on Condition {cond}: {exc}"
                break
            except Exception as exc:
                incomplete_reason = f"Execution failure during task {task.task_id} on Condition {cond}: {exc}"
                break
        if incomplete_reason:
            break

    is_complete = (len(records) == MAX_PILOT_RUNS) and (incomplete_reason is None)
    if is_complete:
        pilot_status = "completed"
    elif incomplete_reason and ("ceiling" in incomplete_reason.lower() or "requests exceeded" in incomplete_reason.lower()):
        pilot_status = "incomplete_quota_ceiling_reached"
    else:
        pilot_status = "incomplete_run_failed"

    # Export artifacts
    runner.export_artifacts(records, target_dir)
    _patch_gemini_pilot_analysis_markdown(
        target_dir / "analysis.md",
        run_id=run_id,
        is_live=False,
        billing_tier=billing_tier,
        is_complete=is_complete,
        status=pilot_status,
        incomplete_reason=incomplete_reason,
        runs_completed=len(records),
        runs_planned=MAX_PILOT_RUNS,
        max_requests=max_requests,
    )

    return {
        "status": pilot_status,
        "is_complete": is_complete,
        "incomplete_reason": incomplete_reason,
        "pilot_run_id": run_id,
        "is_live": False,
        "provider": "Google",
        "model": PILOT_MODEL,
        "billing_tier": billing_tier,
        "output_directory": str(target_dir),
        "total_runs": len(records),
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
        "raw_runs_file": str(target_dir / "raw_runs.jsonl"),
        "comparison_file": str(target_dir / "comparison.csv"),
        "analysis_file": str(target_dir / "analysis.md"),
        "tag": PILOT_TAG + (" (DRY_RUN_MODE)" if is_complete else f" ({pilot_status.upper()})"),
    }


# =============================================================================
# 5. LIVE PILOT EXECUTION (DUAL CONFIRMATION REQUIRED)
# =============================================================================

def run_live_gemini_pilot(
    live: bool = False,
    confirm_live: bool = False,
    api_key: Optional[str] = None,
    billing_tier: str = "unknown",
    catalog_path: Union[str, Path] = "data/catalog_seed.json",
    fixtures_path: Union[str, Path] = "data/plan_fixtures.json",
    tasks_path: Union[str, Path] = "data/tasks.jsonl",
    output_dir: Optional[Union[str, Path]] = None,
    client: Optional[Any] = None,
    rate_limiter: Optional[SequentialRateLimiter] = None,
    clock: Optional[Callable[[], float]] = None,
    sleeper: Optional[Callable[[float], None]] = None,
    max_requests: int = MAX_TOTAL_PROVIDER_REQUESTS,
) -> Dict[str, Any]:
    """Execute the controlled live Google Gemini API pilot for 3 tasks across 3 conditions.

    Strict Invariants:
    1. Requires BOTH live=True AND confirm_live=True. Without both, raises ValueError.
    2. Exactly 3 tasks * 3 conditions = 9 total task runs executed sequentially.
    3. Strict ceiling of 20 provider requests enforced by GeminiPilotRequestGuardrailProvider.
    4. Target model: gemini-3.8-flash with low thinking level, max_output_tokens=2048, max_retries=0.
    5. Sequential rate limiter enforces >= 20 seconds between request starts (3 RPM safe target).
    6. Stops safely when 20 requests reached; records incomplete pilot distinctly.
    7. Outputs generated under artifacts/phase6_5_gemini_pilot/<run_id>/ and tagged with
       REAL_API_PILOT (GEMINI) — LIMITED_SAMPLE — NOT_FULL_EXPERIMENT.
    """
    if not (live and confirm_live):
        raise ValueError(
            "SAFETY VIOLATION: Live API pilot execution requires BOTH live=True and confirm_live=True. "
            "Without explicit dual confirmation, network requests are strictly prohibited."
        )

    # Preflight integrity check
    preflight = run_gemini_pilot_preflight(
        catalog_path=catalog_path,
        fixtures_path=fixtures_path,
        tasks_path=tasks_path,
        api_key=api_key,
        billing_tier=billing_tier,
    )
    if preflight["status"] != "passed":
        raise ValueError(f"Preflight validation failed: {preflight['errors']}")
    if not preflight["api_key_configured"] and client is None:
        raise ValueError("GEMINI_API_KEY is not configured in the environment.")

    # Load validated inputs
    ingredients, catalog = load_catalog_from_json(catalog_path)
    fixtures = load_plan_fixtures(fixtures_path, catalog, ingredients)
    all_tasks = load_benchmark_tasks(tasks_path, fixtures, catalog)
    task_map = {t.task_id: t for t in all_tasks}
    pilot_tasks = [task_map[tid] for tid in PILOT_TASK_IDS]

    # Initialize live Gemini provider with Stage 6 specifications (max_retries=0)
    base_provider = GeminiProvider(
        api_key=api_key,
        model=PILOT_MODEL,
        thinking_level="low",
        timeout_seconds=60.0,
        client=client,
        max_output_tokens=DEFAULT_OUTPUT_TOKEN_CEILING,
        max_retries=0,
    )

    # Initialize sequential rate limiter (target: 3 RPM, >= 20s between request starts)
    actual_rate_limiter = rate_limiter or SequentialRateLimiter(
        min_interval_seconds=MIN_REQUEST_INTERVAL_SECONDS,
        clock=clock,
        sleeper=sleeper,
    )

    # Wrap provider in request-guardrail with rate limiter
    guardrail = GeminiPilotRequestGuardrailProvider(
        provider=base_provider,
        max_requests=max_requests,
        rate_limiter=actual_rate_limiter,
    )

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"gemini_pilot_live_{ts_str}"
    target_dir = Path(output_dir) if output_dir else Path(f"artifacts/phase6_5_gemini_pilot/{run_id}")
    target_dir.mkdir(parents=True, exist_ok=True)

    pricing = create_gemini_38_flash_pricing(billing_tier=billing_tier)
    runner = ExperimentRunner(
        fixtures=fixtures,
        tasks=pilot_tasks,
        catalog=catalog,
        ingredients=ingredients,
        pricing=pricing,
        default_provider=guardrail,
        experiment_run_id=run_id,
        experiment_config_version="phase6.5-gemini-live-pilot",
        provider_mode="live",
        model_name=PILOT_MODEL,
        condition_c_max_model_requests=MAX_REQUESTS_PER_CONDITION_C,
        condition_c_max_inspections=MAX_INSPECTIONS_PER_CONDITION_C,
    )

    # Execute runs sequentially, stopping safely if ceiling is reached
    records: List[RunRecord] = []
    incomplete_reason: Optional[str] = None
    for task in pilot_tasks:
        for cond in PILOT_CONDITIONS:
            if guardrail.total_requests >= guardrail.max_requests:
                incomplete_reason = (
                    f"Safety ceiling of {guardrail.max_requests} total provider requests reached "
                    f"before executing task {task.task_id} on Condition {cond}."
                )
                break
            try:
                rec = runner.run_task(task=task, condition_id=cond, provider=guardrail)
                records.append(rec)
                if rec.actual_outcome == "ERROR":
                    err_msg = rec.error_detail or "Unknown execution error"
                    if "safety ceiling" in err_msg.lower() or "requests exceeded" in err_msg.lower():
                        incomplete_reason = err_msg
                    else:
                        incomplete_reason = f"Run failure on task {task.task_id} (Condition {cond}): {err_msg}"
                    break
            except RuntimeError as exc:
                if "safety ceiling reached" in str(exc) or "requests exceeded" in str(exc):
                    incomplete_reason = str(exc)
                    break
                incomplete_reason = f"RuntimeError during task {task.task_id} on Condition {cond}: {exc}"
                break
            except Exception as exc:
                incomplete_reason = f"Execution failure during task {task.task_id} on Condition {cond}: {exc}"
                break
        if incomplete_reason:
            break

    is_complete = (len(records) == MAX_PILOT_RUNS) and (incomplete_reason is None)
    if is_complete:
        pilot_status = "completed"
    elif incomplete_reason and ("ceiling" in incomplete_reason.lower() or "requests exceeded" in incomplete_reason.lower()):
        pilot_status = "incomplete_quota_ceiling_reached"
    else:
        pilot_status = "incomplete_run_failed"

    # Export artifacts
    runner.export_artifacts(records, target_dir)
    _patch_gemini_pilot_analysis_markdown(
        target_dir / "analysis.md",
        run_id=run_id,
        is_live=True,
        billing_tier=billing_tier,
        is_complete=is_complete,
        status=pilot_status,
        incomplete_reason=incomplete_reason,
        runs_completed=len(records),
        runs_planned=MAX_PILOT_RUNS,
        max_requests=max_requests,
    )

    return {
        "status": pilot_status,
        "is_complete": is_complete,
        "incomplete_reason": incomplete_reason,
        "pilot_run_id": run_id,
        "is_live": True,
        "provider": "Google",
        "model": PILOT_MODEL,
        "billing_tier": billing_tier,
        "output_directory": str(target_dir),
        "total_runs": len(records),
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
            "input_tokens": guardrail.cumulative_input_tokens,
            "output_tokens": guardrail.cumulative_output_tokens,
            "total_tokens": guardrail.cumulative_total_tokens,
            "derived_total_tokens": guardrail.cumulative_derived_total_tokens,
            "cached_input_tokens": guardrail.cumulative_cached_input_tokens,
            "reasoning_tokens": guardrail.cumulative_reasoning_tokens,
            "sleep_seconds": guardrail.cumulative_sleep_seconds,
        },
        "raw_runs_file": str(target_dir / "raw_runs.jsonl"),
        "comparison_file": str(target_dir / "comparison.csv"),
        "analysis_file": str(target_dir / "analysis.md"),
        "tag": PILOT_TAG + ("" if is_complete else " (INCOMPLETE_CEILING_REACHED)"),
    }


def _patch_gemini_pilot_analysis_markdown(
    analysis_path: Path,
    run_id: str,
    is_live: bool,
    billing_tier: str = "unknown",
    is_complete: bool = True,
    status: str = "completed",
    incomplete_reason: Optional[str] = None,
    runs_completed: int = MAX_PILOT_RUNS,
    runs_planned: int = MAX_PILOT_RUNS,
    max_requests: int = MAX_TOTAL_PROVIDER_REQUESTS,
) -> None:
    """Inject Gemini pilot-specific header banner and privacy notice into analysis.md."""
    if not analysis_path.exists():
        return

    existing_content = analysis_path.read_text(encoding="utf-8")
    if billing_tier == "free_tier":
        tier_disclaimer = (
            "USER REPORTING ASSUMPTION (UNVERIFIED): --billing-tier free_tier was specified as a "
            "reporting assumption, NOT an account-billing verification. The system never infers "
            "that an account is on the free tier from model name or availability. Google terms note "
            "that free tier submitted content may be used for product improvement. Benchmark uses "
            "synthetic data only; no personal or confidential information included."
        )
    elif billing_tier == "paid_tier":
        tier_disclaimer = (
            "USER REPORTING ASSUMPTION: --billing-tier paid_tier specified as a reporting assumption. "
            "Standard commercial API privacy terms apply."
        )
    else:
        tier_disclaimer = (
            "UNKNOWN BILLING TIER (DEFAULT): Account billing tier is unverified and unspecified. "
            "To prevent false zero-cost reporting, all monetary costs evaluate strictly to None."
        )

    completion_badge = ""
    if not is_complete:
        completion_badge = f"""
> [!WARNING]
> **INCOMPLETE PILOT EXECUTION NOTICE**:
> Execution was stopped early because {incomplete_reason or 'one or more runs failed'}.
> - **Runs Completed:** {runs_completed} of {runs_planned} planned task-condition runs.
> - **Status:** **INCOMPLETE** (`{status}`) — This run must **NOT** be reported as a completed nine-run pilot.
> - **Termination Reason:** {incomplete_reason}
"""

    banner = f"""# {PILOT_TAG}
{completion_badge}
> [!IMPORTANT]
> **LIMITED SAMPLE SIZE NOTICE**: This report represents a controlled preliminary pilot
> consisting of {runs_completed} task-condition runs (across Conditions A, B, and C).
> It is **NOT** the full 25-task empirical benchmark. Findings should be interpreted as
> pilot verification of provider integration, format parsing, and error behavior.

> [!NOTE]
> **PRIVACY & PROVENANCE NOTICE**:
> - Provider: **Google**
> - Model: `{PILOT_MODEL}` (`thinking_level=low`, `max_output_tokens={DEFAULT_OUTPUT_TOKEN_CEILING}`)
> - Billing Tier: `{billing_tier}` (Default: `unknown`)
> - Billing Tier Assumption: {tier_disclaimer}
> - Rate Limits: Quota: {QUOTA_RPM_LIMIT} RPM / {QUOTA_TPM_LIMIT:,} TPM / {QUOTA_RPD_LIMIT} RPD | Configured Safe Target: {CONFIGURED_SAFE_TARGET_RPM} RPM (at least {MIN_REQUEST_INTERVAL_SECONDS}s between request starts)

- **Pilot Run ID:** `{run_id}`
- **Execution Mode:** `{'LIVE GOOGLE GEMINI API' if is_live else 'OFFLINE SIMULATED TEST DOUBLE'}`
- **Pilot Status:** `{'COMPLETED' if is_complete else f'INCOMPLETE — {status.upper()}'}`
- **Runs Completed:** `{runs_completed} / {runs_planned}`
- **Planned Request Budget:** `{MAX_THEORETICAL_TOTAL_REQUESTS} requests max ({len(PILOT_TASK_IDS)} tasks × [{MAX_REQUESTS_PER_CONDITION_A} A + {MAX_REQUESTS_PER_CONDITION_B} B + {MAX_REQUESTS_PER_CONDITION_C} C])`
- **Total Provider Requests Ceiling:** `{max_requests} (safety margin: {max_requests - MAX_THEORETICAL_TOTAL_REQUESTS} requests)`
- **Benchmark Subset:**
  1. `task_01_commit_replace_wed_dinner` (Category: `meal_replacement`, Expected: `COMMIT`)
  2. `task_14_reject_allergen_peanut_toast` (Category: `allergen_conflict`, Expected: `REJECT`)
  3. `task_22_clarify_missing_meal_slot` (Category: `ambiguous_request`, Expected: `CLARIFY`)

---

"""
    new_content = banner + existing_content
    analysis_path.write_text(new_content, encoding="utf-8")


# =============================================================================
# 6. COMMAND LINE ENTRYPOINT
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the Gemini pilot runner."""
    parser = argparse.ArgumentParser(
        description="ShopConnect Stage 6 Phase 6.5 Controlled Gemini Pilot Runner",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run offline preflight validation of environment, files, and pricing (0 network calls).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run offline simulated execution of 9 pilot runs using test doubles (0 network calls).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Request live Google Gemini API pilot execution. Requires --confirm-live.",
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Explicit confirmation flag required alongside --live for live network execution.",
    )
    parser.add_argument(
        "--billing-tier",
        choices=["unknown", "free_tier", "paid_tier"],
        default="unknown",
        help=(
            "Gemini account billing tier reporting assumption (default: unknown; monetary cost evaluates to None). "
            "Specifying 'free_tier' is a user-provided reporting assumption, NOT an account-billing verification. "
            "Never infer that an account is on the free tier from the model name or availability."
        ),
    )
    parser.add_argument(
        "--catalog",
        default="data/catalog_seed.json",
        help="Path to catalog JSON file (default: data/catalog_seed.json).",
    )
    parser.add_argument(
        "--fixtures",
        default="data/plan_fixtures.json",
        help="Path to plan fixtures JSON file (default: data/plan_fixtures.json).",
    )
    parser.add_argument(
        "--tasks",
        default="data/tasks.jsonl",
        help="Path to tasks JSONL file (default: data/tasks.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output directory for artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entrypoint enforcing safety confirmation."""
    args = parse_args(argv)

    # 1. Dual-confirmation safety gate
    if args.live:
        if not args.confirm_live:
            print(
                "ERROR: Live execution requires dual confirmation.\n"
                "Please specify BOTH --live and --confirm-live to authorize live API calls.\n"
                "Execution aborted safely.",
                file=sys.stderr,
            )
            return 1

        print("Executing controlled Live API Pilot on gemini-3.8-flash...")
        try:
            res = run_live_gemini_pilot(
                live=True,
                confirm_live=True,
                billing_tier=args.billing_tier,
                catalog_path=args.catalog,
                fixtures_path=args.fixtures,
                tasks_path=args.tasks,
                output_dir=args.output_dir,
            )
            if not res.get("is_complete", True):
                print(
                    f"NOTICE: Live pilot halted early at request ceiling ({res['runs_completed']}/{res['runs_planned']} runs completed). "
                    f"Status: {res['status']}. Incomplete pilot recorded distinctly.",
                    file=sys.stderr,
                )
            else:
                print(f"Live Gemini pilot completed successfully. Artifacts: {res['output_directory']}")
            return 0
        except Exception as e:
            print(f"Live Gemini pilot failed: {e}", file=sys.stderr)
            return 2

    # 2. Simulated dry-run
    if args.dry_run:
        print("Executing simulated dry-run Gemini pilot (0 network calls)...")
        res = run_gemini_pilot_dry_run(
            catalog_path=args.catalog,
            fixtures_path=args.fixtures,
            tasks_path=args.tasks,
            output_dir=args.output_dir,
            billing_tier=args.billing_tier,
        )
        if not res.get("is_complete", True):
            print(
                f"NOTICE: Dry-run halted early at request ceiling ({res['runs_completed']}/{res['runs_planned']} runs completed). "
                f"Status: {res['status']}. Incomplete pilot recorded distinctly.",
                file=sys.stderr,
            )
        else:
            print(f"Dry run completed successfully. Artifacts: {res['output_directory']}")
        return 0

    # 3. Default or explicit preflight
    print("Executing Gemini pilot preflight validation (0 network calls)...")
    res = run_gemini_pilot_preflight(
        catalog_path=args.catalog,
        fixtures_path=args.fixtures,
        tasks_path=args.tasks,
        billing_tier=args.billing_tier,
    )
    print(json.dumps(res, indent=2))
    return 0 if res["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
