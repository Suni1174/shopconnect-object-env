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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

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
MAX_REQUESTS_PER_CONDITION_C: int = 6
MAX_TOTAL_PROVIDER_REQUESTS: int = 24  # Strict pilot ceiling: 3 (Cond A) + 3 (Cond B) + 18 (Cond C max)
DEFAULT_OUTPUT_TOKEN_CEILING: int = 2048
PILOT_MODEL: str = "gemini-3.8-flash"
PILOT_TAG: str = "REAL_API_PILOT (GEMINI) — LIMITED_SAMPLE — NOT_FULL_EXPERIMENT"


# =============================================================================
# 2. PILOT REQUEST GUARDRAIL PROVIDER
# =============================================================================

class GeminiPilotRequestGuardrailProvider:
    """Decorates an LLMProvider to enforce strict invocation and token limits.

    Guarantees:
    - Counts every single provider request across all pilot runs.
    - Strictly halts execution by raising RuntimeError if request ceiling is reached.
    - Aggregates provider-reported tokens (prompt, candidate, cached, reasoning) and latencies.
    - Captures request IDs for audit trails.
    """

    def __init__(
        self,
        provider: LLMProvider,
        max_requests: int = MAX_TOTAL_PROVIDER_REQUESTS,
    ) -> None:
        self.provider = provider
        self.max_requests = max_requests
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
        """Execute request under strict count and limit guardrails."""
        if self.total_requests >= self.max_requests:
            raise RuntimeError(
                f"Gemini pilot execution safety ceiling reached: ceiling of {self.max_requests} "
                "total provider requests exceeded. Cleanly aborting pilot execution to "
                "prevent unexpected billing or runaways."
            )

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
        "max_provider_requests_ceiling": MAX_TOTAL_PROVIDER_REQUESTS,
        "output_token_ceiling": DEFAULT_OUTPUT_TOKEN_CEILING,
        "model": PILOT_MODEL,
        "thinking_level": "low",
        "billing_tier": billing_tier,
        "billing_tier_is_user_assumption": True if billing_tier != "unknown" else False,
        "billing_tier_note": (
            "User-provided reporting assumption, NOT an account-billing verification. "
            "Never inferred from model name or availability."
        ),
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
) -> Dict[str, Any]:
    """Execute complete simulated Gemini pilot evaluation without network access.

    Performs preflight checks, loads frozen artifacts, and runs all 9 pilot
    runs sequentially against a deterministic test double.
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
        max_requests=MAX_TOTAL_PROVIDER_REQUESTS,
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
    )

    records: List[RunRecord] = []
    for task in pilot_tasks:
        for cond in PILOT_CONDITIONS:
            rec = runner.run_task(task=task, condition_id=cond, provider=guardrail)
            records.append(rec)

    # Export artifacts
    runner.export_artifacts(records, target_dir)
    _patch_gemini_pilot_analysis_markdown(
        target_dir / "analysis.md",
        run_id=run_id,
        is_live=False,
        billing_tier=billing_tier,
    )

    return {
        "status": "completed",
        "pilot_run_id": run_id,
        "is_live": False,
        "provider": "Google",
        "model": PILOT_MODEL,
        "billing_tier": billing_tier,
        "output_directory": str(target_dir),
        "total_runs": len(records),
        "total_provider_requests": guardrail.total_requests,
        "max_requests_ceiling": MAX_TOTAL_PROVIDER_REQUESTS,
        "raw_runs_file": str(target_dir / "raw_runs.jsonl"),
        "comparison_file": str(target_dir / "comparison.csv"),
        "analysis_file": str(target_dir / "analysis.md"),
        "tag": PILOT_TAG + " (DRY_RUN_MODE)",
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
) -> Dict[str, Any]:
    """Execute the controlled live Google Gemini API pilot for 3 tasks across 3 conditions.

    Strict Invariants:
    1. Requires BOTH live=True AND confirm_live=True. Without both, raises ValueError.
    2. Exactly 3 tasks * 3 conditions = 9 total task runs executed sequentially.
    3. Strict ceiling of 24 provider requests enforced by GeminiPilotRequestGuardrailProvider.
    4. Target model: gemini-3.8-flash with low thinking level, max_output_tokens=2048.
    5. Outputs generated under artifacts/phase6_5_gemini_pilot/<run_id>/ and tagged with
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

    # Initialize live Gemini provider with Stage 6 specifications
    base_provider = GeminiProvider(
        api_key=api_key,
        model=PILOT_MODEL,
        thinking_level="low",
        timeout_seconds=60.0,
        client=client,
        max_output_tokens=DEFAULT_OUTPUT_TOKEN_CEILING,
    )

    # Wrap provider in request-guardrail
    guardrail = GeminiPilotRequestGuardrailProvider(
        provider=base_provider,
        max_requests=MAX_TOTAL_PROVIDER_REQUESTS,
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
    )

    # Execute all 9 runs sequentially
    records: List[RunRecord] = []
    for task in pilot_tasks:
        for cond in PILOT_CONDITIONS:
            # Check ceiling before launching run
            if guardrail.total_requests >= MAX_TOTAL_PROVIDER_REQUESTS:
                raise RuntimeError(
                    f"Ceiling of {MAX_TOTAL_PROVIDER_REQUESTS} requests reached before executing "
                    f"task {task.task_id} on Condition {cond}."
                )
            rec = runner.run_task(task=task, condition_id=cond, provider=guardrail)
            records.append(rec)

    # Export artifacts
    runner.export_artifacts(records, target_dir)
    _patch_gemini_pilot_analysis_markdown(
        target_dir / "analysis.md",
        run_id=run_id,
        is_live=True,
        billing_tier=billing_tier,
    )

    return {
        "status": "completed",
        "pilot_run_id": run_id,
        "is_live": True,
        "provider": "Google",
        "model": PILOT_MODEL,
        "billing_tier": billing_tier,
        "output_directory": str(target_dir),
        "total_runs": len(records),
        "total_provider_requests": guardrail.total_requests,
        "max_requests_ceiling": MAX_TOTAL_PROVIDER_REQUESTS,
        "cumulative_tokens": {
            "input_tokens": guardrail.cumulative_input_tokens,
            "output_tokens": guardrail.cumulative_output_tokens,
            "total_tokens": guardrail.cumulative_total_tokens,
            "derived_total_tokens": guardrail.cumulative_derived_total_tokens,
            "cached_input_tokens": guardrail.cumulative_cached_input_tokens,
            "reasoning_tokens": guardrail.cumulative_reasoning_tokens,
        },
        "raw_runs_file": str(target_dir / "raw_runs.jsonl"),
        "comparison_file": str(target_dir / "comparison.csv"),
        "analysis_file": str(target_dir / "analysis.md"),
        "tag": PILOT_TAG,
    }


def _patch_gemini_pilot_analysis_markdown(
    analysis_path: Path,
    run_id: str,
    is_live: bool,
    billing_tier: str = "unknown",
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

    banner = f"""# {PILOT_TAG}

> [!IMPORTANT]
> **LIMITED SAMPLE SIZE NOTICE**: This report represents a controlled preliminary pilot
> consisting of 9 task-condition runs (3 representative tasks across Conditions A, B, and C).
> It is **NOT** the full 25-task empirical benchmark. Findings should be interpreted as
> pilot verification of provider integration, format parsing, and error behavior.

> [!NOTE]
> **PRIVACY & PROVENANCE NOTICE**:
> - Provider: **Google**
> - Model: `{PILOT_MODEL}` (`thinking_level=low`, `max_output_tokens={DEFAULT_OUTPUT_TOKEN_CEILING}`)
> - Billing Tier: `{billing_tier}` (Default: `unknown`)
> - Billing Tier Assumption: {tier_disclaimer}

- **Pilot Run ID:** `{run_id}`
- **Execution Mode:** `{'LIVE GOOGLE GEMINI API' if is_live else 'OFFLINE SIMULATED TEST DOUBLE'}`
- **Total Provider Requests Ceiling:** `{MAX_TOTAL_PROVIDER_REQUESTS}`
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
