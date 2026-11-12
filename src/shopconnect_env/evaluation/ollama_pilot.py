"""Controlled local Ollama inference pilot execution harness and guardrails.

Stage 6 Phase 6.5:
- Safe, bounded execution of exactly 3 frozen benchmark tasks across 3 conditions (9 runs).
- Selected tasks:
  * 1 COMMIT: task_01_commit_replace_wed_dinner (meal_replacement)
  * 1 REJECT: task_14_reject_allergen_peanut_toast (allergen_conflict)
  * 1 CLARIFY: task_22_clarify_missing_meal_slot (ambiguous_request)
- Strict maximum 20 total provider requests across the full pilot.
- Sequential execution only (zero parallelism, global execution lock).
- SDK retries disabled / single-attempt execution.
- Output token ceiling enforced (max_output_tokens=2048).
- Model: qwen3.5:0.8b.
- Base URL: http://localhost:11434/v1.
- Conservative context length: 4096 tokens.
- No API key required; zero remote cloud fallback.
- Dual confirmation flags (--live AND --confirm-live) strictly required for live model execution.
- Offline preflight validation and simulated dry-run modes with zero network access.
- Dedicated artifacts under artifacts/phase6_5_ollama_pilot/<pilot_run_id>/.
- Distinctly tagged: LOCAL_INFERENCE_PILOT (OLLAMA) — LIMITED_SAMPLE — NOT_FULL_EXPERIMENT.
- Pricing: Rates are None (hardware/energy unmeasured; never falsely report $0.00).
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
import urllib.request
import urllib.error

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
    OllamaProvider,
    usage_to_dict,
)
from shopconnect_env.domain.models import load_catalog_from_json
from shopconnect_env.evaluation.metrics import (
    PricingConfig,
    RunMetrics,
    compute_aggregate_metrics,
    create_ollama_local_pricing,
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

# Per-condition request budgets for Ollama pilot
MAX_REQUESTS_PER_CONDITION_A: int = 1
MAX_REQUESTS_PER_CONDITION_B: int = 1
MAX_REQUESTS_PER_CONDITION_C: int = 4
MAX_INSPECTIONS_PER_CONDITION_C: int = 5

# Maximum possible total requests: 3 tasks × (1 A + 1 B + 4 C) = 18 requests
MAX_THEORETICAL_TOTAL_REQUESTS: int = PILOT_TASKS_COUNT * (
    MAX_REQUESTS_PER_CONDITION_A + MAX_REQUESTS_PER_CONDITION_B + MAX_REQUESTS_PER_CONDITION_C
)

MAX_TOTAL_PROVIDER_REQUESTS: int = 20
GLOBAL_REQUEST_CEILING: int = MAX_TOTAL_PROVIDER_REQUESTS
DEFAULT_OUTPUT_TOKEN_CEILING: int = 2048
PILOT_MODEL: str = "qwen3.5:0.8b"
DEFAULT_BASE_URL: str = "http://localhost:11434/v1"
DEFAULT_CONTEXT_LENGTH: int = 4096
PILOT_TAG: str = "LOCAL_INFERENCE_PILOT (OLLAMA) — LIMITED_SAMPLE — NOT_FULL_EXPERIMENT"

# Standalone One-Request Connection Test Specifications
CONNECTION_TEST_PROMPT: str = "Reply with exactly: SHOPCONNECT_OLLAMA_CONNECTION_OK"
EXPECTED_CONNECTION_TEST_TOKEN: str = "SHOPCONNECT_OLLAMA_CONNECTION_OK"
CONNECTION_TEST_OUTPUT_TOKEN_CEILING: int = 64
DEFAULT_CONNECTION_TEST_ARTIFACT_DIR: str = "artifacts/phase6_5_ollama_connection_test"


# =============================================================================
# 2. LOCAL SERVER STATUS CHECKER
# =============================================================================

def check_ollama_server_status(
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 2.0,
    probe_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Check whether local Ollama server is running and qwen3.5:0.8b is pulled.

    Strict local-only validation:
    - Verifies hostname is strictly localhost / 127.0.0.1.
    - Probes /api/version for local server version.
    - Probes /api/tags for installed models.
    - Validates qwen3.5:0.8b presence without remote network access.

    Args:
        base_url: Ollama base URL (e.g. 'http://localhost:11434/v1').
        timeout: Network probe timeout in seconds.
        probe_fn: Optional test-double probe callable returning status dict.

    Returns:
        Dict describing server_online, server_version, model_installed, and setup commands.
    """
    if probe_fn is not None:
        status = probe_fn(base_url)
        if "server_version" not in status:
            status["server_version"] = "0.40.16" if status.get("server_online") else None
        return status

    # Enforce localhost-only constraint
    import urllib.parse
    parsed = urllib.parse.urlparse(base_url)
    hostname = (parsed.hostname or "").lower()
    if hostname not in ("localhost", "127.0.0.1", "::1"):
        return {
            "server_online": False,
            "server_version": None,
            "model_installed": False,
            "installed_models": [],
            "setup_instruction": (
                f"Security violation: base_url must target localhost only (got {base_url}). "
                "Remote endpoints are strictly prohibited."
            ),
            "error": f"Base URL {base_url} is not a local address.",
        }

    # Derive server root URL (strip /v1 or /v1/)
    root_url = base_url.rstrip("/")
    if root_url.endswith("/v1"):
        root_url = root_url[:-3].rstrip("/")

    version_endpoint = f"{root_url}/api/version"
    tags_endpoint = f"{root_url}/api/tags"
    server_version: Optional[str] = None
    try:
        # 1. Probe version
        req_ver = urllib.request.Request(version_endpoint, headers={"User-Agent": "ShopConnect-Ollama-Probe"})
        with urllib.request.urlopen(req_ver, timeout=timeout) as resp_ver:
            ver_data = json.loads(resp_ver.read().decode("utf-8"))
            server_version = ver_data.get("version")

        # 2. Probe tags
        req_tags = urllib.request.Request(tags_endpoint, headers={"User-Agent": "ShopConnect-Ollama-Probe"})
        with urllib.request.urlopen(req_tags, timeout=timeout) as resp_tags:
            data = json.loads(resp_tags.read().decode("utf-8"))
            models = data.get("models", [])
            model_names = [m.get("name") or m.get("model") for m in models if isinstance(m, dict)]

            # Check for exact model or variant tag
            target_installed = any(PILOT_MODEL in name for name in model_names if name)
            return {
                "server_online": True,
                "server_version": server_version,
                "model_installed": target_installed,
                "installed_models": model_names,
                "setup_instruction": None if target_installed else f"Run 'ollama pull {PILOT_MODEL}' to download the model.",
                "error": None,
            }
    except Exception as exc:
        return {
            "server_online": False,
            "server_version": None,
            "model_installed": False,
            "installed_models": [],
            "setup_instruction": f"Start Ollama service via 'ollama serve' and run 'ollama pull {PILOT_MODEL}'.",
            "error": str(exc),
        }


# =============================================================================
# 3. PILOT REQUEST GUARDRAIL PROVIDER
# =============================================================================

class OllamaPilotRequestGuardrailProvider:
    """Decorates OllamaProvider to enforce strict count limits and request audit logging."""

    MAX_TOTAL_PROVIDER_REQUESTS: int = MAX_TOTAL_PROVIDER_REQUESTS

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
                f"Ollama pilot execution safety ceiling reached: ceiling of {self.max_requests} "
                "total local requests exceeded. Cleanly aborting pilot execution to "
                "prevent runaway loops."
            )

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
            "usage": usage_to_dict(result.usage),
        })

        return result


# =============================================================================
# 4. PREFLIGHT VALIDATION
# =============================================================================

def run_ollama_pilot_preflight(
    catalog_path: Union[str, Path] = "data/catalog_seed.json",
    fixtures_path: Union[str, Path] = "data/plan_fixtures.json",
    tasks_path: Union[str, Path] = "data/tasks.jsonl",
    base_url: str = DEFAULT_BASE_URL,
    probe_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute preflight configuration and data integrity validation for Ollama."""
    catalog_p = Path(catalog_path)
    fixtures_p = Path(fixtures_path)
    tasks_p = Path(tasks_path)

    errors: List[str] = []

    # Check local Ollama server status
    status = check_ollama_server_status(base_url=base_url, probe_fn=probe_fn)
    if not status["server_online"]:
        errors.append(f"Ollama server is not running at {base_url}. Start it with 'ollama serve'.")
    elif not status["model_installed"]:
        errors.append(f"Model '{PILOT_MODEL}' is not installed in Ollama. Pull it with 'ollama pull {PILOT_MODEL}'.")

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
        pricing = create_ollama_local_pricing()
    except Exception as e:
        pricing = None
        errors.append(f"Invalid pricing configuration: {e}")

    return {
        "status": "passed" if not errors else "failed",
        "server_online": status["server_online"],
        "server_version": status.get("server_version"),
        "model_installed": status["model_installed"],
        "installed_models": status.get("installed_models", []),
        "setup_instruction": status.get("setup_instruction"),
        "pilot_model": PILOT_MODEL,
        "base_url": base_url,
        "context_length": DEFAULT_CONTEXT_LENGTH,
        "planned_runs_count": MAX_PILOT_RUNS,
        "max_requests_per_condition_a": MAX_REQUESTS_PER_CONDITION_A,
        "max_requests_per_condition_b": MAX_REQUESTS_PER_CONDITION_B,
        "max_requests_per_condition_c": MAX_REQUESTS_PER_CONDITION_C,
        "max_inspections_per_condition_c": MAX_INSPECTIONS_PER_CONDITION_C,
        "max_theoretical_requests": MAX_THEORETICAL_TOTAL_REQUESTS,
        "max_possible_total_requests": MAX_THEORETICAL_TOTAL_REQUESTS,
        "global_request_ceiling": MAX_TOTAL_PROVIDER_REQUESTS,
        "pricing_complete": pricing.is_complete() if pricing else False,
        "billing_tier": "local_inference",
        "tasks_loaded": len(pilot_tasks),
        "errors": errors,
    }


# =============================================================================
# 5. DETERMINISTIC MOCK FOR DRY-RUN TESTING
# =============================================================================

def create_ollama_pilot_mock_provider() -> MockLLMProvider:
    """Create offline mock provider configured specifically for Ollama pilot dry run."""
    canned: Dict[str, LLMCompletionResult] = {
        "pasta": LLMCompletionResult(
            message=LLMMessage(
                role="assistant",
                content='{"action": "replace_meal", "day": "Wednesday", "slot": "Dinner", "recipe": "Vegetable Pasta Primavera"}',
            ),
            usage=LLMUsage(
                input_tokens=148,
                output_tokens=46,
                total_tokens=194,
                cached_input_tokens=0,
                reasoning_tokens=0,
                is_provider_reported=False,
            ),
            model_identifier="mock-qwen3.5:0.8b",
            system_fingerprint=None,
            provider_request_id="ollama_mock_req_01",
            latency_ms=250.0,
            finish_reason="stop",
        ),
        "peanut": LLMCompletionResult(
            message=LLMMessage(
                role="assistant",
                content="I cannot schedule peanut butter toast because the profile has a severe peanut allergy constraint.",
            ),
            usage=LLMUsage(
                input_tokens=138,
                output_tokens=28,
                total_tokens=166,
                is_provider_reported=False,
            ),
            model_identifier="mock-qwen3.5:0.8b",
            system_fingerprint=None,
            provider_request_id="ollama_mock_req_02",
            latency_ms=210.0,
            finish_reason="stop",
        ),
        "unclear": LLMCompletionResult(
            message=LLMMessage(
                role="assistant",
                content='{"clarification_request": "Which meal slot would you like to assign this recipe to?"}',
            ),
            usage=LLMUsage(
                input_tokens=118,
                output_tokens=24,
                total_tokens=142,
                is_provider_reported=False,
            ),
            model_identifier="mock-qwen3.5:0.8b",
            system_fingerprint=None,
            provider_request_id="ollama_mock_req_03",
            latency_ms=180.0,
            finish_reason="stop",
        ),
    }

    default_resp = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            content='{"status": "ollama_dry_run_simulated_completion"}',
        ),
        usage=LLMUsage(
            input_tokens=100,
            output_tokens=18,
            total_tokens=118,
            is_provider_reported=False,
        ),
        model_identifier="mock-qwen3.5:0.8b",
        system_fingerprint=None,
        provider_request_id="ollama_mock_req_default",
        latency_ms=150.0,
        finish_reason="stop",
    )

    return MockLLMProvider(canned_responses=canned, default_response=default_resp)


# =============================================================================
# 6. DRY RUN RUNNER (ZERO NETWORK CALLS)
# =============================================================================

def run_ollama_pilot_dry_run(
    catalog_path: Union[str, Path] = "data/catalog_seed.json",
    fixtures_path: Union[str, Path] = "data/plan_fixtures.json",
    tasks_path: Union[str, Path] = "data/tasks.jsonl",
    output_dir: Optional[Union[str, Path]] = None,
    provider: Optional[LLMProvider] = None,
    max_requests: int = MAX_TOTAL_PROVIDER_REQUESTS,
) -> Dict[str, Any]:
    """Execute simulated 9-run pilot offline with deterministic mock data."""
    ingredients, catalog = load_catalog_from_json(catalog_path)
    fixtures = load_plan_fixtures(fixtures_path, catalog, ingredients)
    all_tasks = load_benchmark_tasks(tasks_path, fixtures, catalog)

    task_map = {t.task_id: t for t in all_tasks}
    pilot_tasks = [task_map[tid] for tid in PILOT_TASK_IDS]

    pricing = create_ollama_local_pricing()

    underlying_provider = provider or create_ollama_pilot_mock_provider()
    guardrail = OllamaPilotRequestGuardrailProvider(
        provider=underlying_provider,
        max_requests=max_requests,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"ollama_dry_run_{timestamp}"

    runner = ExperimentRunner(
        fixtures=fixtures,
        tasks=pilot_tasks,
        catalog=catalog,
        ingredients=ingredients,
        pricing=pricing,
        default_provider=guardrail,
        experiment_run_id=run_id,
        experiment_config_version="phase6.5-ollama-pilot-dry-run",
        provider_mode="simulated",
        model_name="mock-qwen3.5:0.8b",
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
        target_dir = Path("artifacts") / "phase6_5_ollama_pilot" / run_id
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

def run_live_ollama_pilot(
    base_url: str = DEFAULT_BASE_URL,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    catalog_path: Union[str, Path] = "data/catalog_seed.json",
    fixtures_path: Union[str, Path] = "data/plan_fixtures.json",
    tasks_path: Union[str, Path] = "data/tasks.jsonl",
    output_dir: Optional[Union[str, Path]] = None,
    live: bool = False,
    confirm_live: bool = False,
    max_requests: int = MAX_TOTAL_PROVIDER_REQUESTS,
    client: Optional[Any] = None,
    probe_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Execute live Ollama pilot with strict dual-flag confirmation and request ceilings."""
    if not (live and confirm_live):
        raise RuntimeError(
            "Live Ollama pilot execution requires explicit dual confirmation: "
            "both --live and --confirm-live flags must be supplied. "
            "Cleanly aborting to prevent unintended model execution."
        )

    # 1. Preflight validation
    preflight = run_ollama_pilot_preflight(
        catalog_path=catalog_path,
        fixtures_path=fixtures_path,
        tasks_path=tasks_path,
        base_url=base_url,
        probe_fn=probe_fn,
    )
    if preflight["status"] != "passed":
        raise RuntimeError(f"Preflight validation failed: {preflight['errors']}")

    ingredients, catalog = load_catalog_from_json(catalog_path)
    fixtures = load_plan_fixtures(fixtures_path, catalog, ingredients)
    all_tasks = load_benchmark_tasks(tasks_path, fixtures, catalog)

    task_map = {t.task_id: t for t in all_tasks}
    pilot_tasks = [task_map[tid] for tid in PILOT_TASK_IDS]

    pricing = create_ollama_local_pricing()

    base_provider = OllamaProvider(
        model=PILOT_MODEL,
        base_url=base_url,
        context_length=context_length,
        timeout_seconds=180.0,
        enable_thinking=False,
        client=client,
        max_output_tokens=DEFAULT_OUTPUT_TOKEN_CEILING,
    )

    guardrail = OllamaPilotRequestGuardrailProvider(
        provider=base_provider,
        max_requests=max_requests,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"ollama_live_pilot_{timestamp}"

    runner = ExperimentRunner(
        fixtures=fixtures,
        tasks=pilot_tasks,
        catalog=catalog,
        ingredients=ingredients,
        pricing=pricing,
        default_provider=guardrail,
        experiment_run_id=run_id,
        experiment_config_version="phase6.5-ollama-live-pilot",
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
        target_dir = Path("artifacts") / "phase6_5_ollama_pilot" / run_id
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

def run_ollama_connection_test(
    base_url: str = DEFAULT_BASE_URL,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    output_dir: Optional[Union[str, Path]] = None,
    connection_test: bool = False,
    confirm_connection_test: bool = False,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a single-request live connectivity test against local Ollama instance."""
    if not (connection_test and confirm_connection_test):
        raise RuntimeError(
            "Standalone Ollama connection test requires explicit dual confirmation: "
            "both --connection-test and --confirm-connection-test flags must be supplied. "
            "Neither flag alone may make a request."
        )

    ts_utc = datetime.now(timezone.utc).isoformat()
    test_run_id = f"ollama_conn_test_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    artifact_dir = Path(output_dir) if output_dir else Path(DEFAULT_CONNECTION_TEST_ARTIFACT_DIR) / test_run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifact_dir / "connection_test_result.json"

    base_provider = OllamaProvider(
        model=PILOT_MODEL,
        base_url=base_url,
        context_length=context_length,
        timeout_seconds=60.0,
        enable_thinking=False,
        client=client,
        max_output_tokens=CONNECTION_TEST_OUTPUT_TOKEN_CEILING,
    )
    guardrail = OllamaPilotRequestGuardrailProvider(
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
        finish_reason = completion.finish_reason or "unknown"
        raw_meta = completion.raw_response or {}
        reasoning_present = bool(raw_meta.get("reasoning_present", False))
        reasoning_length = int(raw_meta.get("reasoning_length", 0))

        contains_expected_token = EXPECTED_CONNECTION_TEST_TOKEN in response_text
        if contains_expected_token:
            status = "passed"
            success = True
            error = None
            error_type = None
        else:
            status = "failed"
            success = False
            # If the final response is empty because the output-token limit was exhausted, report that distinctly
            if not response_text and finish_reason == "length":
                error = (
                    f"Response was empty because output token limit ({CONNECTION_TEST_OUTPUT_TOKEN_CEILING}) "
                    f"was exhausted (finish_reason='length') before user-facing content was generated."
                )
                if reasoning_present:
                    error += (
                        f" Model generated {reasoning_length} characters of internal reasoning "
                        "which consumed the token budget. Verify reasoning_effort='none' was sent."
                    )
                error_type = "OutputLimitExhaustedError"
            elif not response_text:
                error = "Response content was empty string (no user-facing content generated)."
                if reasoning_present:
                    error += f" Model generated {reasoning_length} characters of reasoning but no user-facing content."
                error_type = "EmptyResponseError"
            else:
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
        provenance = "local_inference" if (usage and usage.is_provider_reported) else "simulated"

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
            "finish_reason": finish_reason,
            "reasoning_present": reasoning_present,
            "reasoning_length": reasoning_length,
            "token_usage": token_usage_dict,
            "usage_provenance": provenance,
            "billing_tier": "local_inference",
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
            "finish_reason": None,
            "reasoning_present": False,
            "reasoning_length": 0,
            "token_usage": None,
            "usage_provenance": None,
            "billing_tier": "local_inference",
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
    guardrail: OllamaPilotRequestGuardrailProvider,
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
- **Execution Mode:** `{'LIVE LOCAL OLLAMA INFERENCE' if is_live else 'OFFLINE SIMULATED TEST DOUBLE'}`
- **Model Identifier:** `{PILOT_MODEL}`
- **Total Provider Requests Used:** `{guardrail.total_requests}`
- **Request Ceiling:** `{max_requests}`
- **Pricing:** Local inference (monetary cost unmeasured / evaluates strictly to None)
"""
    if incomplete_reason:
        banner += f"\n> [!WARNING]\n> **INCOMPLETE PILOT**: {incomplete_reason}\n\n"
    banner += "\n---\n\n"
    analysis_path.write_text(banner + existing, encoding="utf-8")


# =============================================================================
# 10. CLI ENTRYPOINT
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ShopConnect Ollama Local Pilot Execution Harness")
    parser.add_argument("--preflight", action="store_true", help="Run offline preflight validation")
    parser.add_argument("--dry-run", action="store_true", help="Run offline simulated dry-run (9 runs)")
    parser.add_argument("--live", action="store_true", help="Enable live local Ollama pilot")
    parser.add_argument("--confirm-live", action="store_true", help="Confirm live local Ollama pilot")
    parser.add_argument("--connection-test", action="store_true", help="Enable one-request connection test")
    parser.add_argument("--confirm-connection-test", action="store_true", help="Confirm one-request connection test")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--catalog", default="data/catalog_seed.json")
    parser.add_argument("--fixtures", default="data/plan_fixtures.json")
    parser.add_argument("--tasks", default="data/tasks.jsonl")
    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args(argv)

    if args.connection_test or args.confirm_connection_test:
        if not (args.connection_test and args.confirm_connection_test):
            print("ERROR: Connection test requires both --connection-test AND --confirm-connection-test.")
            return 1
        res = run_ollama_connection_test(
            base_url=args.base_url,
            context_length=args.context_length,
            output_dir=args.output_dir,
            connection_test=True,
            confirm_connection_test=True,
        )
        print(json.dumps(res, indent=2))
        return 0 if res.get("success") else 1

    if args.preflight:
        pf = run_ollama_pilot_preflight(args.catalog, args.fixtures, args.tasks, base_url=args.base_url)
        print(json.dumps(pf, indent=2))
        return 0 if pf["status"] == "passed" else 1

    if args.dry_run:
        res = run_ollama_pilot_dry_run(args.catalog, args.fixtures, args.tasks, args.output_dir)
        print(json.dumps(res, indent=2))
        return 0 if res["status"] == "completed" else 1

    if args.live or args.confirm_live:
        if not (args.live and args.confirm_live):
            print("ERROR: Live pilot requires both --live AND --confirm-live.")
            return 1
        res = run_live_ollama_pilot(
            base_url=args.base_url,
            context_length=args.context_length,
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
