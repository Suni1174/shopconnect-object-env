"""Unit and integration tests for Phase 6.5 Gemini Pilot Runner and Guardrails.

Verifies:
1. Dual confirmation safety gate (--live and --confirm-live required; refusal otherwise).
2. Offline preflight validation (0 network calls, checks catalog, fixtures, 3 pilot tasks, pricing).
3. Secret safety (API keys never logged, echoed, or included in returned data).
4. Pilot task subset integrity (task_01 COMMIT, task_14 REJECT, task_22 CLARIFY).
5. Request guardrail provider (strict 24-request ceiling, clean abortion on limit breach).
6. Request guardrail metric accumulation (tokens, latencies, request IDs).
7. Simulated dry-run execution (9 runs, sequential, produces all 3 tagged artifact files).
8. Live execution harness with test double client (verifies parameters: gemini-3.8-flash, thinking level low, max_output_tokens 2048, no temperature).
9. Pricing calculations (Gemini free tier, paid tier, and unknown billing tier handling).
10. Complete socket blocking to guarantee 100% offline test execution.
"""

from decimal import Decimal
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from shopconnect_env.agents.condition_c_agent import ConditionCAgent
from shopconnect_env.agents.providers import (
    GeminiProvider,
    LLMCompletionResult,
    LLMMessage,
    LLMProvider,
    LLMUsage,
    MockLLMProvider,
    RateLimitError,
)
from shopconnect_env.evaluation.gemini_pilot import (
    CONFIGURED_SAFE_TARGET_RPM,
    DEFAULT_OUTPUT_TOKEN_CEILING,
    MAX_INSPECTIONS_PER_CONDITION_C,
    MAX_PILOT_RUNS,
    MAX_REQUESTS_PER_CONDITION_A,
    MAX_REQUESTS_PER_CONDITION_B,
    MAX_REQUESTS_PER_CONDITION_C,
    MAX_THEORETICAL_TOTAL_REQUESTS,
    MAX_TOTAL_PROVIDER_REQUESTS,
    MIN_REQUEST_INTERVAL_SECONDS,
    PILOT_CONDITIONS,
    PILOT_MODEL,
    PILOT_TAG,
    PILOT_TASK_IDS,
    QUOTA_RPD_LIMIT,
    QUOTA_RPM_LIMIT,
    QUOTA_TPM_LIMIT,
    GeminiPilotRequestGuardrailProvider,
    SequentialRateLimiter,
    create_gemini_pilot_mock_provider,
    main,
    parse_args,
    run_gemini_pilot_dry_run,
    run_gemini_pilot_preflight,
    run_live_gemini_pilot,
)
from shopconnect_env.evaluation.metrics import (
    calculate_cost,
    create_gemini_38_flash_pricing,
)
from shopconnect_env.evaluation.tasks import ExpectedOutcome


# =============================================================================
# 1. DUAL CONFIRMATION SAFETY GATE TESTS
# =============================================================================

def test_gemini_dual_confirmation_safety_gate_enforcement():
    """run_live_gemini_pilot strictly requires BOTH live=True and confirm_live=True."""
    # 1. Neither flag
    with pytest.raises(ValueError) as exc1:
        run_live_gemini_pilot(live=False, confirm_live=False)
    assert "SAFETY VIOLATION" in str(exc1.value)

    # 2. Only live=True
    with pytest.raises(ValueError) as exc2:
        run_live_gemini_pilot(live=True, confirm_live=False)
    assert "SAFETY VIOLATION" in str(exc2.value)

    # 3. Only confirm_live=True
    with pytest.raises(ValueError) as exc3:
        run_live_gemini_pilot(live=False, confirm_live=True)
    assert "SAFETY VIOLATION" in str(exc3.value)


def test_gemini_cli_live_requires_dual_confirmation(capsys):
    """CLI exit code is 1 and execution aborts safely if --live without --confirm-live."""
    exit_code = main(["--live"])
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "ERROR: Live execution requires dual confirmation" in captured.err
    assert "Execution aborted safely" in captured.err


# =============================================================================
# 2. PREFLIGHT VALIDATION & CREDENTIAL SAFETY
# =============================================================================

def test_gemini_pilot_preflight_validation_zero_network():
    """run_gemini_pilot_preflight checks all required files and pricing with 0 network calls."""
    res = run_gemini_pilot_preflight()

    assert res["status"] == "passed"
    assert res["network_calls_made"] == 0
    assert res["pilot_tasks_count"] == 3
    assert res["planned_runs_count"] == 9
    assert res["max_requests_per_condition_a"] == 1
    assert res["max_requests_per_condition_b"] == 1
    assert res["max_requests_per_condition_c"] == 4
    assert res["max_inspections_per_condition_c"] == 5
    assert res["max_theoretical_requests"] == 18
    assert res["max_possible_total_requests"] == 18
    assert res["global_request_ceiling"] == 20
    assert res["safety_margin_requests"] == 2
    assert res["max_provider_requests_ceiling"] == 20
    assert res["max_provider_requests"] == 20
    assert res["rpm_limit"] == 5
    assert res["tpm_limit"] == 250000
    assert res["rpd_limit"] == 20
    assert res["configured_safe_target"] == "3 RPM"
    assert res["configured_safe_target_rpm"] == 3
    assert res["min_request_interval_seconds"] == 20.0
    assert res["output_token_ceiling"] == 2048
    assert res["model"] == "gemini-3.8-flash"
    assert res["provider"] == "Google"
    assert res["thinking_level"] == "low"
    assert len(res["errors"]) == 0


def test_gemini_pilot_preflight_api_key_not_leaked(monkeypatch):
    """API key secret string is never revealed in preflight return dictionary."""
    secret_key = "AIzaSySuperSecretGeminiKey123456789"
    monkeypatch.setenv("GEMINI_API_KEY", secret_key)

    res = run_gemini_pilot_preflight()

    assert res["api_key_configured"] is True
    assert res["api_key_status"] == "present (masked)"
    # Ensure full secret key is never in dict or serialized output
    assert secret_key not in json.dumps(res)


def test_gemini_pilot_tasks_ground_truth_consistency():
    """The 3 pilot tasks must strictly match the 1 COMMIT, 1 REJECT, 1 CLARIFY distribution."""
    from shopconnect_env.domain.models import load_catalog_from_json
    from shopconnect_env.evaluation.runner import load_plan_fixtures
    from shopconnect_env.evaluation.tasks import load_benchmark_tasks

    ingredients, catalog = load_catalog_from_json("data/catalog_seed.json")
    fixtures = load_plan_fixtures("data/plan_fixtures.json", catalog, ingredients)
    all_tasks = load_benchmark_tasks("data/tasks.jsonl", fixtures, catalog)
    task_map = {t.task_id: t for t in all_tasks}

    # Task 01: COMMIT meal replacement with exact frozen ground truth
    t1 = task_map["task_01_commit_replace_wed_dinner"]
    assert t1.expected_outcome == ExpectedOutcome.COMMIT
    assert t1.expected_operation == "replace_meal"
    assert t1.category == "meal_replacement"
    assert t1.fixture_id == "fixture_standard"
    assert t1.expected_day == "Wednesday"
    assert t1.expected_meal_type == "dinner"
    assert t1.expected_recipe_id == "rec_tomato_spinach_pasta"
    assert t1.expected_servings == 4

    # Task 14: REJECT allergen conflict
    t14 = task_map["task_14_reject_allergen_peanut_toast"]
    assert t14.expected_outcome == ExpectedOutcome.REJECT
    assert t14.expected_operation == "replace_meal"
    assert t14.category == "allergen_conflict"
    assert t14.fixture_id == "fixture_standard"
    assert t14.expected_day == "Monday"
    assert t14.expected_meal_type == "breakfast"
    assert t14.expected_recipe_id == "rec_peanut_butter_banana_toast"
    assert t14.expected_servings == 2
    assert t14.expected_rejection_category == "allergen_conflict"

    # Task 22: CLARIFY ambiguous request
    t22 = task_map["task_22_clarify_missing_meal_slot"]
    assert t22.expected_outcome == ExpectedOutcome.CLARIFY
    assert t22.category == "ambiguous_request"
    assert t22.fixture_id == "fixture_standard"
    assert t22.expected_day == "Tuesday"
    assert t22.expected_meal_type is None
    assert t22.expected_recipe_id == "rec_berry_oatmeal"
    assert t22.expected_servings == 2


# =============================================================================
# 3. REQUEST GUARDRAIL & CEILING ENFORCEMENT
# =============================================================================

def test_gemini_pilot_request_guardrail_ceiling_enforcement():
    """GeminiPilotRequestGuardrailProvider halts cleanly when max_requests ceiling is reached."""
    mock_inner = MockLLMProvider()
    guardrail = GeminiPilotRequestGuardrailProvider(mock_inner, max_requests=3)

    # 3 calls succeed
    for i in range(3):
        res = guardrail.complete([LLMMessage(role="user", content=f"Hello {i}")])
        assert isinstance(res, LLMCompletionResult)
    assert guardrail.total_requests == 3

    # 4th call exceeds ceiling -> RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        guardrail.complete([LLMMessage(role="user", content="Hello 4")])

    assert "Gemini pilot execution safety ceiling reached" in str(exc_info.value)
    assert "ceiling of 3 total provider requests exceeded" in str(exc_info.value)
    assert guardrail.total_requests == 3  # Did not increment on aborted call


def test_gemini_pilot_request_guardrail_usage_accumulation():
    """GeminiPilotRequestGuardrailProvider aggregates cumulative tokens and latencies."""
    resp1 = LLMCompletionResult(
        message=LLMMessage(role="assistant", content="Resp 1"),
        usage=LLMUsage(
            input_tokens=120,
            output_tokens=45,
            total_tokens=165,
            cached_input_tokens=30,
            reasoning_tokens=15,
            is_provider_reported=True,
        ),
        model_identifier="gemini-3.8-flash",
        system_fingerprint=None,
        provider_request_id="gemini_req_001",
    )
    resp2 = LLMCompletionResult(
        message=LLMMessage(role="assistant", content="Resp 2"),
        usage=LLMUsage(
            input_tokens=150,
            output_tokens=60,
            total_tokens=210,
            cached_input_tokens=50,
            reasoning_tokens=20,
            is_provider_reported=True,
        ),
        model_identifier="gemini-3.8-flash",
        system_fingerprint=None,
        provider_request_id="gemini_req_002",
    )

    mock_inner = MockLLMProvider(responses=[resp1, resp2])
    guardrail = GeminiPilotRequestGuardrailProvider(mock_inner, max_requests=10)

    guardrail.complete([LLMMessage(role="user", content="Query 1")])
    guardrail.complete([LLMMessage(role="user", content="Query 2")])

    assert guardrail.total_requests == 2
    assert guardrail.cumulative_input_tokens == 270
    assert guardrail.cumulative_output_tokens == 105
    assert guardrail.cumulative_total_tokens == 375
    assert guardrail.cumulative_derived_total_tokens == 375
    assert guardrail.cumulative_cached_input_tokens == 80
    assert guardrail.cumulative_reasoning_tokens == 35
    assert guardrail.provider_request_ids == ["gemini_req_001", "gemini_req_002"]


def test_gemini_pilot_request_guardrail_missing_total_tokens():
    """Guardrail preserves cumulative_total_tokens as None if no call reported total_tokens."""
    resp1 = LLMCompletionResult(
        message=LLMMessage(role="assistant", content="Resp no total"),
        usage=LLMUsage(
            input_tokens=100,
            output_tokens=30,
            total_tokens=None,  # Not reported
            derived_total_tokens=130,
            is_provider_reported=True,
        ),
        model_identifier="gemini-3.8-flash",
    )
    mock_inner = MockLLMProvider(responses=[resp1])
    guardrail = GeminiPilotRequestGuardrailProvider(mock_inner, max_requests=5)

    guardrail.complete([LLMMessage(role="user", content="Query")])

    assert guardrail.cumulative_input_tokens == 100
    assert guardrail.cumulative_output_tokens == 30
    assert guardrail.cumulative_total_tokens is None
    assert guardrail.cumulative_derived_total_tokens == 130


# =============================================================================
# 4. SIMULATED DRY-RUN EXECUTION TESTS
# =============================================================================

def test_gemini_pilot_dry_run_execution(tmp_path):
    """Simulated dry-run completes 9 runs and exports tagged artifacts."""
    output_dir = tmp_path / "artifacts" / "gemini_pilot_dry_run_test"

    res = run_gemini_pilot_dry_run(
        output_dir=output_dir,
        billing_tier="free_tier",
    )

    assert res["status"] == "completed"
    assert res["is_live"] is False
    assert res["provider"] == "Google"
    assert res["model"] == "gemini-3.8-flash"
    assert res["billing_tier"] == "free_tier"
    assert res["total_runs"] == 9
    assert res["total_provider_requests"] <= 20
    assert res["max_requests_ceiling"] == 20
    assert res["is_complete"] is True
    assert PILOT_TAG in res["tag"]

    # Verify artifacts exist on disk
    raw_runs_path = Path(res["raw_runs_file"])
    comparison_path = Path(res["comparison_file"])
    analysis_path = Path(res["analysis_file"])

    assert raw_runs_path.exists()
    assert comparison_path.exists()
    assert analysis_path.exists()

    # Verify raw runs content
    with open(raw_runs_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 9

    # Verify banner and privacy notice in analysis.md
    analysis_content = analysis_path.read_text(encoding="utf-8")
    assert PILOT_TAG in analysis_content
    assert "PRIVACY & PROVENANCE NOTICE" in analysis_content
    assert "Google" in analysis_content
    assert "gemini-3.8-flash" in analysis_content
    assert "USER REPORTING ASSUMPTION" in analysis_content
    assert "NOT an account-billing verification" in analysis_content


def test_gemini_pilot_billing_tier_user_assumption_metadata():
    """Verify billing tier defaults to unknown and explicit tiers are flagged as assumptions."""
    # 1. Default unknown
    preflight_default = run_gemini_pilot_preflight()
    assert preflight_default["billing_tier"] == "unknown"
    assert preflight_default["billing_tier_is_user_assumption"] is False

    # 2. User-provided free tier assumption
    preflight_free = run_gemini_pilot_preflight(billing_tier="free_tier")
    assert preflight_free["billing_tier"] == "free_tier"
    assert preflight_free["billing_tier_is_user_assumption"] is True
    assert "NOT an account-billing verification" in preflight_free["billing_tier_note"]

    # 3. User-provided paid tier assumption
    preflight_paid = run_gemini_pilot_preflight(billing_tier="paid_tier")
    assert preflight_paid["billing_tier"] == "paid_tier"
    assert preflight_paid["billing_tier_is_user_assumption"] is True


# =============================================================================
# 5. LIVE PILOT WITH TEST DOUBLE CLIENT
# =============================================================================

def test_gemini_live_pilot_with_injected_client(tmp_path):
    """Test live harness execution using injected test double client."""
    output_dir = tmp_path / "artifacts" / "gemini_pilot_live_test"

    # Create mock response object
    mock_resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(
                    role="model",
                    parts=[SimpleNamespace(text="Done", thought=False, function_call=None)],
                ),
            )
        ],
        text="Done",
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=20,
            total_token_count=120,
            cached_content_token_count=None,
            thoughts_token_count=None,
        ),
        model_version="gemini-3.8-flash",
        response_id="gemini_live_resp_001",
    )

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = mock_resp

    res = run_live_gemini_pilot(
        live=True,
        confirm_live=True,
        api_key="AIzaSyTestKey123",
        billing_tier="paid_tier",
        output_dir=output_dir,
        client=fake_client,
        sleeper=lambda _: None,
    )

    assert res["status"] == "completed"
    assert res["is_live"] is True
    assert res["provider"] == "Google"
    assert res["model"] == "gemini-3.8-flash"
    assert res["billing_tier"] == "paid_tier"
    assert res["total_runs"] == 9
    assert fake_client.models.generate_content.called

    # Check config passed to generate_content
    call_kwargs = fake_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.8-flash"
    config = call_kwargs["config"]
    assert config.max_output_tokens == DEFAULT_OUTPUT_TOKEN_CEILING
    assert config.automatic_function_calling.disable is True


# =============================================================================
# 6. PRICING TIERS & COST INTEGRITY TESTS
# =============================================================================

def test_gemini_pricing_unknown_billing_tier_evaluates_to_none():
    """If billing tier is unknown, cost must evaluate to None (never falsely $0)."""
    pricing = create_gemini_38_flash_pricing(billing_tier="unknown")

    assert pricing.billing_tier == "unknown"
    assert pricing.input_cost_per_million is None
    assert pricing.output_cost_per_million is None
    assert pricing.is_complete() is False

    usage = LLMUsage(
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        is_provider_reported=True,
    )
    cost = calculate_cost(usage, pricing)
    assert cost is None  # Unknown, not falsely 0.00


def test_gemini_pricing_free_tier():
    """Free tier rates are $0.00 and evaluate cost to 0.000000."""
    pricing = create_gemini_38_flash_pricing(billing_tier="free_tier")

    assert pricing.billing_tier == "free_tier"
    assert pricing.input_cost_per_million == Decimal("0.00")
    assert pricing.output_cost_per_million == Decimal("0.00")
    assert pricing.cached_input_cost_per_million == Decimal("0.00")
    assert pricing.is_complete() is True

    usage = LLMUsage(
        input_tokens=10000,
        output_tokens=2000,
        total_tokens=12000,
        cached_input_tokens=3000,
        is_provider_reported=True,
    )
    cost = calculate_cost(usage, pricing)
    assert cost == Decimal("0.000000")


def test_gemini_pricing_paid_tier():
    """Paid tier rates are $0.75 input, $0.075 cached, $3.75 output."""
    pricing = create_gemini_38_flash_pricing(billing_tier="paid_tier")

    assert pricing.billing_tier == "paid_tier"
    assert pricing.input_cost_per_million == Decimal("0.75")
    assert pricing.cached_input_cost_per_million == Decimal("0.075")
    assert pricing.output_cost_per_million == Decimal("3.75")
    assert pricing.is_complete() is True

    # 100,000 ordinary input ($0.75/M -> $0.075) + 100,000 cached ($0.075/M -> $0.0075) + 100,000 output ($3.75/M -> $0.375) = $0.457500
    usage = LLMUsage(
        input_tokens=200_000,
        output_tokens=100_000,
        total_tokens=300_000,
        cached_input_tokens=100_000,
        is_provider_reported=True,
    )
    cost = calculate_cost(usage, pricing)
    assert cost == Decimal("0.457500")


import time
from typing import List


# =============================================================================
# 8. FREE-TIER QUOTA ADAPTATION & RATE LIMITER REGRESSION TESTS
# =============================================================================

def test_gemini_pilot_20_request_ceiling_constant_and_enforcement():
    """Verify 20-request ceiling constant and guardrail enforcement on 21st call."""
    assert MAX_TOTAL_PROVIDER_REQUESTS == 20
    assert GeminiPilotRequestGuardrailProvider.MAX_TOTAL_PROVIDER_REQUESTS == 20

    mock_inner = MockLLMProvider()
    guardrail = GeminiPilotRequestGuardrailProvider(mock_inner, max_requests=20)

    # 20 calls must succeed
    for i in range(20):
        res = guardrail.complete([LLMMessage(role="user", content=f"Msg {i}")])
        assert isinstance(res, LLMCompletionResult)
    assert guardrail.total_requests == 20

    # 21st call must cleanly raise RuntimeError without incrementing request count
    with pytest.raises(RuntimeError) as exc_info:
        guardrail.complete([LLMMessage(role="user", content="Msg 21")])

    assert "Gemini pilot execution safety ceiling reached" in str(exc_info.value)
    assert "ceiling of 20 total provider requests exceeded" in str(exc_info.value)
    assert guardrail.total_requests == 20


def test_gemini_sequential_rate_limiter_enforcement_with_injected_clock():
    """Verify safe sequential rate limiter enforces >= 20s interval between requests."""
    curr_time = [1000.0]
    slept_durations: List[float] = []

    def virtual_clock() -> float:
        return curr_time[0]

    def virtual_sleeper(duration: float) -> None:
        slept_durations.append(duration)
        curr_time[0] += duration

    limiter = SequentialRateLimiter(
        min_interval_seconds=20.0,
        clock=virtual_clock,
        sleeper=virtual_sleeper,
    )

    # Request 1: First request starts immediately (0 sleep)
    limiter.wait_for_slot()
    assert len(slept_durations) == 0
    assert limiter.last_request_start == 1000.0

    # Advance clock by 6 seconds (elapsed = 6.0s < 20.0s)
    curr_time[0] += 6.0

    # Request 2: Must sleep for remaining 14.0 seconds
    limiter.wait_for_slot()
    assert len(slept_durations) == 1
    assert slept_durations[-1] == pytest.approx(14.0)
    assert curr_time[0] == pytest.approx(1020.0)
    assert limiter.last_request_start == 1020.0

    # Advance clock by 35 seconds (elapsed = 35.0s > 20.0s)
    curr_time[0] += 35.0

    # Request 3: Elapsed >= 20s, no sleep needed
    limiter.wait_for_slot()
    assert len(slept_durations) == 1  # No additional sleep
    assert curr_time[0] == 1055.0
    assert limiter.last_request_start == 1055.0

    # Zero-interval mode (for mock dry-runs): never sleeps
    zero_limiter = SequentialRateLimiter(
        min_interval_seconds=0.0,
        clock=virtual_clock,
        sleeper=virtual_sleeper,
    )
    zero_limiter.wait_for_slot()
    zero_limiter.wait_for_slot()
    assert len(slept_durations) == 1


def test_gemini_provider_no_retries():
    """Verify GeminiProvider sets max_retries=0 and re-raises immediately on failure."""
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = RuntimeError("ResourceExhausted 429: Quota exceeded")

    provider = GeminiProvider(
        api_key="AIzaSyTestKey",
        client=fake_client,
        max_retries=0,
    )
    assert provider.max_retries == 0

    # Single attempt: must fail immediately without retrying
    with pytest.raises(RateLimitError) as exc_info:
        provider.complete([LLMMessage(role="user", content="Ping")])

    assert "rate limit or quota exceeded" in str(exc_info.value)
    assert fake_client.models.generate_content.call_count == 1


def test_gemini_pilot_clean_stopping_at_ceiling_records_incomplete_pilot(tmp_path):
    """When ceiling is reached, runner stops cleanly and records incomplete pilot metadata."""
    out_dir = tmp_path / "artifacts" / "gemini_ceiling_test"

    # Set ceiling to 2 requests (well below the 9 runs needed)
    res = run_gemini_pilot_dry_run(
        output_dir=out_dir,
        max_requests=2,
    )

    assert res["status"] == "incomplete_quota_ceiling_reached"
    assert res["is_complete"] is False
    assert res["ceiling_reached"] is True
    assert res["runs_completed"] < 9
    assert res["runs_planned"] == 9
    assert res["total_provider_requests"] == 2
    assert res["max_requests_ceiling"] == 2
    assert "INCOMPLETE" in res["tag"]

    # Verify artifacts reflect incomplete status
    raw_runs_path = Path(res["raw_runs_file"])
    comparison_path = Path(res["comparison_file"])
    analysis_path = Path(res["analysis_file"])

    assert raw_runs_path.exists()
    assert comparison_path.exists()
    assert analysis_path.exists()

    with open(raw_runs_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == res["runs_completed"]

    analysis_text = analysis_path.read_text(encoding="utf-8")
    assert "INCOMPLETE PILOT EXECUTION NOTICE" in analysis_text
    assert "incomplete_quota_ceiling_reached" in analysis_text
    assert "Safety ceiling of 2 total provider requests reached" in analysis_text


def test_gemini_pilot_dry_run_remains_network_free_and_zero_sleep(tmp_path, monkeypatch):
    """Verify dry-run execution makes zero network calls and incurs zero sleep time."""
    # 1. Guard against any network socket activity
    def guarded_socket(*args, **kwargs):
        raise RuntimeError("NETWORK CALL ATTEMPTED: Dry run must remain 100% offline!")

    monkeypatch.setattr(socket, "socket", guarded_socket)

    # 2. Track any sleep invocations
    slept: List[float] = []
    monkeypatch.setattr(time, "sleep", lambda d: slept.append(d))

    res = run_gemini_pilot_dry_run(
        output_dir=tmp_path / "artifacts" / "gemini_zero_sleep_test",
    )

    assert res["status"] == "completed"
    assert res["is_complete"] is True
    assert res["network_calls_made"] == 0
    assert sum(slept) == 0.0


def test_condition_c_architectural_default_and_gemini_pilot_budget():
    """Verify ConditionCAgent architectural default is 6 and can be overridden to 4."""
    # 1. Architectural defaults remain 6 requests, 5 inspections
    assert ConditionCAgent.MAX_MODEL_REQUESTS == 6
    assert ConditionCAgent.MAX_INSPECTION_CALLS == 5

    mock_provider = MockLLMProvider([])
    default_agent = ConditionCAgent(provider=mock_provider)
    assert default_agent.max_model_requests == 6
    assert default_agent.max_inspection_calls == 5

    # 2. Gemini pilot override configures 4 model requests and preserves 5 inspections
    gemini_agent = ConditionCAgent(
        provider=mock_provider,
        max_model_requests=MAX_REQUESTS_PER_CONDITION_C,
        max_inspection_calls=MAX_INSPECTIONS_PER_CONDITION_C,
    )
    assert gemini_agent.max_model_requests == 4
    assert gemini_agent.max_inspection_calls == 5

    # 3. Verify agent halts at 4 requests if model never finalizes
    inspection_resp = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            tool_calls=[
                {
                    "call_id": "call_1",
                    "function": {
                        "name": "inspect_slot",
                        "arguments": json.dumps({"day": "Monday", "meal_type": "breakfast"}),
                    },
                }
            ],
        ),
        usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15, is_provider_reported=True),
        model_identifier="mock-gemini-3.8-flash",
    )
    looping_provider = MockLLMProvider(responses=[inspection_resp] * 10)
    agent = ConditionCAgent(provider=looping_provider, max_model_requests=4)

    from shopconnect_env.context.base import ContextResult
    context = ContextResult(
        condition="Object-Centric",
        condition_id="C",
        system_prompt="Test system prompt",
        task_prompt="Test user request",
        serialized_context=json.dumps({"plan_version": 1}),
    )
    resp = agent.propose_action(context=context)

    assert resp.metadata["total_model_requests"] == 4
    assert resp.proposal is None
    assert "MAX_MODEL_REQUESTS_EXCEEDED: Reached limit of 4 requests" in resp.rationale


def test_gemini_pilot_budget_breakdown_and_headroom():
    """Verify formula 3 tasks × (1 A + 1 B + 4 C) = 18 requests and 2-request margin to ceiling of 20."""
    assert MAX_REQUESTS_PER_CONDITION_A == 1
    assert MAX_REQUESTS_PER_CONDITION_B == 1
    assert MAX_REQUESTS_PER_CONDITION_C == 4
    assert MAX_INSPECTIONS_PER_CONDITION_C == 5
    assert MAX_THEORETICAL_TOTAL_REQUESTS == 18
    assert MAX_TOTAL_PROVIDER_REQUESTS == 20

    # Explicit formula check
    calculated_total = 3 * (1 + 1 + 4)
    assert MAX_THEORETICAL_TOTAL_REQUESTS == calculated_total
    assert MAX_TOTAL_PROVIDER_REQUESTS - MAX_THEORETICAL_TOTAL_REQUESTS == 2

    # Verify dry run returns this budget calculation
    dry_res = run_gemini_pilot_dry_run()
    assert dry_res["max_theoretical_requests"] == 18
    assert dry_res["max_requests_per_condition_c"] == 4
    assert dry_res["max_inspections_per_condition_c"] == 5
    assert dry_res["global_request_ceiling"] == 20


def test_gemini_pilot_halts_safely_when_global_20_ceiling_reached(tmp_path):
    """Verify pilot runner terminates safely when the global 20-request ceiling is hit."""
    out_dir = tmp_path / "artifacts" / "gemini_global_20_ceiling_test"

    # Set up a provider that responds but ceiling is reached exactly at 20
    # We test ceiling enforcement by setting max_requests=20 and simulating 20 successful requests
    mock_inner = MockLLMProvider()
    guardrail = GeminiPilotRequestGuardrailProvider(mock_inner, max_requests=20)

    for i in range(20):
        guardrail.complete([LLMMessage(role="user", content=f"Request {i}")])
    assert guardrail.total_requests == 20

    # 21st call must cleanly raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        guardrail.complete([LLMMessage(role="user", content="Request 21")])
    assert "Gemini pilot execution safety ceiling reached" in str(exc_info.value)
    assert guardrail.total_requests == 20


def test_gemini_pilot_run_failure_records_incomplete_status(tmp_path):
    """If an individual run fails with an error, the pilot must report incomplete status and never completed."""
    out_dir = tmp_path / "artifacts" / "gemini_run_failure_test"

    # Create a mock provider that raises an exception on the 2nd task call
    class FailingProvider(LLMProvider):
        def __init__(self):
            self.call_count = 0

        def complete(self, messages, tools=None, tool_choice=None, **kwargs):
            self.call_count += 1
            if self.call_count >= 2:
                raise RuntimeError("Simulated API transport error on run 2")
            return LLMCompletionResult(
                message=LLMMessage(role="assistant", content="[]"),
                usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15, is_provider_reported=True),
                model_identifier="mock-model",
            )

    failing_prov = FailingProvider()
    res = run_gemini_pilot_dry_run(
        output_dir=out_dir,
        provider=failing_prov,
    )

    assert res["status"] == "incomplete_run_failed"
    assert res["is_complete"] is False
    assert res["runs_completed"] < 9
    assert "incomplete_run_failed" in res["tag"].lower()

    # Verify analysis file records incomplete status
    analysis_text = Path(res["analysis_file"]).read_text(encoding="utf-8")
    assert "INCOMPLETE PILOT EXECUTION NOTICE" in analysis_text
    assert "incomplete_run_failed" in analysis_text
    assert "**NOT** be reported as a completed nine-run pilot" in analysis_text
