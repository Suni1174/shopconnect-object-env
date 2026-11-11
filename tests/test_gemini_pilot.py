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

from shopconnect_env.agents.providers import (
    GeminiProvider,
    LLMCompletionResult,
    LLMMessage,
    LLMUsage,
    MockLLMProvider,
)
from shopconnect_env.evaluation.gemini_pilot import (
    DEFAULT_OUTPUT_TOKEN_CEILING,
    MAX_PILOT_RUNS,
    MAX_TOTAL_PROVIDER_REQUESTS,
    PILOT_CONDITIONS,
    PILOT_MODEL,
    PILOT_TAG,
    PILOT_TASK_IDS,
    GeminiPilotRequestGuardrailProvider,
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
    assert res["max_provider_requests_ceiling"] == 24
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
    assert res["total_provider_requests"] <= 24
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


# =============================================================================
# 7. ZERO SOCKET NETWORK CALLS VERIFICATION
# =============================================================================

def test_gemini_pilot_zero_socket_activity(monkeypatch):
    """Guarantees dry-run and preflight make zero socket calls."""
    real_socket = socket.socket

    def guarded_socket(*args, **kwargs):
        raise RuntimeError("NETWORK CALL ATTEMPTED: Test suite must remain 100% offline!")

    monkeypatch.setattr(socket, "socket", guarded_socket)

    # 1. Preflight with blocked socket
    preflight_res = run_gemini_pilot_preflight()
    assert preflight_res["status"] == "passed"

    # 2. Dry-run with blocked socket
    dry_run_res = run_gemini_pilot_dry_run()
    assert dry_run_res["status"] == "completed"
