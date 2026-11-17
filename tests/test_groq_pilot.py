"""Unit tests for Groq pilot runner, guardrails, dry run, and connection test.

Verifies:
- Dual confirmation flags required for live execution (--live AND --confirm-live).
- Standalone connection test dual confirmation (--connection-test AND --confirm-connection-test).
- Preflight validation with and without GROQ_API_KEY (zero network access).
- Strict 28-request ceiling enforcement in GroqPilotRequestGuardrailProvider.
- Sequential rate limiting (5.0s minimum interval) with mock clock/sleeper.
- Dispatched failed attempts counted against request ceiling.
- Simulated 9-run dry-run execution across conditions A, B, and C.
- Condition C max 6 model requests and 5 inspections budget for Protocol v2.
- Protocol v2 metrics propagation in raw_runs.jsonl.
- Output directory routing to artifacts/protocol_v2_pilot/groq_<run_id>/.
- Request ceiling enforcement when ceiling reached mid-run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from shopconnect_env.agents.condition_b_agent import ConditionBAgent
from shopconnect_env.agents.condition_c_agent import ConditionCAgent
from shopconnect_env.agents.providers import (
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    GroqProvider,
    LLMCompletionResult,
    LLMMessage,
    LLMUsage,
    MockLLMProvider,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from shopconnect_env.context.base import ContextResult
from shopconnect_env.evaluation.groq_pilot import (
    CONNECTION_TEST_PROMPT,
    EXPECTED_CONNECTION_TEST_TOKEN,
    INSPECT_PLAN_TOOL_CHOICE,
    INSPECT_PLAN_TOOL_DEFINITION,
    MAX_PILOT_RUNS,
    MAX_REQUESTS_PER_CONDITION_A,
    MAX_REQUESTS_PER_CONDITION_B,
    PROTOCOL_V2_GLOBAL_REQUEST_CEILING,
    PROTOCOL_V2_MAX_INSPECTIONS_PER_CONDITION_C,
    PROTOCOL_V2_MAX_REQUESTS_PER_CONDITION_C,
    PROTOCOL_V2_MAX_THEORETICAL_TOTAL_REQUESTS,
    PILOT_MODEL,
    GroqPilotRequestGuardrailProvider,
    SequentialRateLimiter,
    classify_reservation_action,
    create_groq_pilot_mock_provider,
    format_conditions_label,
    format_pilot_sample_size_disclaimer,
    get_project_root,
    is_pre_dispatch_connection_error,
    load_project_env,
    run_groq_connection_test,
    run_groq_json_smoke_test,
    run_groq_pilot_dry_run,
    run_groq_pilot_preflight,
    run_groq_tool_calling_smoke_test,
    run_live_groq_pilot,
)
from shopconnect_env.agents.base import AgentResponse
from shopconnect_env.evaluation.runner import OutcomeEvaluator, PlanDiffResult, RunRecord
from shopconnect_env.evaluation.tasks import BenchmarkTask, ExpectedOutcome


# =============================================================================
# 1. PREFLIGHT TESTS (ZERO NETWORK ACTIVITY)
# =============================================================================

def test_groq_preflight_missing_api_key(monkeypatch):
    """Preflight must report api_key_configured=False when variable is missing."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    res = run_groq_pilot_preflight()
    assert res["status"] == "passed"  # Data files pass, only key is reported false
    assert res["api_key_configured"] is False
    assert res["planned_runs_count"] == 9
    assert res["max_requests_per_condition_c"] == 6
    assert res["max_inspections_per_condition_c"] == 5
    assert res["max_theoretical_requests"] == 24
    assert res["global_request_ceiling"] == 28
    assert res["protocol_version"] == "v2"


def test_groq_preflight_with_key(monkeypatch):
    """Preflight passes completely when data files and key are present."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-key")
    res = run_groq_pilot_preflight()
    assert res["status"] == "passed"
    assert res["api_key_configured"] is True
    assert res["tasks_loaded"] == 3
    assert res["pilot_model"] == "qwen/qwen3.8-27b"
    assert res["billing_tier"] == "free"


# =============================================================================
# 2. GUARDRAIL & RATE LIMITER TESTS
# =============================================================================

def test_groq_guardrail_ceiling_enforcement():
    """Guardrail must raise RuntimeError when ceiling is reached."""
    mock_provider = MockLLMProvider()
    guardrail = GroqPilotRequestGuardrailProvider(
        provider=mock_provider,
        max_requests=3,
    )

    msg = [LLMMessage(role="user", content="Ping")]
    guardrail.complete(msg)
    guardrail.complete(msg)
    guardrail.complete(msg)
    assert guardrail.total_requests == 3

    with pytest.raises(RuntimeError, match="Groq pilot execution safety ceiling reached"):
        guardrail.complete(msg)


def test_groq_guardrail_counts_failed_attempts():
    """Failed provider invocations must be counted against total_requests ceiling."""
    mock_provider = MagicMock()
    mock_provider.complete.side_effect = RateLimitError("Rate limit reached")

    guardrail = GroqPilotRequestGuardrailProvider(
        provider=mock_provider,
        max_requests=2,
    )

    msg = [LLMMessage(role="user", content="Ping")]
    with pytest.raises(RateLimitError):
        guardrail.complete(msg)

    assert guardrail.total_requests == 1


def test_groq_sequential_rate_limiter():
    """Rate limiter must enforce minimum 5.0s interval between requests."""
    fake_time = 1000.0
    sleeps: list[float] = []

    def mock_clock() -> float:
        return fake_time

    def mock_sleeper(seconds: float) -> None:
        nonlocal fake_time
        sleeps.append(seconds)
        fake_time += seconds

    limiter = SequentialRateLimiter(
        min_interval_seconds=5.0,
        clock=mock_clock,
        sleeper=mock_sleeper,
    )

    # First request: no prior request -> 0s sleep
    s1 = limiter.wait_for_slot()
    assert s1 == 0.0

    # Simulate 2s elapsed
    fake_time += 2.0
    s2 = limiter.wait_for_slot()
    assert s2 == 3.0  # 5.0 - 2.0 = 3.0s sleep

    # Simulate 10s elapsed (more than 5s) -> 0s sleep
    fake_time += 10.0
    s3 = limiter.wait_for_slot()
    assert s3 == 0.0
    assert len(sleeps) == 1
    assert sleeps[0] == 3.0


# =============================================================================
# 3. DRY RUN TESTS (ZERO NETWORK CALLS)
# =============================================================================

def test_groq_pilot_dry_run_execution(tmp_path):
    """Dry run must complete 9 runs and generate isolated artifacts with zero network activity."""
    res = run_groq_pilot_dry_run(output_dir=tmp_path)
    assert res["status"] == "completed"
    assert res["runs_completed"] == 9
    assert res["runs_planned"] == 9
    assert res["network_calls_made"] == 0
    assert res["total_provider_requests"] <= PROTOCOL_V2_MAX_THEORETICAL_TOTAL_REQUESTS
    assert res["max_requests_per_condition_c"] == 6
    assert res["max_inspections_per_condition_c"] == 5
    assert res["max_requests_ceiling"] == 28
    assert res["protocol_version"] == "v2"

    raw_runs = tmp_path / "raw_runs.jsonl"
    comparison = tmp_path / "comparison.csv"
    analysis = tmp_path / "analysis.md"

    assert raw_runs.exists()
    assert comparison.exists()
    assert analysis.exists()

    lines = raw_runs.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 9

    # Verify each record has protocol_version v2 and decoupled metrics
    for line in lines:
        rec = json.loads(line)
        assert rec["protocol_version"] == "v2"
        assert "legacy_v1_task_success" in rec
        assert "system_safety_success" in rec
        assert "model_decision_correct" in rec
        assert "target_operation_correct" in rec


def test_groq_pilot_dry_run_default_output_routing():
    """Default output directory must route to artifacts/protocol_v2_pilot/groq_<run_id>/."""
    res = run_groq_pilot_dry_run()
    out_dir = Path(res["output_directory"])
    try:
        assert "artifacts" in out_dir.parts
        assert "protocol_v2_pilot" in out_dir.parts
        assert out_dir.name.startswith("groq_dry_run_")
        assert (out_dir / "raw_runs.jsonl").exists()
        assert (out_dir / "comparison.csv").exists()
        assert (out_dir / "analysis.md").exists()
    finally:
        import shutil
        if out_dir.exists() and "protocol_v2_pilot" in out_dir.parts:
            shutil.rmtree(out_dir, ignore_errors=True)


def test_groq_pilot_dry_run_ceiling_enforcement(tmp_path):
    """Dry run must halt cleanly when request ceiling is reached."""
    res = run_groq_pilot_dry_run(output_dir=tmp_path, max_requests=4)
    assert res["status"] == "incomplete"
    assert res["ceiling_reached"] is True
    assert res["total_provider_requests"] == 4
    assert "ceiling" in res["incomplete_reason"].lower()


# =============================================================================
# 4. CONNECTION TEST TESTS
# =============================================================================

def test_groq_connection_test_requires_dual_confirmation():
    """Connection test must reject execution if either flag is missing."""
    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_groq_connection_test(connection_test=True, confirm_connection_test=False)

    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_groq_connection_test(connection_test=False, confirm_connection_test=True)


def test_groq_connection_test_missing_key(monkeypatch, tmp_path):
    """Connection test records clean failure artifact when API key is missing."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    res = run_groq_connection_test(
        output_dir=tmp_path,
        connection_test=True,
        confirm_connection_test=True,
    )
    assert res["status"] == "failed"
    assert res["success"] is False
    assert res["error_type"] == "MissingAPIKeyError"
    assert res["requests_made"] == 0
    assert (tmp_path / "connection_test_result.json").exists()


def test_groq_connection_test_success_with_injected_client(tmp_path):
    """Connection test succeeds when expected token is returned by mock client."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="groq-conn-123",
        model="qwen/qwen3.8-27b",
        system_fingerprint="fp_groq",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content=f"Hello! {EXPECTED_CONNECTION_TEST_TOKEN}",
                    tool_calls=None,
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=15,
            completion_tokens=8,
            total_tokens=23,
        ),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    res = run_groq_connection_test(
        api_key="gsk-test",
        output_dir=tmp_path,
        connection_test=True,
        confirm_connection_test=True,
        client=mock_client,
    )

    assert res["status"] == "passed"
    assert res["success"] is True
    assert res["requests_made"] == 1
    assert res["actual_model_identifier"] == "qwen/qwen3.8-27b"
    assert res["model_requested"] == PILOT_MODEL
    assert (tmp_path / "connection_test_result.json").exists()


# =============================================================================
# 5. LIVE RUNNER DUAL CONFIRMATION GATE
# =============================================================================

def test_groq_live_runner_dual_confirmation():
    """Live runner must reject execution if either flag is missing."""
    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_live_groq_pilot(live=True, confirm_live=False)

    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_live_groq_pilot(live=False, confirm_live=True)


# =============================================================================
# 6. TOOL CALLING SMOKE TEST
# =============================================================================

def test_groq_tool_calling_smoke_test_requires_dual_confirmation():
    """Tool calling smoke test must reject execution if either confirmation flag is missing."""
    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_groq_tool_calling_smoke_test(smoke_test_tool_call=True, confirm_smoke_test_tool_call=False)

    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_groq_tool_calling_smoke_test(smoke_test_tool_call=False, confirm_smoke_test_tool_call=True)


def test_groq_tool_calling_smoke_test_missing_key(monkeypatch, tmp_path):
    """Tool calling smoke test records failure artifact when API key is missing."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    res = run_groq_tool_calling_smoke_test(
        output_dir=tmp_path,
        smoke_test_tool_call=True,
        confirm_smoke_test_tool_call=True,
    )
    assert res["status"] == "failed"
    assert res["success"] is False
    assert res["error_type"] == "MissingAPIKeyError"
    assert res["requests_made"] == 0
    assert (tmp_path / "tool_calling_smoke_test_result.json").exists()


def test_groq_tool_calling_smoke_test_success_with_injected_client(tmp_path):
    """Tool calling smoke test succeeds when inspect_plan tool call is returned by mock client."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="groq-tool-123",
        model="qwen/qwen3.8-27b",
        system_fingerprint="fp_groq_tool",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_inspect_plan_1",
                            type="function",
                            function=SimpleNamespace(
                                name="inspect_plan",
                                arguments="{}",
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=42,
            completion_tokens=12,
            total_tokens=54,
        ),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    res = run_groq_tool_calling_smoke_test(
        api_key="gsk-test",
        output_dir=tmp_path,
        smoke_test_tool_call=True,
        confirm_smoke_test_tool_call=True,
        client=mock_client,
    )

    assert res["status"] == "passed"
    assert res["success"] is True
    assert res["requests_made"] == 1
    assert res["tool_calls_count"] == 1
    assert res["tool_call_name_matches"] is True
    assert res["tool_call_arguments_valid_json"] is True
    assert res["function_executed"] is False
    assert res["state_mutated"] is False
    assert (tmp_path / "tool_calling_smoke_test_result.json").exists()

    # Verify tool_choice was passed as named inspect_plan and response_format was None
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs.get("tools") is not None
    assert call_kwargs.get("tool_choice") == INSPECT_PLAN_TOOL_CHOICE
    assert "response_format" not in call_kwargs or call_kwargs.get("response_format") is None


def test_groq_tool_calling_smoke_test_mismatched_or_multiple_calls(tmp_path):
    """Tool calling smoke test fails cleanly when multiple tool calls or wrong tool returned."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="groq-tool-err",
        model="qwen/qwen3.8-27b",
        system_fingerprint="fp_err",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            type="function",
                            function=SimpleNamespace(
                                name="wrong_tool_name",
                                arguments="{}",
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    res = run_groq_tool_calling_smoke_test(
        api_key="gsk-test",
        output_dir=tmp_path,
        smoke_test_tool_call=True,
        confirm_smoke_test_tool_call=True,
        client=mock_client,
    )

    assert res["status"] == "failed"
    assert res["success"] is False
    assert res["error_type"] == "ToolNameMismatchError"
    assert res["tool_call_name_matches"] is False


# =============================================================================
# 7. CONDITION A JSON ENVELOPE SMOKE TEST
# =============================================================================

def test_groq_json_smoke_test_requires_dual_confirmation():
    """Condition A JSON smoke test must reject execution if either confirmation flag is missing."""
    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_groq_json_smoke_test(smoke_test_json=True, confirm_smoke_test_json=False)

    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_groq_json_smoke_test(smoke_test_json=False, confirm_smoke_test_json=True)


def test_groq_json_smoke_test_missing_key(monkeypatch, tmp_path):
    """JSON smoke test records failure artifact when API key is missing."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    res = run_groq_json_smoke_test(
        output_dir=tmp_path,
        smoke_test_json=True,
        confirm_smoke_test_json=True,
    )
    assert res["status"] == "failed"
    assert res["success"] is False
    assert res["error_type"] == "MissingAPIKeyError"
    assert res["requests_made"] == 0
    assert (tmp_path / "json_smoke_test_result.json").exists()


def test_groq_json_smoke_test_success_with_injected_client(tmp_path):
    """JSON smoke test succeeds when valid Condition A REJECT envelope is returned."""
    mock_client = MagicMock()
    canned_json = (
        '{\n'
        '  "decision": "REJECT",\n'
        '  "rejection_category": "allergen_conflict",\n'
        '  "reason": "Peanut oil in Thai Peanut Noodles conflicts with registered child peanut allergy."\n'
        '}'
    )
    mock_resp = SimpleNamespace(
        id="groq-json-123",
        model="qwen/qwen3.8-27b",
        system_fingerprint="fp_groq_json",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content=f"```json\n{canned_json}\n```",
                    tool_calls=None,
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=85,
            completion_tokens=35,
            total_tokens=120,
        ),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    res = run_groq_json_smoke_test(
        api_key="gsk-test",
        output_dir=tmp_path,
        smoke_test_json=True,
        confirm_smoke_test_json=True,
        client=mock_client,
    )

    assert res["status"] == "passed"
    assert res["success"] is True
    assert res["requests_made"] == 1
    assert res["json_parsed_successfully"] is True
    assert res["pydantic_envelope_validated"] is True
    assert res["envelope_decision"] == "REJECT"
    assert res["envelope_rejection_category"] == "allergen_conflict"
    assert "peanut allergy" in res["envelope_reason"].lower()
    assert (tmp_path / "json_smoke_test_result.json").exists()

    # Verify tools was None, tool_choice was None, response_format was None (prompt-directed JSON)
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "tools" not in call_kwargs or call_kwargs.get("tools") is None
    assert "tool_choice" not in call_kwargs or call_kwargs.get("tool_choice") is None
    assert "response_format" not in call_kwargs or call_kwargs.get("response_format") is None


def test_groq_json_smoke_test_invalid_envelope(tmp_path):
    """JSON smoke test fails cleanly when envelope fails Pydantic schema validation."""
    mock_client = MagicMock()
    # Missing reason and invalid rejection category
    invalid_json = '{"decision": "REJECT", "rejection_category": "unknown_cat", "reason": ""}'
    mock_resp = SimpleNamespace(
        id="groq-json-err",
        model="qwen/qwen3.8-27b",
        system_fingerprint="fp_err",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content=invalid_json,
                    tool_calls=None,
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=40, completion_tokens=15, total_tokens=55),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    res = run_groq_json_smoke_test(
        api_key="gsk-test",
        output_dir=tmp_path,
        smoke_test_json=True,
        confirm_smoke_test_json=True,
        client=mock_client,
    )

    assert res["status"] == "failed"
    assert res["success"] is False
    assert res["error_type"] == "ValidationError"
    assert res["pydantic_envelope_validated"] is False


# =============================================================================
# 7. RATE LIMIT PACING, PROMPT ESTIMATION & MULTI-TURN EXCEPTION HANDLING TESTS
# =============================================================================

def test_groq_sequential_rate_limiter_pacing_from_headers():
    """Rate limiter must enforce pacing when retry-after or low token quota headers appear."""
    current_time = [100.0]
    sleeps = []

    def mock_clock():
        return current_time[0]

    def mock_sleep(d):
        sleeps.append(d)
        current_time[0] += d

    limiter = SequentialRateLimiter(min_interval_seconds=5.0, clock=mock_clock, sleeper=mock_sleep)

    # 1. Normal first call - no sleep needed
    limiter.wait_for_slot()
    assert len(sleeps) == 0

    # 2. Update with retry-after header = 15.0
    limiter.update_from_headers({"retry-after": "15.0"})
    limiter.wait_for_slot()
    assert sleeps[-1] == 15.0

    # 3. Update with low remaining tokens and reset time
    limiter.update_from_headers({
        "x-ratelimit-remaining-tokens": "1200",
        "x-ratelimit-reset-tokens": "8.5s",
    })
    limiter.wait_for_slot()
    assert sleeps[-1] == 8.5


def test_groq_adaptive_rate_limiter_next_request_size_comparison():
    """Rate limiter must trigger backoff when remaining tokens (e.g. 2600) is less than next request size (e.g. 2800)."""
    current_time = [100.0]
    sleeps = []

    def mock_clock():
        return current_time[0]

    def mock_sleep(d):
        sleeps.append(d)
        current_time[0] += d

    limiter = SequentialRateLimiter(min_interval_seconds=5.0, clock=mock_clock, sleeper=mock_sleep)

    # Initial call to establish baseline
    limiter.wait_for_slot()

    # Remaining tokens is 2600 (above old 2500 threshold), but next request requires 2800 tokens.
    # A fixed 2500 threshold would NOT have waited.
    # The adaptive rate limiter MUST wait for the reset duration (6.5s).
    limiter.update_from_headers(
        {"x-ratelimit-remaining-tokens": "2600", "x-ratelimit-reset-tokens": "6.5s"},
        estimated_next_tokens=2800,
    )
    limiter.wait_for_slot(estimated_tokens=2800)
    assert sleeps[-1] == 6.5

    # Conversely, if remaining tokens is 2600 and next request is small (700 tokens):
    # 2600 >= (700 + 200 = 900), so no quota reset delay should be required.
    limiter.update_from_headers(
        {"x-ratelimit-remaining-tokens": "2600", "x-ratelimit-reset-tokens": "6.5s"},
        estimated_next_tokens=700,
    )
    current_time[0] += 10.0
    slept = limiter.wait_for_slot(estimated_tokens=700)
    assert slept == 0.0


def test_groq_preflight_prompt_estimation_condition_a_rejection(monkeypatch):
    """Preflight must reject Condition A when enforce_itpm_limits=True because prompt size exceeds 7k ITPM."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    res = run_groq_pilot_preflight(enforce_itpm_limits=True)
    assert res["status"] == "failed"
    assert len(res["errors"]) >= 1
    assert any(
        "Condition A estimated prompt size" in err and "exceeds Groq Free Tier ITPM limit of 7000" in err
        for err in res["errors"]
    )


def test_groq_preflight_prompt_estimation_conditions_b_and_c_pass(monkeypatch):
    """Preflight passes with enforce_itpm_limits=True when only Conditions B and C are selected."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    res = run_groq_pilot_preflight(conditions=("B", "C"), enforce_itpm_limits=True)
    assert res["status"] == "passed"
    assert len(res["errors"]) == 0
    assert res["conditions"] == ["B", "C"]
    assert res["planned_runs_count"] == 6
    assert res["estimated_prompt_tokens"]["B"] < 7000
    assert res["estimated_prompt_tokens"]["C"] < 7000


def test_condition_c_agent_preserves_turn1_usage_on_turn2_exception():
    """Condition C agent must preserve Turn 1 token usage and request count when Turn 2 raises RateLimitError."""
    mock_provider = MagicMock()
    turn1_resp = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            content=None,
            tool_calls=[{
                "id": "call_1",
                "name": "inspect_plan",
                "arguments": "{}",
            }],
        ),
        usage=LLMUsage(input_tokens=1700, output_tokens=30, total_tokens=1730),
        model_identifier="qwen/qwen3.8-27b",
        provider_request_id="groq_req_turn1",
        latency_ms=250.0,
        finish_reason="tool_calls",
    )
    # Turn 1 succeeds with inspect_plan tool call, Turn 2 raises RateLimitError
    mock_provider.complete.side_effect = [
        turn1_resp,
        RateLimitError("Rate limit exceeded 429"),
    ]

    agent = ConditionCAgent(provider=mock_provider, max_model_requests=6, max_inspection_calls=5)
    ctx = ContextResult(
        condition="Condition C: Object-Centric Intelligence",
        condition_id="C",
        system_prompt="system",
        task_prompt="task",
        serialized_context='{"plan_version": 1}',
        metadata={"plan_version": 1},
    )
    mock_inspector = MagicMock()
    mock_inspector.inspect_plan.return_value = {"status": "ok", "plan_version": 1}

    resp = agent.propose_action(context=ctx, inspector=mock_inspector)

    assert resp.proposal is None
    assert resp.metadata["status"] == "error"
    assert resp.metadata["total_model_requests"] == 2
    assert "Rate limit exceeded 429" in resp.metadata["error"]
    # Turn 1 usage must be preserved!
    usage_dict = resp.metadata_dict["usage"]
    assert usage_dict["input_tokens"] == 1700
    assert usage_dict["output_tokens"] == 30
    assert usage_dict["total_tokens"] == 1730
    assert "groq_req_turn1" in resp.metadata["provider_request_ids"]


def test_outcome_evaluator_decouples_infrastructure_errors():
    """OutcomeEvaluator must set model_decision_correct=None on infrastructure errors (413/429/timeout)."""
    task = BenchmarkTask(
        task_id="task_01_commit_replace_wed_dinner",
        fixture_id="fixture_standard",
        category="meal_replacement",
        user_request="Replace Wednesday dinner with Vegetable Pasta Primavera",
        expected_outcome=ExpectedOutcome.COMMIT,
    )
    diff = PlanDiffResult(
        target_day="Wednesday",
        target_meal_type="dinner",
        target_slot_before=None,
        target_slot_after=None,
        target_changed=False,
        actual_target_recipe_id=None,
        actual_target_servings=None,
        all_changed_coordinates=[],
        non_target_changes=[],
        non_target_drift_count=0,
        unintended_assignments=0,
        non_target_recipe_overwrites=0,
        basket_consistent=True,
        policy_violations=[],
        plan_mutated=False,
        basket_mutated=False,
    )

    # 1. Infrastructure error run (e.g. 413) with no raw output
    resp_infra = AgentResponse(
        proposal=None,
        rationale="Agent crashed during completion: 413 Request too large",
        raw_output="",
    )
    eval_infra = OutcomeEvaluator.evaluate(
        task=task,
        response=resp_infra,
        action_result=None,
        plan_diff=diff,
        execution_error="Request failed with status code 413: Limit 7000, Requested 12182.",
        protocol_version="v2",
    )
    assert eval_infra["model_decision_correct"] is None
    assert eval_infra["target_accuracy"] is None
    assert eval_infra["actual_outcome"] == "ERROR"

    # 2. Non-infra model decision error (e.g. malformed model response)
    resp_non_infra = AgentResponse(
        proposal=None,
        rationale="Model returned invalid JSON",
        raw_output="{broken json",
    )
    eval_model_err = OutcomeEvaluator.evaluate(
        task=task,
        response=resp_non_infra,
        action_result=None,
        plan_diff=diff,
        execution_error="JSONDecodeError: Expecting property name",
        protocol_version="v2",
    )
    assert eval_model_err["model_decision_correct"] is False
    assert eval_model_err["target_accuracy"] is False

    # 3. Model schema and tool errors (HTTP 400, bad tool name, missing properties) must NOT decouple
    eval_tool_err = OutcomeEvaluator.evaluate(
        task=task,
        response=resp_infra,
        action_result=None,
        plan_diff=diff,
        execution_error="Groq API provider error (BadRequestError): Error code: 400 - tool call validation failed: attempted to call tool 'replace_memal' which was not in request.tools",
        protocol_version="v2",
    )
    assert eval_tool_err["model_decision_correct"] is False
    assert eval_tool_err["target_accuracy"] is False
    assert eval_tool_err["actual_outcome"] == "ERROR"

    eval_schema_err = OutcomeEvaluator.evaluate(
        task=task,
        response=resp_infra,
        action_result=None,
        plan_diff=diff,
        execution_error="Groq API provider error (BadRequestError): Error code: 400 - tool call validation failed: parameters for tool finalize_no_action did not match schema: errors: [missing properties: 'rejection_category']",
        protocol_version="v2",
    )
    assert eval_schema_err["model_decision_correct"] is False
    assert eval_schema_err["target_accuracy"] is False
    assert eval_schema_err["actual_outcome"] == "ERROR"


def test_condition_b_groq_passes_parallel_tool_calls_false():
    """Condition B agent interfacing with GroqProvider must pass parallel_tool_calls=False."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="chatcmpl-b-ptc",
        model="qwen/qwen3.8-27b",
        system_fingerprint="fp_b",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="No action", tool_calls=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    provider = GroqProvider(api_key="gsk-test", client=mock_client)
    guardrail = GroqPilotRequestGuardrailProvider(provider=provider, max_requests=10)
    agent = ConditionBAgent(provider=guardrail)

    ctx = ContextResult(
        condition="Condition B: Flat Tool Definitions",
        condition_id="B",
        system_prompt="system",
        task_prompt="Replace Wednesday dinner with Vegetable Pasta Primavera",
        serialized_context='{"plan_version": 1}',
        metadata={"plan_version": 1},
    )

    agent.propose_action(context=ctx)

    assert mock_client.chat.completions.create.called
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["parallel_tool_calls"] is False


def test_condition_c_groq_passes_parallel_tool_calls_false():
    """Condition C agent interfacing with GroqProvider must pass parallel_tool_calls=False."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="chatcmpl-c-ptc",
        model="qwen/qwen3.8-27b",
        system_fingerprint="fp_c",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="Finalized", tool_calls=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    provider = GroqProvider(api_key="gsk-test", client=mock_client)
    guardrail = GroqPilotRequestGuardrailProvider(provider=provider, max_requests=10)
    agent = ConditionCAgent(provider=guardrail, max_model_requests=6, max_inspection_calls=5)

    ctx = ContextResult(
        condition="Condition C: Object-Centric Intelligence",
        condition_id="C",
        system_prompt="system",
        task_prompt="Check peanut allergy",
        serialized_context='{"plan_version": 1}',
        metadata={"plan_version": 1},
    )

    agent.propose_action(context=ctx)

    assert mock_client.chat.completions.create.called
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["parallel_tool_calls"] is False


# =============================================================================
# 8. PILOT 3 READINESS REGRESSION TESTS
# =============================================================================

def test_rate_limiter_accounts_for_request_duration_without_undersleeping():
    """Rate limiter must use monotonic deadline to prevent execution latency from reducing token wait."""
    current_time = [100.0]
    sleeps = []

    def mock_clock() -> float:
        return current_time[0]

    def mock_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        current_time[0] += seconds

    limiter = SequentialRateLimiter(min_interval_seconds=5.0, clock=mock_clock, sleeper=mock_sleep)

    # 1. First request starts at T=100.0
    s1 = limiter.wait_for_slot()
    assert s1 == 0.0

    # 2. Request execution takes 4.5 seconds
    current_time[0] += 4.5  # Monotonic clock advances to T=104.5

    # 3. At response receipt, Groq reports remaining_tokens=500 and reset_tokens=6.8s
    limiter.update_from_headers({
        "x-ratelimit-remaining-tokens": "500",
        "x-ratelimit-reset-tokens": "6.8s",
    })

    # 4. Next request arrives 0.1 seconds later (T=104.6)
    current_time[0] += 0.1  # Monotonic clock is now T=104.6

    # 5. Estimated next request requires 2,800 tokens
    slept = limiter.wait_for_slot(estimated_tokens=2800)

    # Under old implementation (measuring elapsed from last_request_start=100.0):
    # elapsed = 104.6 - 100.0 = 4.6s -> wait_time = 6.8 - 4.6 = 2.2s (under-slept!)
    # Under fixed implementation (measuring deadline from header receipt=104.5):
    # deadline = 104.5 + 6.8 = 111.3 -> wait_time = 111.3 - 104.6 = 6.7s
    assert slept >= 6.7
    assert sleeps[-1] >= 6.7


def test_condition_c_preserves_tokens_when_inspection_dispatch_raises():
    """Condition C must preserve accumulated tokens if an exception occurs inside inspector dispatch."""
    mock_provider = MagicMock()
    turn1_resp = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            content=None,
            tool_calls=[{"id": "c1", "name": "inspect_plan", "arguments": "{}"}],
        ),
        usage=LLMUsage(input_tokens=1500, output_tokens=25, total_tokens=1525, is_provider_reported=True),
        model_identifier="qwen/qwen3.8-27b",
        provider_request_id="groq_turn1",
        latency_ms=300.0,
        finish_reason="tool_calls",
    )
    mock_provider.complete.return_value = turn1_resp

    agent = ConditionCAgent(provider=mock_provider, max_model_requests=6, max_inspection_calls=5)
    ctx = ContextResult(
        condition="Condition C: Object-Centric Intelligence",
        condition_id="C",
        system_prompt="sys",
        task_prompt="task",
        serialized_context='{"plan_version": 1}',
        metadata={"plan_version": 1},
    )

    mock_inspector = MagicMock()
    mock_inspector.inspect_plan.side_effect = RuntimeError("Simulated inspector dispatch crash")

    resp = agent.propose_action(context=ctx, inspector=mock_inspector)

    assert resp.proposal is None
    assert resp.metadata["status"] == "error"
    assert "Inspection dispatch error" in resp.metadata["error"]
    assert "Simulated inspector dispatch crash" in resp.metadata["error"]
    assert resp.metadata["total_model_requests"] == 1
    assert "groq_turn1" in resp.metadata["provider_request_ids"]
    assert 300.0 in resp.metadata["per_request_latencies_ms"]
    assert resp.metadata_dict["usage"]["input_tokens"] == 1500
    assert resp.metadata_dict["usage"]["output_tokens"] == 25
    assert resp.metadata_dict["usage"]["total_tokens"] == 1525
    assert resp.metadata_dict["usage"]["is_provider_reported"] is True


def test_rate_limiter_preserves_deadline_across_benchmark_tasks():
    """A pending quota deadline set during Task 1 must be honored by the first request of Task 2."""
    current_time = [100.0]
    sleeps = []

    def mock_clock() -> float:
        return current_time[0]

    def mock_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        current_time[0] += seconds

    limiter = SequentialRateLimiter(min_interval_seconds=5.0, clock=mock_clock, sleeper=mock_sleep)

    # Task 1 request runs at T=100.0
    limiter.wait_for_slot()
    current_time[0] += 2.0  # T=102.0

    # Task 1 receives response with 10.0s reset duration and low remaining tokens
    limiter.update_from_headers({
        "x-ratelimit-remaining-tokens": "200",
        "x-ratelimit-reset-tokens": "10.0s",
    })
    # Monotonic deadline is 102.0 + 10.0 = 112.0

    # Task 1 completes, runner transitions to Task 2 at T=103.0 (1.0s later)
    current_time[0] += 1.0  # T=103.0

    # Task 2 first request arrives, requiring 2800 tokens (Condition B)
    slept = limiter.wait_for_slot(estimated_tokens=2800)

    # Expected wait: 112.0 - 103.0 = 9.0s
    assert slept >= 9.0
    assert current_time[0] >= 112.0


# =============================================================================
# 9. ENVIRONMENT CONFIGURATION & PROJECT ROOT TESTS
# =============================================================================

def test_get_project_root_resolution():
    """get_project_root must robustly resolve directory containing requirements.txt."""
    root = get_project_root()
    assert isinstance(root, Path)
    assert root.exists()
    assert (root / "requirements.txt").exists()


def test_load_project_env_precedence_and_override_false(monkeypatch, tmp_path):
    """load_project_env must load variables from file without overriding existing env vars."""
    fake_env = tmp_path / ".env"
    fake_env.write_text("GROQ_API_KEY=gsk_from_file_secret\nSHOPCONNECT_SAMPLE_VAR=from_file\n", encoding="utf-8")

    # Set existing variable in os.environ
    monkeypatch.setenv("GROQ_API_KEY", "gsk_existing_env_key")
    monkeypatch.delenv("SHOPCONNECT_SAMPLE_VAR", raising=False)

    success = load_project_env(env_path=fake_env, force_reload=True)
    assert success is True

    # Existing variable takes precedence (override=False)
    assert os.environ.get("GROQ_API_KEY") == "gsk_existing_env_key"
    # Unset variable is loaded from file
    assert os.environ.get("SHOPCONNECT_SAMPLE_VAR") == "from_file"


def test_load_project_env_nonexistent_file(tmp_path):
    """load_project_env returns False cleanly when .env file does not exist."""
    missing_env = tmp_path / "nonexistent.env"
    assert load_project_env(env_path=missing_env, force_reload=True) is False


def test_load_project_env_loads_key_when_unset(monkeypatch, tmp_path):
    """load_project_env loads key into environment when currently unset, without exposing secrets in results."""
    fake_env = tmp_path / ".env"
    test_key_val = "gsk_test_mock_secret_key_abc123"
    fake_env.write_text(f"GROQ_API_KEY={test_key_val}\n", encoding="utf-8")

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert os.environ.get("GROQ_API_KEY") is None

    success = load_project_env(env_path=fake_env, force_reload=True)
    assert success is True
    assert os.environ.get("GROQ_API_KEY") == test_key_val

    # Preflight should now report api_key_configured=True without leaking the secret
    res = run_groq_pilot_preflight()
    assert res["api_key_configured"] is True
    assert test_key_val not in str(res)  # Secret must never be in preflight result dictionary


# =============================================================================
# 10. SLIDING WINDOW RATE LIMITER & PACING REGRESSION TESTS
# =============================================================================

def test_sliding_window_expiration_after_window_seconds():
    """Tokens in sliding window must expire exactly after window_seconds elapses."""
    current_time = [100.0]

    def mock_clock() -> float:
        return current_time[0]

    limiter = SequentialRateLimiter(
        window_seconds=60.0,
        itpm_limit=7000,
        reserve_tokens=1000,
        clock=mock_clock,
    )

    # Reserve 2500 tokens at T=100.0
    res_id = limiter.reserve_slot(2500)
    assert limiter.get_window_tokens() == 2500
    assert len(limiter.window_entries) == 1

    # Advance time by 30 seconds to T=130.0 (within 60s window)
    current_time[0] = 130.0
    assert limiter.get_window_tokens() == 2500
    assert len(limiter.window_entries) == 1

    # Advance time by 31 more seconds to T=161.0 (> 60s elapsed from T=100.0)
    current_time[0] = 161.0
    assert limiter.get_window_tokens() == 0
    assert len(limiter.window_entries) == 0


def test_sliding_window_budget_triggers_pacing_wait():
    """Requests exceeding effective budget must wait until older entries slide out of window."""
    current_time = [100.0]
    sleeps: list[float] = []

    def mock_clock() -> float:
        return current_time[0]

    def mock_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        current_time[0] += seconds

    limiter = SequentialRateLimiter(
        min_interval_seconds=5.0,
        window_seconds=60.0,
        itpm_limit=7000,
        reserve_tokens=1000,  # Effective budget: 6,000 tokens
        clock=mock_clock,
        sleeper=mock_sleep,
    )

    # Request 1 at T=100.0: 2,800 tokens (fits within 6000)
    s1 = limiter.wait_for_slot(estimated_tokens=2800)
    assert s1 == 0.0
    limiter.last_request_start = current_time[0]
    assert limiter.get_window_tokens() == 2800

    # Monotonic clock advances to T=105.0 (min_interval satisfied)
    current_time[0] = 105.0

    # Request 2 at T=105.0: 2,800 tokens (2800 + 2800 = 5600 <= 6000)
    s2 = limiter.wait_for_slot(estimated_tokens=2800)
    assert s2 == 0.0
    limiter.last_request_start = current_time[0]
    assert limiter.get_window_tokens() == 5600

    # Monotonic clock advances to T=110.0
    current_time[0] = 110.0

    # Request 3 at T=110.0: 1,500 tokens.
    # Active window tokens = 5600. 5600 + 1500 = 7100 > 6000.
    # Must wait for Request 1 (at T=100.0) to slide out at T=160.0.
    # Expected wait: 160.0 - 110.0 + 0.05 = 50.05s
    s3 = limiter.wait_for_slot(estimated_tokens=1500)
    assert s3 >= 50.0
    assert len(sleeps) == 1
    assert sleeps[0] >= 50.0
    assert current_time[0] >= 160.0

    # Verify reason in sleep_records
    assert any(rec["reason"] == "sliding_window_budget" for rec in limiter.sleep_records)

    # After wait, Request 1 has expired, leaving Request 2 (2800 tokens) + Request 3 (1500 tokens).
    # 2800 + 1500 = 4300 <= 6000 -> fits!
    assert limiter.get_window_tokens() == 4300


def test_sliding_window_failed_dispatch_releases_reservation():
    """Failed provider dispatch must release reservation so failed attempts do not consume quota."""
    current_time = [100.0]

    def mock_clock() -> float:
        return current_time[0]

    limiter = SequentialRateLimiter(
        window_seconds=60.0,
        itpm_limit=7000,
        reserve_tokens=1000,
        clock=mock_clock,
    )

    res_id = limiter.reserve_slot(2500)
    assert limiter.get_window_tokens() == 2500

    # Simulate dispatch failure and release
    limiter.release_reservation(res_id)
    assert limiter.get_window_tokens() == 0
    assert len(limiter.window_entries) == 0


def test_sliding_window_successful_dispatch_reconciles_actual_tokens():
    """Successful dispatch must reconcile estimated reservation with actual input tokens."""
    current_time = [100.0]

    def mock_clock() -> float:
        return current_time[0]

    limiter = SequentialRateLimiter(
        window_seconds=60.0,
        itpm_limit=7000,
        reserve_tokens=1000,
        clock=mock_clock,
    )

    res_id = limiter.reserve_slot(2800)
    assert limiter.get_window_tokens() == 2800

    # Provider reports 2650 input tokens (including any cached tokens)
    limiter.reconcile_reservation(res_id, actual_tokens=2650)
    assert limiter.get_window_tokens() == 2650
    assert limiter.window_entries[0]["status"] == "confirmed"
    assert limiter.window_entries[0]["tokens"] == 2650


def test_sliding_window_quota_reset_headers_merge_with_sliding_window_delay():
    """Rate limiter must wait for the maximum of sliding window wait, quota deadline, and interval."""
    current_time = [100.0]
    sleeps: list[float] = []

    def mock_clock() -> float:
        return current_time[0]

    def mock_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        current_time[0] += seconds

    limiter = SequentialRateLimiter(
        min_interval_seconds=5.0,
        window_seconds=60.0,
        itpm_limit=7000,
        reserve_tokens=1000,
        clock=mock_clock,
        sleeper=mock_sleep,
    )

    # 1. Fill window with 5000 tokens at T=100.0
    limiter.reserve_slot(5000)
    limiter.last_request_start = 100.0

    # Advance to T=120.0
    current_time[0] = 120.0

    # Headers report low remaining tokens with 15.0s reset duration (reset deadline: 120.0 + 15.0 = 135.0)
    # Sliding window wait: 5000 + 2000 = 7000 > 6000 -> expires at 100.0 + 60.0 = 160.0 (wait: 40.05s)
    # Since sliding_wait (40.05s) > quota_wait (15.0s), sliding_wait must dominate
    limiter.update_from_headers({
        "x-ratelimit-remaining-tokens": "300",
        "x-ratelimit-reset-tokens": "15.0s",
    })

    slept = limiter.wait_for_slot(estimated_tokens=2000)
    assert slept >= 40.0
    assert limiter.sleep_records[-1]["reason"] == "sliding_window_budget"


def test_sliding_window_persistence_across_multiple_tasks():
    """Sliding-window token budget must persist across multiple tasks via guardrail provider."""
    current_time = [100.0]
    sleeps: list[float] = []

    def mock_clock() -> float:
        return current_time[0]

    def mock_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        current_time[0] += seconds

    limiter = SequentialRateLimiter(
        min_interval_seconds=5.0,
        window_seconds=60.0,
        itpm_limit=7000,
        reserve_tokens=1000,
        clock=mock_clock,
        sleeper=mock_sleep,
    )

    mock_provider = MagicMock()
    mock_provider.complete.return_value = LLMCompletionResult(
        message=LLMMessage(role="assistant", content="OK"),
        usage=LLMUsage(input_tokens=3500, output_tokens=50, total_tokens=3550, is_provider_reported=True),
        model_identifier="qwen/qwen3.8-27b",
        provider_request_id="task1_req1",
        latency_ms=250.0,
        finish_reason="stop",
    )

    guardrail = GroqPilotRequestGuardrailProvider(
        provider=mock_provider,
        max_requests=10,
        rate_limiter=limiter,
    )

    # Task 1 dispatch at T=100.0
    msg = [LLMMessage(role="user", content="Task 1 Prompt")]
    res1 = guardrail.complete(msg, estimated_tokens=3500)
    assert res1.usage.input_tokens == 3500
    assert limiter.get_window_tokens() == 3500

    # Advance to T=120.0 (Task 2 starts 20 seconds later)
    current_time[0] = 120.0

    # Task 2 wants to dispatch a request needing 3200 tokens
    # 3500 + 3200 = 6700 > 6000. Must wait until Task 1 tokens expire (100.0 + 60.0 = 160.0, i.e. 40.05s)
    mock_provider.complete.return_value = LLMCompletionResult(
        message=LLMMessage(role="assistant", content="Task 2 OK"),
        usage=LLMUsage(input_tokens=3200, output_tokens=60, total_tokens=3260, is_provider_reported=True),
        model_identifier="qwen/qwen3.8-27b",
        provider_request_id="task2_req1",
        latency_ms=300.0,
        finish_reason="stop",
    )

    res2 = guardrail.complete(msg, estimated_tokens=3200)
    assert len(sleeps) >= 1
    assert sleeps[-1] >= 40.0
    assert current_time[0] >= 160.0
    assert res2.usage.input_tokens == 3200
    # Task 1 entry expired, so only Task 2 entry remains in window
    assert limiter.get_window_tokens() == 3200


# =============================================================================
# 11. SLIDING-WINDOW RESERVATION SAFETY PATCH REGRESSION TESTS
# =============================================================================

def test_classify_reservation_action_mapping():
    """Verify exception classification into release vs retain categories."""
    # Release cases:
    assert classify_reservation_action(RateLimitError("Rate limit reached")) == "release"
    assert classify_reservation_action(ProviderError("Error code: 400 - tool call validation failed")) == "release"
    assert classify_reservation_action(AuthenticationError("401 Unauthorized")) == "release"
    assert classify_reservation_action(ConnectionError("Failed to resolve 'api.groq.com': getaddrinfo failed")) == "release"
    assert classify_reservation_action(ConnectionError("Connection refused by peer")) == "release"

    # Retain cases:
    assert classify_reservation_action(ProviderError("Error code: 500 - Internal Server Error")) == "retain"
    assert classify_reservation_action(ProviderError("Error code: 502 - Bad Gateway")) == "retain"
    assert classify_reservation_action(ProviderError("Error code: 503 - Service Unavailable")) == "retain"
    assert classify_reservation_action(TimeoutError("Groq API request timed out.")) == "retain"
    assert classify_reservation_action(ConnectionError("Connection reset by peer")) == "retain"
    assert classify_reservation_action(RuntimeError("Unexpected unclassified failure")) == "retain"


def test_reservation_release_on_http_400_schema_error():
    """Confirmed HTTP 400 schema error releases reservation immediately."""
    limiter = SequentialRateLimiter(window_seconds=60.0, itpm_limit=7000, reserve_tokens=1000)
    mock_provider = MagicMock()
    mock_provider.complete.side_effect = ProviderError(
        "Groq API provider error (BadRequestError): Error code: 400 - tool call validation failed: parameters for tool inspect_recipe did not match schema"
    )

    guardrail = GroqPilotRequestGuardrailProvider(provider=mock_provider, max_requests=5, rate_limiter=limiter)
    msg = [LLMMessage(role="user", content="Ping")]

    with pytest.raises(ProviderError):
        guardrail.complete(msg, estimated_tokens=2500)

    # Reservation released
    assert limiter.get_window_tokens() == 0
    assert len(limiter.window_entries) == 0
    # Request counter incremented
    assert guardrail.total_requests == 1
    # Unconfirmed tokens never in usage totals
    assert guardrail.cumulative_input_tokens == 0


def test_reservation_release_on_http_429_quota_rejection():
    """HTTP 429 quota error releases reservation and updates quota reset deadline."""
    current_time = [100.0]

    def mock_clock() -> float:
        return current_time[0]

    def mock_sleep(seconds: float) -> None:
        current_time[0] += seconds

    limiter = SequentialRateLimiter(
        window_seconds=60.0,
        itpm_limit=7000,
        reserve_tokens=1000,
        clock=mock_clock,
        sleeper=mock_sleep,
    )
    mock_provider = MagicMock()

    def raise_429(*args, **kwargs):
        mock_provider.last_response_headers = {
            "x-ratelimit-remaining-tokens": "100",
            "x-ratelimit-reset-tokens": "14.5s",
        }
        raise RateLimitError("Groq API rate limit or quota exceeded.")

    mock_provider.complete.side_effect = raise_429
    mock_provider.last_response_headers = {}

    guardrail = GroqPilotRequestGuardrailProvider(provider=mock_provider, max_requests=5, rate_limiter=limiter)
    msg = [LLMMessage(role="user", content="Ping")]

    with pytest.raises(RateLimitError):
        guardrail.complete(msg, estimated_tokens=2500)

    # Reservation released
    assert limiter.get_window_tokens() == 0
    # Quota deadline preserved: 100.0 + 14.5 = 114.5
    assert limiter.quota_deadline == 114.5
    assert guardrail.total_requests == 1
    assert guardrail.cumulative_input_tokens == 0


def test_reservation_retained_on_http_500_server_error():
    """HTTP 500 server-side error conservatively retains reservation as unconfirmed_failed."""
    limiter = SequentialRateLimiter(window_seconds=60.0, itpm_limit=7000, reserve_tokens=1000)
    mock_provider = MagicMock()
    mock_provider.complete.side_effect = ProviderError("Groq API provider error (InternalServerError): Error code: 500 - internal server error")

    guardrail = GroqPilotRequestGuardrailProvider(provider=mock_provider, max_requests=5, rate_limiter=limiter)
    msg = [LLMMessage(role="user", content="Ping")]

    with pytest.raises(ProviderError):
        guardrail.complete(msg, estimated_tokens=2500)

    # Reservation conservatively retained
    assert limiter.get_window_tokens() == 2500
    assert len(limiter.window_entries) == 1
    assert limiter.window_entries[0]["status"] == "unconfirmed_failed"
    assert limiter.window_entries[0]["tokens"] == 2500
    # Request counter incremented
    assert guardrail.total_requests == 1
    # Unconfirmed estimates NEVER added to confirmed usage totals
    assert guardrail.cumulative_input_tokens == 0


def test_reservation_retained_on_timeout_error():
    """Post-dispatch TimeoutError conservatively retains reservation as unconfirmed_failed."""
    limiter = SequentialRateLimiter(window_seconds=60.0, itpm_limit=7000, reserve_tokens=1000)
    mock_provider = MagicMock()
    mock_provider.complete.side_effect = TimeoutError("Groq API request timed out.")

    guardrail = GroqPilotRequestGuardrailProvider(provider=mock_provider, max_requests=5, rate_limiter=limiter)
    msg = [LLMMessage(role="user", content="Ping")]

    with pytest.raises(TimeoutError):
        guardrail.complete(msg, estimated_tokens=2800)

    # Retained
    assert limiter.get_window_tokens() == 2800
    assert limiter.window_entries[0]["status"] == "unconfirmed_failed"
    assert guardrail.total_requests == 1
    assert guardrail.cumulative_input_tokens == 0


def test_reservation_retained_on_post_dispatch_connection_drop():
    """Ambiguous post-dispatch ConnectionError conservatively retains reservation as unconfirmed_failed."""
    limiter = SequentialRateLimiter(window_seconds=60.0, itpm_limit=7000, reserve_tokens=1000)
    mock_provider = MagicMock()
    mock_provider.complete.side_effect = ConnectionError("Connection reset by peer during read")

    guardrail = GroqPilotRequestGuardrailProvider(provider=mock_provider, max_requests=5, rate_limiter=limiter)
    msg = [LLMMessage(role="user", content="Ping")]

    with pytest.raises(ConnectionError):
        guardrail.complete(msg, estimated_tokens=2800)

    # Retained
    assert limiter.get_window_tokens() == 2800
    assert limiter.window_entries[0]["status"] == "unconfirmed_failed"
    assert guardrail.total_requests == 1
    assert guardrail.cumulative_input_tokens == 0


def test_reservation_released_on_pre_dispatch_connection_failure():
    """Confirmed pre-dispatch connection error (DNS failure / connection refused) releases reservation."""
    limiter = SequentialRateLimiter(window_seconds=60.0, itpm_limit=7000, reserve_tokens=1000)
    mock_provider = MagicMock()
    mock_provider.complete.side_effect = ConnectionError("Failed to establish a new connection: [Errno 11001] getaddrinfo failed")

    guardrail = GroqPilotRequestGuardrailProvider(provider=mock_provider, max_requests=5, rate_limiter=limiter)
    msg = [LLMMessage(role="user", content="Ping")]

    with pytest.raises(ConnectionError):
        guardrail.complete(msg, estimated_tokens=2800)

    # Released
    assert limiter.get_window_tokens() == 0
    assert len(limiter.window_entries) == 0
    assert guardrail.total_requests == 1
    assert guardrail.cumulative_input_tokens == 0


def test_retained_failed_reservation_expires_after_60_seconds():
    """Retained unconfirmed_failed reservation expires cleanly after 60 seconds."""
    current_time = [100.0]

    def mock_clock() -> float:
        return current_time[0]

    def mock_sleep(seconds: float) -> None:
        current_time[0] += seconds

    limiter = SequentialRateLimiter(
        window_seconds=60.0,
        itpm_limit=7000,
        reserve_tokens=1000,
        clock=mock_clock,
        sleeper=mock_sleep,
    )
    mock_provider = MagicMock()
    mock_provider.complete.side_effect = TimeoutError("Groq API request timed out.")

    guardrail = GroqPilotRequestGuardrailProvider(provider=mock_provider, max_requests=5, rate_limiter=limiter)
    msg = [LLMMessage(role="user", content="Ping")]

    with pytest.raises(TimeoutError):
        guardrail.complete(msg, estimated_tokens=2800)

    assert limiter.get_window_tokens() == 2800
    assert limiter.window_entries[0]["status"] == "unconfirmed_failed"

    # Advance time by 30 seconds to T=130.0 (still in window)
    current_time[0] = 130.0
    assert limiter.get_window_tokens() == 2800

    # Advance time past 60s window to T=161.0
    current_time[0] = 161.0
    assert limiter.get_window_tokens() == 0
    assert len(limiter.window_entries) == 0


def test_cross_task_persistence_of_retained_reservation():
    """Retained unconfirmed_failed reservation from Task 1 paces Task 2 until window expires."""
    current_time = [100.0]
    sleeps: list[float] = []

    def mock_clock() -> float:
        return current_time[0]

    def mock_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        current_time[0] += seconds

    limiter = SequentialRateLimiter(
        min_interval_seconds=5.0,
        window_seconds=60.0,
        itpm_limit=7000,
        reserve_tokens=1000,  # Effective budget: 6,000 tokens
        clock=mock_clock,
        sleeper=mock_sleep,
    )

    mock_provider = MagicMock()
    guardrail = GroqPilotRequestGuardrailProvider(provider=mock_provider, max_requests=5, rate_limiter=limiter)
    msg = [LLMMessage(role="user", content="Task 1 Prompt")]

    # Task 1 at T=100.0 fails with HTTP 503 Service Unavailable, retaining 3,500 tokens
    mock_provider.complete.side_effect = ProviderError("Groq API provider error (ServiceUnavailableError): Error code: 503")
    with pytest.raises(ProviderError):
        guardrail.complete(msg, estimated_tokens=3500)

    assert limiter.get_window_tokens() == 3500
    assert limiter.window_entries[0]["status"] == "unconfirmed_failed"
    assert guardrail.total_requests == 1
    assert guardrail.cumulative_input_tokens == 0

    # Runner transitions to Task 2 at T=120.0 (20s later)
    current_time[0] = 120.0

    # Task 2 wants to dispatch 3,200 tokens
    # 3500 (retained) + 3200 = 6700 > 6000. Limiter must wait until Task 1 entry expires at 100.0 + 60.0 = 160.0 (40.05s wait)
    mock_provider.complete.side_effect = None
    mock_provider.complete.return_value = LLMCompletionResult(
        message=LLMMessage(role="assistant", content="Task 2 OK"),
        usage=LLMUsage(input_tokens=3200, output_tokens=50, total_tokens=3250, is_provider_reported=True),
        model_identifier="qwen/qwen3.8-27b",
        provider_request_id="task2_req1",
        latency_ms=250.0,
        finish_reason="stop",
    )

    res2 = guardrail.complete(msg, estimated_tokens=3200)
    assert len(sleeps) >= 1
    assert sleeps[-1] >= 40.0
    assert current_time[0] >= 160.0
    assert res2.usage.input_tokens == 3200
    # Task 1's unconfirmed reservation expired; only Task 2's confirmed usage remains
    assert limiter.get_window_tokens() == 3200
    assert guardrail.total_requests == 2
    # Confirmed token usage strictly contains only Task 2's usage, NOT Task 1's unconfirmed tokens!
    assert guardrail.cumulative_input_tokens == 3200


# =============================================================================
# 12. CLI ENTRYPOINT & STARTUP SANITY TESTS
# =============================================================================

def test_groq_pilot_cli_startup_clean_no_runpy_warning():
    """CLI execution via both -m module and direct script path must not produce runpy RuntimeWarning."""
    import subprocess
    import sys

    # 1. Test direct script path invocation
    script_res = subprocess.run(
        [sys.executable, "src/shopconnect_env/evaluation/groq_pilot.py", "--preflight"],
        capture_output=True,
        text=True,
    )
    assert script_res.returncode == 0
    assert "RuntimeWarning" not in script_res.stderr
    assert "status" in script_res.stdout

    # 2. Test python -m shopconnect_env.evaluation.groq_pilot module invocation
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    module_res = subprocess.run(
        [sys.executable, "-m", "shopconnect_env.evaluation.groq_pilot", "--preflight"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert module_res.returncode == 0
    assert "RuntimeWarning" not in module_res.stderr
    assert "status" in module_res.stdout


# =============================================================================
# 13. RELIABILITY, PACING TELEMETRY & INCREMENTAL PERSISTENCE TESTS
# =============================================================================

def test_pacing_feedback_messages():
    """SequentialRateLimiter must output human-readable pacing feedback for all three wait reasons."""
    messages = []
    current_time = [100.0]

    def mock_clock() -> float:
        return current_time[0]

    def mock_sleep(seconds: float) -> None:
        current_time[0] += seconds

    limiter = SequentialRateLimiter(
        min_interval_seconds=12.0,
        clock=mock_clock,
        sleeper=mock_sleep,
        printer=messages.append,
        window_seconds=60.0,
        itpm_limit=7000,
        reserve_tokens=1000,
    )

    # 1. First request dispatches immediately without pacing message
    limiter.wait_for_slot(estimated_tokens=500)
    assert len(messages) == 0

    # Advance clock by 2 seconds (elapsed=2s < 12s min_interval)
    current_time[0] = 102.0
    limiter.wait_for_slot(estimated_tokens=500)
    assert len(messages) == 1
    assert "[Pacing] Waiting 10.0s (reason: minimum interval)..." in messages[0]
    assert "gsk-" not in messages[0]
    assert limiter.sleep_records[-1]["reason"] == "min_interval"
    assert limiter.sleep_records[-1]["reason_display"] == "minimum interval"

    # 2. Test quota reset pacing message
    headers = {
        "x-ratelimit-remaining-tokens": "200",
        "x-ratelimit-reset-tokens": "25.0s",
    }
    limiter.update_from_headers(headers, estimated_next_tokens=1000, receipt_time=current_time[0])
    limiter.wait_for_slot(estimated_tokens=1000)
    assert len(messages) == 2
    assert "(reason: provider quota reset)" in messages[1]
    assert "gsk-" not in messages[1]
    assert limiter.sleep_records[-1]["reason"] == "quota_deadline"
    assert limiter.sleep_records[-1]["reason_display"] == "provider quota reset"

    # 3. Test sliding-window budget pacing message
    limiter.window_entries.clear()
    limiter.last_request_start = current_time[0] - 20.0
    limiter.reserve_slot(5500, now=current_time[0] - 10.0)
    limiter.wait_for_slot(estimated_tokens=1000)  # 5500 + 1000 = 6500 > 6000
    assert len(messages) == 3
    assert "(reason: sliding-window budget)" in messages[2]
    assert "gsk-" not in messages[2]
    assert limiter.sleep_records[-1]["reason"] == "sliding_window_budget"
    assert limiter.sleep_records[-1]["reason_display"] == "sliding-window budget"


def test_request_level_telemetry_persistence(tmp_path):
    """GroqPilotRequestGuardrailProvider must append telemetry to request_telemetry.jsonl on success and error."""
    telemetry_file = tmp_path / "request_telemetry.jsonl"
    mock_provider = MagicMock()
    mock_provider.last_response_headers = {}

    guardrail = GroqPilotRequestGuardrailProvider(
        provider=mock_provider,
        max_requests=5,
        telemetry_path=telemetry_file,
    )

    # 1. Successful request
    mock_provider.complete.return_value = LLMCompletionResult(
        message=LLMMessage(role="assistant", content="hello"),
        usage=LLMUsage(input_tokens=100, output_tokens=20, total_tokens=120, is_provider_reported=True),
        provider_request_id="req_ok_123",
        model_identifier="test-model",
    )
    guardrail.complete([LLMMessage(role="user", content="hi")])

    assert telemetry_file.exists()
    lines = [json.loads(line) for line in telemetry_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 1
    assert lines[0]["request_index"] == 1
    assert lines[0]["status"] == "success"
    assert lines[0]["provider_request_id"] == "req_ok_123"
    assert lines[0]["usage"]["input_tokens"] == 100
    assert "timestamp_utc" in lines[0]

    # 2. Failing request
    mock_provider.complete.side_effect = RuntimeError("Rate limit exceeded")
    with pytest.raises(RuntimeError):
        guardrail.complete([LLMMessage(role="user", content="hi again")])

    lines = [json.loads(line) for line in telemetry_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 2
    assert lines[1]["request_index"] == 2
    assert lines[1]["status"] == "error"
    assert lines[1]["error"] == "RuntimeError"
    assert "duration_ms" in lines[1]
    assert "timestamp_utc" in lines[1]


def test_pre_creation_of_directory_and_metadata(tmp_path):
    """Dry run and live harness must create target_dir and experiment_meta.json before completion."""
    res = run_groq_pilot_dry_run(output_dir=tmp_path)
    assert res["status"] == "completed"

    meta_file = tmp_path / "experiment_meta.json"
    telemetry_file = tmp_path / "request_telemetry.jsonl"
    raw_runs_file = tmp_path / "raw_runs.jsonl"

    assert meta_file.exists()
    assert telemetry_file.exists()
    assert raw_runs_file.exists()

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["status"] == "completed"
    assert meta["completed_runs"] == 9
    assert meta["planned_runs"] == 9
    assert "created_at_utc" in meta
    assert "finished_at_utc" in meta


def test_incremental_persistence_of_run_records(tmp_path, monkeypatch):
    """RunRecords must be appended incrementally to raw_runs.jsonl after each task completes."""
    from shopconnect_env.evaluation.runner import ExperimentRunner

    original_run_task = ExperimentRunner.run_task

    def instrumented_run_task(self, task, condition_id, provider=None):
        rec = original_run_task(self, task, condition_id, provider=provider)
        return rec

    monkeypatch.setattr(ExperimentRunner, "run_task", instrumented_run_task)

    res = run_groq_pilot_dry_run(output_dir=tmp_path)
    assert res["status"] == "completed"
    lines = (tmp_path / "raw_runs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 9


def test_graceful_interruption_flushes_artifacts(tmp_path, monkeypatch):
    """KeyboardInterrupt must flush partial RunRecords, update metadata to interrupted, and write partial analysis."""
    from shopconnect_env.evaluation.runner import ExperimentRunner

    call_count = [0]
    original_run_task = ExperimentRunner.run_task

    def interrupt_after_two_runs(self, task, condition_id, provider=None):
        call_count[0] += 1
        if call_count[0] > 2:
            raise KeyboardInterrupt("Simulated Ctrl+C")
        return original_run_task(self, task, condition_id, provider=provider)

    monkeypatch.setattr(ExperimentRunner, "run_task", interrupt_after_two_runs)

    res = run_groq_pilot_dry_run(output_dir=tmp_path)

    # 1. Returns interrupted status
    assert res["status"] == "interrupted"
    assert "Interrupted by user" in res["incomplete_reason"]
    assert res["runs_completed"] == 2

    # 2. Metadata file reflects interrupted state
    meta_file = tmp_path / "experiment_meta.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["status"] == "interrupted"
    assert meta["completed_runs"] == 2
    assert "Interrupted by user" in meta["incomplete_reason"]

    # 3. raw_runs.jsonl has exactly the 2 completed records
    raw_runs = tmp_path / "raw_runs.jsonl"
    assert raw_runs.exists()
    lines = raw_runs.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    # 4. comparison.csv exists and has 2 data rows
    comparison = tmp_path / "comparison.csv"
    assert comparison.exists()
    csv_lines = comparison.read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) == 3  # header + 2 rows

    # 5. analysis.md exists and contains the INCOMPLETE PILOT warning banner
    analysis = tmp_path / "analysis.md"
    assert analysis.exists()
    content = analysis.read_text(encoding="utf-8")
    assert "INCOMPLETE PILOT" in content
    assert "Interrupted by user" in content


def test_live_groq_pilot_graceful_interruption(tmp_path, monkeypatch):
    """run_live_groq_pilot must handle KeyboardInterrupt gracefully and flush partial artifacts."""
    from shopconnect_env.evaluation.runner import ExperimentRunner

    # Mock preflight to pass
    monkeypatch.setattr(
        "shopconnect_env.evaluation.groq_pilot.run_groq_pilot_preflight",
        lambda **kwargs: {"status": "passed", "api_key_configured": True, "errors": []},
    )

    call_count = [0]
    original_run_task = ExperimentRunner.run_task

    def interrupt_after_one_run(self, task, condition_id, provider=None):
        call_count[0] += 1
        if call_count[0] > 1:
            raise KeyboardInterrupt("Simulated Ctrl+C during live pilot")
        return original_run_task(self, task, condition_id, provider=provider)

    monkeypatch.setattr(ExperimentRunner, "run_task", interrupt_after_one_run)

    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="chatcmpl-live-test",
        model="qwen/qwen3.8-27b",
        system_fingerprint="fp_test",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="No-op", tool_calls=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    res = run_live_groq_pilot(
        api_key="gsk-test-key",
        client=mock_client,
        output_dir=tmp_path,
        live=True,
        confirm_live=True,
        conditions=("B", "C"),
    )

    assert res["status"] == "interrupted"
    assert "Interrupted by user" in res["incomplete_reason"]
    assert res["runs_completed"] == 1

    meta_file = tmp_path / "experiment_meta.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["status"] == "interrupted"
    assert meta["completed_runs"] == 1

    raw_runs = tmp_path / "raw_runs.jsonl"
    assert raw_runs.exists()
    lines = raw_runs.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    analysis = tmp_path / "analysis.md"
    assert analysis.exists()
    assert "INCOMPLETE PILOT" in analysis.read_text(encoding="utf-8")


def test_interrupted_in_flight_request_telemetry(tmp_path):
    """In-flight KeyboardInterrupt during provider.complete must log outcome_unknown and retain reservation."""
    telemetry_file = tmp_path / "request_telemetry.jsonl"
    mock_provider = MagicMock()
    mock_provider.last_response_headers = {}
    mock_provider.complete.side_effect = KeyboardInterrupt("Ctrl+C during in-flight network dispatch")

    current_time = [100.0]
    limiter = SequentialRateLimiter(
        clock=lambda: current_time[0],
        sleeper=lambda s: None,
        verbose=False,
    )
    guardrail = GroqPilotRequestGuardrailProvider(
        provider=mock_provider,
        max_requests=5,
        rate_limiter=limiter,
        telemetry_path=telemetry_file,
    )

    with pytest.raises(KeyboardInterrupt):
        guardrail.complete([LLMMessage(role="user", content="hello in-flight")])

    # 1. Total requests incremented because dispatch was attempted
    assert guardrail.total_requests == 1

    # 2. Cumulative tokens remain 0 (no claim that provider consumed tokens)
    assert guardrail.cumulative_input_tokens == 0
    assert guardrail.cumulative_output_tokens == 0

    # 3. Reservation is retained as unconfirmed_failed
    assert len(limiter.window_entries) == 1
    assert limiter.window_entries[0]["status"] == "unconfirmed_failed"

    # 4. Telemetry record is written with outcome_unknown and duration
    assert telemetry_file.exists()
    lines = [json.loads(line) for line in telemetry_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["request_index"] == 1
    assert rec["status"] == "outcome_unknown"
    assert rec["error"] == "KeyboardInterrupt"
    assert rec["interruption_type"] == "KeyboardInterrupt"
    assert rec["usage"] is None
    assert rec["duration_ms"] >= 0.0
    assert "timestamp_utc" in rec


def test_interrupted_pacing_wait_does_not_record_http_attempt(tmp_path):
    """KeyboardInterrupt during pacing sleep must not increment request counter or log an HTTP attempt."""
    telemetry_file = tmp_path / "request_telemetry.jsonl"
    mock_provider = MagicMock()

    mock_limiter = MagicMock(spec=SequentialRateLimiter)
    mock_limiter.wait_for_slot.side_effect = KeyboardInterrupt("Ctrl+C during pacing sleep")

    guardrail = GroqPilotRequestGuardrailProvider(
        provider=mock_provider,
        max_requests=5,
        rate_limiter=mock_limiter,
        telemetry_path=telemetry_file,
    )

    with pytest.raises(KeyboardInterrupt):
        guardrail.complete([LLMMessage(role="user", content="hello pacing")])

    # Dispatch never occurred: total_requests is 0
    assert guardrail.total_requests == 0
    assert not mock_provider.complete.called

    # No HTTP telemetry record created
    assert not telemetry_file.exists()
    assert len(guardrail.request_history) == 0


def test_telemetry_write_failure_emits_warning_and_marks_metadata(tmp_path, capsys):
    """Telemetry write failure must not crash complete(), must emit warning to stderr, and mark metadata incomplete."""
    invalid_telemetry_dir = tmp_path / "telemetry_is_a_dir"
    invalid_telemetry_dir.mkdir()

    mock_provider = MagicMock()
    mock_provider.complete.return_value = LLMCompletionResult(
        message=LLMMessage(role="assistant", content="ok"),
        usage=LLMUsage(input_tokens=50, output_tokens=10, total_tokens=60, is_provider_reported=True),
        provider_request_id="req_write_fail",
        model_identifier="test-model",
    )

    guardrail = GroqPilotRequestGuardrailProvider(
        provider=mock_provider,
        max_requests=5,
        telemetry_path=invalid_telemetry_dir,
    )

    # complete() succeeds despite telemetry write error
    res = guardrail.complete([LLMMessage(role="user", content="hi")])
    assert res.message.content == "ok"

    # Warning was emitted to stderr without prompt/key exposure
    captured = capsys.readouterr()
    assert "[WARNING] Telemetry write failed for request 1" in captured.err
    assert "gsk-" not in captured.err
    assert "hello" not in captured.err

    # Error tracked in guardrail
    assert guardrail.telemetry_write_errors == 1
    assert len(guardrail.telemetry_error_messages) == 1


def test_experiment_metadata_reflects_telemetry_write_failures(tmp_path, monkeypatch):
    """experiment_meta.json must record telemetry_complete=False if any telemetry write fails."""
    def failing_log(self, entry):
        self.telemetry_write_errors += 1
        self.telemetry_error_messages.append("MockPermissionError: Permission denied")

    monkeypatch.setattr(GroqPilotRequestGuardrailProvider, "_log_request_telemetry", failing_log)

    res = run_groq_pilot_dry_run(output_dir=tmp_path)
    assert res["status"] == "completed"
    assert res["telemetry_complete"] is False
    assert res["telemetry_write_errors"] > 0

    meta = json.loads((tmp_path / "experiment_meta.json").read_text(encoding="utf-8"))
    assert meta["telemetry_complete"] is False
    assert meta["telemetry_write_errors"] > 0
    assert "telemetry_error_messages" in meta


# =============================================================================
# 18. DYNAMIC REPORTING & SAMPLE SIZE DISCLAIMER REGRESSION TESTS
# =============================================================================

def test_format_conditions_label_formatting():
    """format_conditions_label formats empty, single, double, and triple condition lists."""
    assert format_conditions_label([]) == "No Conditions"
    assert format_conditions_label(["A"]) == "Condition A"
    assert format_conditions_label(["B"]) == "Condition B"
    assert format_conditions_label(["C"]) == "Condition C"
    assert format_conditions_label(["B", "C"]) == "Conditions B and C"
    assert format_conditions_label(["A", "C"]) == "Conditions A and C"
    assert format_conditions_label(["A", "B", "C"]) == "Conditions A, B, and C"


def test_format_pilot_sample_size_disclaimer_unit():
    """format_pilot_sample_size_disclaimer generates accurate sample-size notices."""
    # B/C completed (6 runs)
    disc_bc = format_pilot_sample_size_disclaimer(
        conditions=["B", "C"],
        tasks_count=3,
        planned_runs=6,
        completed_runs=6,
    )
    assert "consisting of 3 tasks across Conditions B and C (6 runs)." in disc_bc
    assert "Conditions A, B, and C" not in disc_bc
    assert "INCOMPLETE" not in disc_bc

    # A/B/C completed (9 runs)
    disc_abc = format_pilot_sample_size_disclaimer(
        conditions=["A", "B", "C"],
        tasks_count=3,
        planned_runs=9,
        completed_runs=9,
    )
    assert "consisting of 3 tasks across Conditions A, B, and C (9 runs)." in disc_abc

    # Incomplete runs with ceiling reason
    disc_inc = format_pilot_sample_size_disclaimer(
        conditions=["B", "C"],
        tasks_count=3,
        planned_runs=6,
        completed_runs=2,
        incomplete_reason="Request ceiling reached",
    )
    assert "consisting of 3 tasks across Conditions B and C (2/6 runs completed)." in disc_inc

    # Incomplete runs with interruption
    disc_int = format_pilot_sample_size_disclaimer(
        conditions=["A", "B", "C"],
        tasks_count=3,
        planned_runs=9,
        completed_runs=4,
        incomplete_reason="Interrupted by user (KeyboardInterrupt / SIGINT).",
    )
    assert "consisting of 3 tasks across Conditions A, B, and C (4/9 runs completed)." in disc_int

    # Single task, single condition
    disc_single = format_pilot_sample_size_disclaimer(
        conditions=["B"],
        tasks_count=1,
        planned_runs=1,
        completed_runs=1,
    )
    assert "consisting of 1 task across Condition B (1 run)." in disc_single

    # Default fallback invocation
    disc_default = format_pilot_sample_size_disclaimer()
    assert "consisting of 3 tasks across Conditions A, B, and C (9 runs)." in disc_default


def test_groq_pilot_dry_run_bc_conditions_dynamic_reporting(tmp_path):
    """Dry run with conditions B and C must generate dynamic B/C title and 6-run disclaimer."""
    res = run_groq_pilot_dry_run(output_dir=tmp_path, conditions=["B", "C"])
    assert res["status"] == "completed"
    assert res["runs_completed"] == 6
    assert res["runs_planned"] == 6

    analysis_path = tmp_path / "analysis.md"
    assert analysis_path.exists()
    content = analysis_path.read_text(encoding="utf-8")

    # Dynamic comparative report title
    assert "# Comparative Evaluation Report: Conditions B and C" in content

    # Dynamic sample size notice
    assert "LIMITED SAMPLE SIZE NOTICE" in content
    assert "consisting of 3 tasks across Conditions B and C (6 runs)." in content

    # Must NOT claim A was evaluated or 9 runs
    assert "consisting of 3 tasks across Conditions A, B, and C" not in content
    assert "(9 runs)" not in content
    assert "INCOMPLETE PILOT" not in content


def test_groq_pilot_dry_run_abc_conditions_dynamic_reporting(tmp_path):
    """Dry run with conditions A, B, and C must generate dynamic A/B/C title and 9-run disclaimer."""
    res = run_groq_pilot_dry_run(output_dir=tmp_path, conditions=["A", "B", "C"])
    assert res["status"] == "completed"
    assert res["runs_completed"] == 9
    assert res["runs_planned"] == 9

    analysis_path = tmp_path / "analysis.md"
    assert analysis_path.exists()
    content = analysis_path.read_text(encoding="utf-8")

    # Dynamic comparative report title
    assert "# Comparative Evaluation Report: Conditions A, B, and C" in content

    # Dynamic sample size notice
    assert "LIMITED SAMPLE SIZE NOTICE" in content
    assert "consisting of 3 tasks across Conditions A, B, and C (9 runs)." in content
    assert "INCOMPLETE PILOT" not in content


def test_groq_pilot_dry_run_incomplete_dynamic_reporting(tmp_path):
    """Incomplete dry run must reflect completed vs planned runs in disclaimer banner."""
    res = run_groq_pilot_dry_run(output_dir=tmp_path, conditions=["B", "C"], max_requests=3)
    assert res["status"] == "incomplete"
    assert res["ceiling_reached"] is True
    assert res["runs_completed"] < 6

    analysis_path = tmp_path / "analysis.md"
    assert analysis_path.exists()
    content = analysis_path.read_text(encoding="utf-8")

    # Dynamic sample size notice reflects completed/planned
    assert "LIMITED SAMPLE SIZE NOTICE" in content
    assert f"consisting of 3 tasks across Conditions B and C ({res['runs_completed']}/6 runs completed)." in content
    assert "INCOMPLETE PILOT" in content
