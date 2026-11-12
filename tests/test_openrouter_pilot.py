"""Unit tests for OpenRouter pilot runner, guardrails, dry run, and connection test.

Verifies:
- Dual confirmation flags required for live execution (--live AND --confirm-live).
- Standalone connection test dual confirmation (--connection-test AND --confirm-connection-test).
- Standalone connection test prompt and token validation.
- Connection test preserves model_requested vs actual_model_identifier (None when omitted).
- Zero network activity during preflight and dry-run execution.
- Strict 20-request ceiling enforcement.
- Sequential rate limiting (20s minimum interval, 3 RPM target) with mock clock/sleeper.
- Dispatched failed attempts counted against request ceiling.
- Simulated 9-run dry-run execution across conditions A, B, and C.
- Condition C max 4 model requests budget.
- Artifact generation and directory isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from shopconnect_env.agents.providers import (
    AuthenticationError,
    LLMCompletionResult,
    LLMMessage,
    LLMUsage,
    MockLLMProvider,
    OpenRouterProvider,
    RateLimitError,
)
from shopconnect_env.evaluation.openrouter_pilot import (
    CONNECTION_TEST_PROMPT,
    EXPECTED_CONNECTION_TEST_TOKEN,
    MAX_PILOT_RUNS,
    MAX_REQUESTS_PER_CONDITION_C,
    MAX_THEORETICAL_TOTAL_REQUESTS,
    MAX_TOTAL_PROVIDER_REQUESTS,
    PILOT_MODEL,
    OpenRouterPilotRequestGuardrailProvider,
    SequentialRateLimiter,
    create_openrouter_pilot_mock_provider,
    run_live_openrouter_pilot,
    run_openrouter_connection_test,
    run_openrouter_pilot_dry_run,
    run_openrouter_pilot_preflight,
)


# =============================================================================
# 1. PREFLIGHT TESTS (ZERO NETWORK ACTIVITY)
# =============================================================================

def test_openrouter_preflight_missing_api_key(monkeypatch):
    """Preflight must report api_key_configured=False when variable is missing."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    res = run_openrouter_pilot_preflight()
    assert res["status"] == "passed"  # Data files pass, only key is reported false
    assert res["api_key_configured"] is False
    assert res["planned_runs_count"] == 9
    assert res["max_theoretical_requests"] == 18
    assert res["global_request_ceiling"] == 20


def test_openrouter_preflight_with_key(monkeypatch):
    """Preflight passes completely when data files and key are present."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    res = run_openrouter_pilot_preflight()
    assert res["status"] == "passed"
    assert res["api_key_configured"] is True
    assert res["tasks_loaded"] == 3
    assert res["pricing_complete"] is True
    assert res["billing_tier"] == "free"


# =============================================================================
# 2. GUARDRAIL & RATE LIMITER TESTS
# =============================================================================

def test_openrouter_guardrail_ceiling_enforcement():
    """Guardrail must raise RuntimeError when ceiling is reached."""
    mock_provider = MockLLMProvider()
    guardrail = OpenRouterPilotRequestGuardrailProvider(
        provider=mock_provider,
        max_requests=3,
    )

    msg = [LLMMessage(role="user", content="Ping")]
    guardrail.complete(msg)
    guardrail.complete(msg)
    guardrail.complete(msg)
    assert guardrail.total_requests == 3

    with pytest.raises(RuntimeError, match="OpenRouter pilot execution safety ceiling reached"):
        guardrail.complete(msg)


def test_openrouter_guardrail_counts_failed_attempts():
    """Failed provider invocations must be counted against total_requests ceiling."""
    mock_provider = MagicMock()
    mock_provider.complete.side_effect = RateLimitError("Quota reached")

    guardrail = OpenRouterPilotRequestGuardrailProvider(
        provider=mock_provider,
        max_requests=2,
    )

    msg = [LLMMessage(role="user", content="Ping")]
    with pytest.raises(RateLimitError):
        guardrail.complete(msg)

    assert guardrail.total_requests == 1


def test_openrouter_sequential_rate_limiter():
    """Rate limiter must enforce minimum 20s interval between requests."""
    fake_time = 1000.0
    sleeps: list[float] = []

    def mock_clock() -> float:
        return fake_time

    def mock_sleeper(seconds: float) -> None:
        nonlocal fake_time
        sleeps.append(seconds)
        fake_time += seconds

    limiter = SequentialRateLimiter(
        min_interval_seconds=20.0,
        clock=mock_clock,
        sleeper=mock_sleeper,
    )

    # First request: no prior request -> 0s sleep
    s1 = limiter.wait_for_slot()
    assert s1 == 0.0

    # Simulate 5s elapsed
    fake_time += 5.0
    s2 = limiter.wait_for_slot()
    assert s2 == 15.0  # 20.0 - 5.0 = 15.0s sleep

    # Simulate 25s elapsed (more than 20s) -> 0s sleep
    fake_time += 25.0
    s3 = limiter.wait_for_slot()
    assert s3 == 0.0
    assert len(sleeps) == 1
    assert sleeps[0] == 15.0


# =============================================================================
# 3. DRY RUN TESTS
# =============================================================================

def test_openrouter_pilot_dry_run_execution(tmp_path):
    """Dry run must complete 9 runs and generate isolated artifacts with zero network activity."""
    res = run_openrouter_pilot_dry_run(output_dir=tmp_path)
    assert res["status"] == "completed"
    assert res["runs_completed"] == 9
    assert res["runs_planned"] == 9
    assert res["network_calls_made"] == 0
    assert res["total_provider_requests"] <= MAX_THEORETICAL_TOTAL_REQUESTS

    raw_runs = tmp_path / "raw_runs.jsonl"
    comparison = tmp_path / "comparison.csv"
    analysis = tmp_path / "analysis.md"

    assert raw_runs.exists()
    assert comparison.exists()
    assert analysis.exists()

    lines = raw_runs.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 9

    # Check Condition C budget
    assert res["max_requests_per_condition_c"] == 4
    assert res["max_theoretical_requests"] == 18


# =============================================================================
# 4. CONNECTION TEST TESTS
# =============================================================================

def test_openrouter_connection_test_requires_dual_confirmation():
    """Connection test must reject execution if either flag is missing."""
    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_openrouter_connection_test(connection_test=True, confirm_connection_test=False)

    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_openrouter_connection_test(connection_test=False, confirm_connection_test=True)


def test_openrouter_connection_test_missing_key(monkeypatch, tmp_path):
    """Connection test records clean failure artifact when API key is missing."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    res = run_openrouter_connection_test(
        output_dir=tmp_path,
        connection_test=True,
        confirm_connection_test=True,
    )
    assert res["status"] == "failed"
    assert res["success"] is False
    assert res["error_type"] == "MissingAPIKeyError"
    assert res["requests_made"] == 0
    assert (tmp_path / "connection_test_result.json").exists()


def test_openrouter_connection_test_success_with_injected_client(tmp_path):
    """Connection test succeeds when expected token is returned by mock client."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="gen-conn-123",
        model="qwen/qwen3.8-27b:free",
        system_fingerprint="fp_openrouter",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content=f"Sure! {EXPECTED_CONNECTION_TEST_TOKEN}",
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

    res = run_openrouter_connection_test(
        api_key="test-key",
        output_dir=tmp_path,
        connection_test=True,
        confirm_connection_test=True,
        client=mock_client,
    )

    assert res["status"] == "passed"
    assert res["success"] is True
    assert res["requests_made"] == 1
    assert res["actual_model_identifier"] == "qwen/qwen3.8-27b:free"
    assert res["model_requested"] == PILOT_MODEL
    assert (tmp_path / "connection_test_result.json").exists()


def test_openrouter_connection_test_missing_actual_model_identifier(tmp_path):
    """When provider omits model string, actual_model_identifier is preserved as None."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="gen-conn-no-model",
        model=None,
        system_fingerprint=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content=EXPECTED_CONNECTION_TEST_TOKEN,
                    tool_calls=None,
                ),
            )
        ],
        usage=None,
    )
    mock_client.chat.completions.create.return_value = mock_resp

    res = run_openrouter_connection_test(
        api_key="test-key",
        output_dir=tmp_path,
        connection_test=True,
        confirm_connection_test=True,
        client=mock_client,
    )

    assert res["status"] == "passed"
    assert res["model_requested"] == PILOT_MODEL
    assert res["actual_model_identifier"] is None


# =============================================================================
# 5. LIVE RUNNER DUAL CONFIRMATION GATE
# =============================================================================

def test_openrouter_live_runner_dual_confirmation():
    """Live runner must reject execution if either flag is missing."""
    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_live_openrouter_pilot(live=True, confirm_live=False)

    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_live_openrouter_pilot(live=False, confirm_live=True)
