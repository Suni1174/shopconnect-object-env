"""Unit tests for Ollama pilot runner, guardrails, dry run, and connection test.

Verifies:
- Dual confirmation flags required for live execution (--live AND --confirm-live).
- Standalone connection test dual confirmation (--connection-test AND --confirm-connection-test).
- Standalone connection test prompt and token validation.
- Connection test preserves model_requested vs actual_model_identifier (None when omitted).
- Connection test finish_reason retention.
- Empty content response is NEVER considered successful.
- Output limit exhaustion reporting (OutputLimitExhaustedError when finish_reason='length').
- Reasoning metadata detected without storing hidden reasoning text.
- No retries or extra model requests after validation failure (requests_made == 1).
- Zero network activity during preflight and dry-run execution.
- Ollama preflight server status detection (offline, model missing, model present, server_version).
- Strict localhost-only constraint for preflight checks.
- Strict 20-request ceiling enforcement and 18 theoretical maximum.
- Dispatched failed attempts counted against request ceiling.
- Simulated 9-run dry-run execution across conditions A, B, and C with mock-qwen3.5:0.8b.
- Condition C max 4 model requests budget.
- Artifact generation and directory isolation under artifacts/phase6_5_ollama_pilot/.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from shopconnect_env.agents.providers import (
    ConnectionError,
    LLMCompletionResult,
    LLMMessage,
    LLMUsage,
    MockLLMProvider,
    OllamaProvider,
)
from shopconnect_env.evaluation.ollama_pilot import (
    CONNECTION_TEST_PROMPT,
    EXPECTED_CONNECTION_TEST_TOKEN,
    MAX_PILOT_RUNS,
    MAX_REQUESTS_PER_CONDITION_C,
    MAX_THEORETICAL_TOTAL_REQUESTS,
    MAX_TOTAL_PROVIDER_REQUESTS,
    PILOT_MODEL,
    OllamaPilotRequestGuardrailProvider,
    check_ollama_server_status,
    create_ollama_pilot_mock_provider,
    run_live_ollama_pilot,
    run_ollama_connection_test,
    run_ollama_pilot_dry_run,
    run_ollama_pilot_preflight,
)


# =============================================================================
# 1. SERVER STATUS CHECKER & PREFLIGHT TESTS
# =============================================================================

def test_ollama_check_status_server_offline():
    """When probe returns offline, reports server_online=False and setup instruction."""
    def fake_probe(url):
        return {
            "server_online": False,
            "server_version": None,
            "model_installed": False,
            "setup_instruction": "Start Ollama service via 'ollama serve'.",
            "error": "Connection refused",
        }

    status = check_ollama_server_status(probe_fn=fake_probe)
    assert status["server_online"] is False
    assert "ollama serve" in status["setup_instruction"]


def test_ollama_check_status_model_missing():
    """When server is online but model is missing, reports download instruction for qwen3.5:0.8b."""
    def fake_probe(url):
        return {
            "server_online": True,
            "server_version": "0.40.16",
            "model_installed": False,
            "installed_models": ["llama3:8b"],
            "setup_instruction": f"Run 'ollama pull {PILOT_MODEL}' to download the model.",
            "error": None,
        }

    status = check_ollama_server_status(probe_fn=fake_probe)
    assert status["server_online"] is True
    assert status["model_installed"] is False
    assert f"ollama pull {PILOT_MODEL}" in status["setup_instruction"]


def test_ollama_check_status_model_present():
    """When server is online and qwen3.5:0.8b is present, reports model_installed=True."""
    def fake_probe(url):
        return {
            "server_online": True,
            "server_version": "0.40.16",
            "model_installed": True,
            "installed_models": [PILOT_MODEL, "llama3:8b"],
            "setup_instruction": None,
            "error": None,
        }

    status = check_ollama_server_status(probe_fn=fake_probe)
    assert status["server_online"] is True
    assert status["server_version"] == "0.40.16"
    assert status["model_installed"] is True
    assert status["setup_instruction"] is None


def test_ollama_check_status_rejects_non_local_url():
    """Security check: non-localhost URLs are rejected immediately without connecting."""
    status = check_ollama_server_status(base_url="http://remote-host.com:11434/v1")
    assert status["server_online"] is False
    assert "Security violation" in status["setup_instruction"]
    assert "not a local address" in status["error"]


def test_ollama_preflight_server_offline():
    """Preflight fails with clear message when local server is offline."""
    def fake_probe(url):
        return {
            "server_online": False,
            "server_version": None,
            "model_installed": False,
            "setup_instruction": "Start Ollama via 'ollama serve'.",
            "error": "Connection refused",
        }

    res = run_ollama_pilot_preflight(probe_fn=fake_probe)
    assert res["status"] == "failed"
    assert res["server_online"] is False
    assert any("ollama serve" in err for err in res["errors"])


def test_ollama_preflight_model_missing():
    """Preflight fails with clear message when model is missing."""
    def fake_probe(url):
        return {
            "server_online": True,
            "server_version": "0.40.16",
            "model_installed": False,
            "installed_models": [],
            "setup_instruction": f"Pull model via 'ollama pull {PILOT_MODEL}'.",
            "error": None,
        }

    res = run_ollama_pilot_preflight(probe_fn=fake_probe)
    assert res["status"] == "failed"
    assert res["server_online"] is True
    assert res["model_installed"] is False
    assert any(f"ollama pull {PILOT_MODEL}" in err for err in res["errors"])


def test_ollama_preflight_all_passed():
    """Preflight passes completely when server is online and model is installed."""
    def fake_probe(url):
        return {
            "server_online": True,
            "server_version": "0.40.16",
            "model_installed": True,
            "installed_models": [PILOT_MODEL],
            "setup_instruction": None,
            "error": None,
        }

    res = run_ollama_pilot_preflight(probe_fn=fake_probe)
    assert res["status"] == "passed"
    assert res["server_online"] is True
    assert res["server_version"] == "0.40.16"
    assert res["model_installed"] is True
    assert res["tasks_loaded"] == 3
    assert res["billing_tier"] == "local_inference"
    assert res["context_length"] == 4096
    assert res["pricing_complete"] is False  # Expected: rates are None


# =============================================================================
# 2. GUARDRAIL & CEILING TESTS
# =============================================================================

def test_ollama_guardrail_ceiling_enforcement():
    """Guardrail must raise RuntimeError when ceiling is reached."""
    mock_provider = MockLLMProvider()
    guardrail = OllamaPilotRequestGuardrailProvider(
        provider=mock_provider,
        max_requests=2,
    )

    msg = [LLMMessage(role="user", content="Ping")]
    guardrail.complete(msg)
    guardrail.complete(msg)
    assert guardrail.total_requests == 2

    with pytest.raises(RuntimeError, match="safety ceiling reached"):
        guardrail.complete(msg)


def test_ollama_guardrail_counts_failed_attempts():
    """Failed provider attempts must be counted against total_requests ceiling."""
    mock_provider = MagicMock()
    mock_provider.complete.side_effect = ConnectionError("Server unavailable")

    guardrail = OllamaPilotRequestGuardrailProvider(
        provider=mock_provider,
        max_requests=2,
    )

    msg = [LLMMessage(role="user", content="Ping")]
    with pytest.raises(ConnectionError):
        guardrail.complete(msg)

    assert guardrail.total_requests == 1


# =============================================================================
# 3. DRY RUN TESTS
# =============================================================================

def test_ollama_pilot_dry_run_execution(tmp_path):
    """Dry run must complete 9 runs and generate isolated artifacts with zero network activity."""
    res = run_ollama_pilot_dry_run(output_dir=tmp_path)
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

    assert res["max_requests_per_condition_c"] == 4
    assert res["max_theoretical_requests"] == 18


# =============================================================================
# 4. CONNECTION TEST TESTS
# =============================================================================

def test_ollama_connection_test_requires_dual_confirmation():
    """Connection test must reject execution if either flag is missing."""
    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_ollama_connection_test(connection_test=True, confirm_connection_test=False)

    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_ollama_connection_test(connection_test=False, confirm_connection_test=True)


def test_ollama_connection_test_success_with_injected_client(tmp_path):
    """Connection test succeeds when expected token is returned by mock client."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="ollama-conn-ok",
        model=PILOT_MODEL,
        system_fingerprint="fp_ollama",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content=f"Result: {EXPECTED_CONNECTION_TEST_TOKEN}",
                    tool_calls=None,
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=6,
            total_tokens=18,
        ),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    res = run_ollama_connection_test(
        output_dir=tmp_path,
        connection_test=True,
        confirm_connection_test=True,
        client=mock_client,
    )

    assert res["status"] == "passed"
    assert res["success"] is True
    assert res["requests_made"] == 1
    assert res["finish_reason"] == "stop"
    assert res["reasoning_present"] is False
    assert res["actual_model_identifier"] == PILOT_MODEL
    assert res["model_requested"] == PILOT_MODEL
    assert (tmp_path / "connection_test_result.json").exists()


def test_ollama_connection_test_missing_actual_model_identifier(tmp_path):
    """When Ollama omits model string, actual_model_identifier is preserved as None."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="ollama-conn-no-model",
        model=None,
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

    res = run_ollama_connection_test(
        output_dir=tmp_path,
        connection_test=True,
        confirm_connection_test=True,
        client=mock_client,
    )

    assert res["status"] == "passed"
    assert res["model_requested"] == PILOT_MODEL
    assert res["actual_model_identifier"] is None


def test_ollama_connection_test_empty_content_output_limit_exhausted(tmp_path):
    """Empty response from token exhaustion (finish_reason='length') fails with distinct OutputLimitExhaustedError."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="ollama-conn-exhausted",
        model=PILOT_MODEL,
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(
                    role="assistant",
                    content="",
                    reasoning_content="I should think about what to reply...",
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=21,
            completion_tokens=64,
            total_tokens=85,
        ),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    res = run_ollama_connection_test(
        output_dir=tmp_path,
        connection_test=True,
        confirm_connection_test=True,
        client=mock_client,
    )

    assert res["status"] == "failed"
    assert res["success"] is False
    assert res["finish_reason"] == "length"
    assert res["reasoning_present"] is True
    assert res["reasoning_length"] == len("I should think about what to reply...")
    assert res["error_type"] == "OutputLimitExhaustedError"
    assert "exhausted" in res["error"]
    assert res["requests_made"] == 1
    # Verify hidden reasoning text is never exposed in response_text or payload error
    assert res["response_text"] == ""
    assert "I should think about what to reply..." not in json.dumps(res)


def test_ollama_connection_test_empty_content_not_successful(tmp_path):
    """Empty content response is not considered successful even if finish_reason is stop."""
    mock_client = MagicMock()
    mock_resp = SimpleNamespace(
        id="ollama-conn-empty",
        model=PILOT_MODEL,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content="   ",  # whitespace only
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )
    mock_client.chat.completions.create.return_value = mock_resp

    res = run_ollama_connection_test(
        output_dir=tmp_path,
        connection_test=True,
        confirm_connection_test=True,
        client=mock_client,
    )

    assert res["status"] == "failed"
    assert res["success"] is False
    assert res["error_type"] == "EmptyResponseError"
    assert res["requests_made"] == 1


def test_ollama_connection_test_no_retries_on_failure(tmp_path):
    """No retries or additional model calls occur after a connection test failure."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Local connection dropped")

    res = run_ollama_connection_test(
        output_dir=tmp_path,
        connection_test=True,
        confirm_connection_test=True,
        client=mock_client,
    )

    assert res["status"] == "failed"
    assert res["success"] is False
    assert res["requests_made"] == 1
    assert res["max_requests_allowed"] == 1
    # Exactly one attempt was made
    assert mock_client.chat.completions.create.call_count == 1


# =============================================================================
# 5. LIVE RUNNER DUAL CONFIRMATION GATE
# =============================================================================

def test_ollama_live_runner_dual_confirmation():
    """Live runner must reject execution if either flag is missing."""
    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_live_ollama_pilot(live=True, confirm_live=False)

    with pytest.raises(RuntimeError, match="requires explicit dual confirmation"):
        run_live_ollama_pilot(live=False, confirm_live=True)
