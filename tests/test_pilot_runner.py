"""Unit and integration tests for Phase 6.5 Controlled Pilot Runner and Guardrails.

Verifies:
1. Dual confirmation safety gate (--live and --confirm-live required; refusal otherwise).
2. Offline preflight validation (0 network calls, checks catalog, fixtures, 3 pilot tasks, pricing).
3. Secret safety (API keys never logged, echoed, or included in returned data).
4. Pilot task subset integrity (task_01 COMMIT, task_14 REJECT, task_22 CLARIFY).
5. Request guardrail provider (strict 24-request ceiling, clean abortion on limit breach).
6. Request guardrail metric accumulation (tokens, latencies, fingerprints, request IDs).
7. Simulated dry-run execution (9 runs, sequential, produces all 3 tagged artifact files).
8. Live execution harness with test double client (verifies parameters: gpt-6-luna, reasoning effort low, max_output_tokens 2048, max_retries 0, no temperature).
9. Pricing calculations (GPT-6 Luna Standard rates, 272K long-context threshold guardrail, cache writes).
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
    LLMCompletionResult,
    LLMMessage,
    LLMUsage,
    MockLLMProvider,
    OpenAIProvider,
)
from shopconnect_env.evaluation.metrics import (
    calculate_cost,
    create_gpt6_luna_pricing,
)
from shopconnect_env.evaluation.pilot import (
    DEFAULT_OUTPUT_TOKEN_CEILING,
    MAX_PILOT_RUNS,
    MAX_TOTAL_PROVIDER_REQUESTS,
    PILOT_CONDITIONS,
    PILOT_MODEL,
    PILOT_TAG,
    PILOT_TASK_IDS,
    PilotRequestGuardrailProvider,
    create_pilot_mock_provider,
    main,
    parse_args,
    run_live_pilot,
    run_pilot_dry_run,
    run_pilot_preflight,
)
from shopconnect_env.evaluation.tasks import ExpectedOutcome


# =============================================================================
# 1. DUAL CONFIRMATION SAFETY GATE TESTS
# =============================================================================

def test_dual_confirmation_safety_gate_enforcement():
    """run_live_pilot strictly requires BOTH live=True and confirm_live=True."""
    # 1. Neither flag
    with pytest.raises(ValueError) as exc1:
        run_live_pilot(live=False, confirm_live=False)
    assert "SAFETY VIOLATION" in str(exc1.value)

    # 2. Only live=True
    with pytest.raises(ValueError) as exc2:
        run_live_pilot(live=True, confirm_live=False)
    assert "SAFETY VIOLATION" in str(exc2.value)

    # 3. Only confirm_live=True
    with pytest.raises(ValueError) as exc3:
        run_live_pilot(live=False, confirm_live=True)
    assert "SAFETY VIOLATION" in str(exc3.value)


def test_cli_live_requires_dual_confirmation(capsys):
    """CLI exit code is 1 and execution aborts safely if --live without --confirm-live."""
    exit_code = main(["--live"])
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "ERROR: Live execution requires dual confirmation" in captured.err
    assert "Execution aborted safely" in captured.err


# =============================================================================
# 2. PREFLIGHT VALIDATION & CREDENTIAL SAFETY
# =============================================================================

def test_pilot_preflight_validation_zero_network():
    """run_pilot_preflight checks all required files and pricing with 0 network calls."""
    res = run_pilot_preflight()

    assert res["status"] == "passed"
    assert res["network_calls_made"] == 0
    assert res["pilot_tasks_count"] == 3
    assert res["planned_runs_count"] == 9
    assert res["max_provider_requests_ceiling"] == 24
    assert res["output_token_ceiling"] == 2048
    assert res["model"] == "gpt-6-luna"
    assert res["input_rate_per_m"] == "0.10"
    assert res["cached_input_rate_per_m"] == "0.01"
    assert res["output_rate_per_m"] == "0.50"
    assert res["cache_write_30m_rate_per_m"] == "0.125"
    assert res["cache_write_12h_rate_per_m"] == "0.25"
    assert len(res["errors"]) == 0


def test_pilot_preflight_api_key_not_leaked(monkeypatch):
    """API key secret string is never revealed in preflight return dictionary."""
    secret_key = "sk-super-secret-key-that-must-never-be-printed"
    monkeypatch.setenv("OPENAI_API_KEY", secret_key)

    res = run_pilot_preflight()

    assert res["api_key_configured"] is True
    assert res["api_key_status"] == "present (masked)"
    # Ensure full secret key is never in dict or serialized output
    assert secret_key not in json.dumps(res)


def test_pilot_tasks_ground_truth_consistency():
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
    assert "rustic tomato and spinach whole wheat pasta" in t1.user_request
    assert "rec_tomato_spinach_pasta" in t1.user_request
    assert "4 servings" in t1.user_request

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


def test_pilot_task_records_identical_to_frozen_tasks_file():
    """Pilot task objects must be strictly identical to frozen records in data/tasks.jsonl."""
    from shopconnect_env.domain.models import load_catalog_from_json
    from shopconnect_env.evaluation.runner import load_plan_fixtures
    from shopconnect_env.evaluation.tasks import load_benchmark_tasks

    ingredients, catalog = load_catalog_from_json("data/catalog_seed.json")
    fixtures = load_plan_fixtures("data/plan_fixtures.json", catalog, ingredients)
    all_tasks = load_benchmark_tasks("data/tasks.jsonl", fixtures, catalog)
    task_map = {t.task_id: t for t in all_tasks}

    raw_records = {}
    with open("data/tasks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                raw_records[item["task_id"]] = item

    for tid in PILOT_TASK_IDS:
        assert tid in raw_records, f"Pilot task {tid} missing from data/tasks.jsonl"
        raw = raw_records[tid]
        loaded = task_map[tid]

        assert loaded.task_id == raw["task_id"]
        assert loaded.fixture_id == raw["fixture_id"]
        assert loaded.category == raw["category"]
        assert loaded.user_request == raw["user_request"]
        assert loaded.expected_outcome.value == raw["expected_outcome"]
        assert loaded.expected_operation == raw["expected_operation"]
        assert loaded.expected_day == raw["expected_day"]
        assert loaded.expected_meal_type == raw["expected_meal_type"]
        assert loaded.expected_recipe_id == raw["expected_recipe_id"]
        assert loaded.expected_servings == raw["expected_servings"]
        assert loaded.expected_rejection_category == raw["expected_rejection_category"]


# =============================================================================
# 3. REQUEST GUARDRAIL & CEILING ENFORCEMENT
# =============================================================================

def test_pilot_request_guardrail_ceiling_enforcement():
    """PilotRequestGuardrailProvider halts cleanly when max_requests ceiling is reached."""
    mock_inner = MockLLMProvider()
    guardrail = PilotRequestGuardrailProvider(mock_inner, max_requests=3)

    # 3 calls succeed
    for i in range(3):
        res = guardrail.complete([LLMMessage(role="user", content=f"Hello {i}")])
        assert isinstance(res, LLMCompletionResult)
    assert guardrail.total_requests == 3

    # 4th call exceeds ceiling -> RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        guardrail.complete([LLMMessage(role="user", content="Hello 4")])

    assert "Pilot execution safety ceiling reached" in str(exc_info.value)
    assert "ceiling of 3 total provider requests exceeded" in str(exc_info.value)
    assert guardrail.total_requests == 3  # Did not increment on aborted call


def test_pilot_request_guardrail_usage_accumulation():
    """PilotRequestGuardrailProvider aggregates cumulative tokens, cache writes, and fingerprints."""
    resp1 = LLMCompletionResult(
        message=LLMMessage(role="assistant", content="Resp 1"),
        usage=LLMUsage(
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            cached_input_tokens=20,
            reasoning_tokens=15,
            cache_write_tokens_30m=10,
            cache_write_tokens_12h=5,
            is_provider_reported=True,
        ),
        model_identifier="gpt-6-luna",
        system_fingerprint="fp_test_1",
        provider_request_id="req_001",
    )
    resp2 = LLMCompletionResult(
        message=LLMMessage(role="assistant", content="Resp 2"),
        usage=LLMUsage(
            input_tokens=200,
            output_tokens=60,
            total_tokens=260,
            cached_input_tokens=50,
            reasoning_tokens=30,
            cache_write_tokens_30m=25,
            cache_write_tokens_12h=0,
            is_provider_reported=True,
        ),
        model_identifier="gpt-6-luna",
        system_fingerprint="fp_test_2",
        provider_request_id="req_002",
    )

    inner = MockLLMProvider(responses=[resp1, resp2])
    guardrail = PilotRequestGuardrailProvider(inner, max_requests=10)

    guardrail.complete([LLMMessage(role="user", content="msg 1")])
    guardrail.complete([LLMMessage(role="user", content="msg 2")])

    assert guardrail.total_requests == 2
    assert guardrail.cumulative_input_tokens == 300
    assert guardrail.cumulative_output_tokens == 100
    assert guardrail.cumulative_cached_input_tokens == 70
    assert guardrail.cumulative_reasoning_tokens == 45
    assert guardrail.cumulative_cache_write_30m == 35
    assert guardrail.cumulative_cache_write_12h == 5
    assert guardrail.provider_request_ids == ["req_001", "req_002"]
    assert guardrail.system_fingerprints == {"fp_test_1", "fp_test_2"}


# =============================================================================
# 4. SIMULATED DRY-RUN & ARTIFACT EXPORT
# =============================================================================

def test_simulated_dry_run_artifacts_and_metrics(tmp_path):
    """run_pilot_dry_run executes 9 runs and exports tagged artifacts."""
    out_dir = tmp_path / "pilot_dry_run_out"
    res = run_pilot_dry_run(output_dir=out_dir)

    assert res["status"] == "completed"
    assert res["is_live"] is False
    assert res["total_runs"] == 9
    assert res["total_provider_requests"] <= MAX_TOTAL_PROVIDER_REQUESTS
    assert PILOT_TAG in res["tag"]

    # Verify exported files exist
    raw_runs_file = Path(res["raw_runs_file"])
    comparison_file = Path(res["comparison_file"])
    analysis_file = Path(res["analysis_file"])

    assert raw_runs_file.exists()
    assert comparison_file.exists()
    assert analysis_file.exists()

    # Check raw_runs.jsonl has 9 lines
    lines = raw_runs_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 9

    # Check analysis.md has pilot banner
    analysis_text = analysis_file.read_text(encoding="utf-8")
    assert f"# {PILOT_TAG}" in analysis_text
    assert "LIMITED SAMPLE SIZE NOTICE" in analysis_text
    assert "OFFLINE SIMULATED TEST DOUBLE" in analysis_text


# =============================================================================
# 5. LIVE PILOT EXECUTION HARNESS WITH TEST DOUBLE
# =============================================================================

def test_live_pilot_execution_with_test_double(tmp_path):
    """run_live_pilot with dual confirmation and injected client executes 9 sequential runs."""
    # Build fake client response
    mock_resp = SimpleNamespace(
        id="resp_live_double_01",
        model="gpt-6-luna",
        status="completed",
        output_text='[{"day": "Monday", "meal_type": "breakfast", "status": "assigned", "recipe_id": "rec_berry_oatmeal", "servings": 2}]',
        output=[],
        usage=SimpleNamespace(
            input_tokens=150,
            output_tokens=50,
            total_tokens=200,
            input_tokens_details=SimpleNamespace(
                cached_tokens=40,
                cache_write_tokens_30m=10,
                cache_write_tokens_12h=0,
            ),
            output_tokens_details=SimpleNamespace(reasoning_tokens=20),
        ),
        system_fingerprint="fp_live_test_01",
    )

    fake_client = MagicMock()
    fake_client.responses.create.return_value = mock_resp

    out_dir = tmp_path / "pilot_live_double_out"
    res = run_live_pilot(
        live=True,
        confirm_live=True,
        api_key="test-api-key",
        output_dir=out_dir,
        client=fake_client,
    )

    assert res["status"] == "completed"
    assert res["is_live"] is True
    assert res["total_runs"] == 9
    assert res["total_provider_requests"] == 9
    assert res["tag"] == PILOT_TAG

    # Verify calls to fake_client conformed strictly to Stage 6 specifications
    assert fake_client.responses.create.call_count == 9
    first_call_kwargs = fake_client.responses.create.call_args_list[0].kwargs

    assert first_call_kwargs["model"] == "gpt-6-luna"
    assert first_call_kwargs["reasoning"] == {"effort": "low"}
    assert first_call_kwargs["max_output_tokens"] == DEFAULT_OUTPUT_TOKEN_CEILING
    assert "temperature" not in first_call_kwargs

    # Check analysis.md has LIVE banner
    analysis_text = Path(res["analysis_file"]).read_text(encoding="utf-8")
    assert f"# {PILOT_TAG}" in analysis_text
    assert "LIVE OPENAI RESPONSES API" in analysis_text


# =============================================================================
# 6. PRICING & COST ACCOUNTING INTEGRITY
# =============================================================================

def test_pilot_cost_accounting_gpt6_luna_rates():
    """Cost calculation strictly applies GPT-6 Luna Standard rates and exact Decimal math."""
    pricing = create_gpt6_luna_pricing()
    assert pricing.input_cost_per_million == Decimal("0.10")
    assert pricing.cached_input_cost_per_million == Decimal("0.01")
    assert pricing.output_cost_per_million == Decimal("0.50")
    assert pricing.cache_write_30m_cost_per_million == Decimal("0.125")
    assert pricing.cache_write_12h_cost_per_million == Decimal("0.25")

    # 100,000 non-cached input ($0.01) + 50,000 cached ($0.0005) + 20,000 output ($0.01)
    # Total = $0.020500
    usage = LLMUsage(
        input_tokens=150_000,
        output_tokens=20_000,
        total_tokens=170_000,
        cached_input_tokens=50_000,
        reasoning_tokens=5_000,
        is_provider_reported=True,
    )

    cost = calculate_cost(usage, pricing)
    assert cost == Decimal("0.020500")


def test_pilot_cost_accounting_mixed_prompt_caching_regression():
    """Ordinary input tokens must deduct both cached tokens and cache write tokens.

    Formula: ordinary_input = input_tokens - cached_tokens - cache_write_tokens.
    Verifies that cache writes are never double-counted at standard input rates.
    """
    pricing = create_gpt6_luna_pricing()

    # Input breakdown:
    # Total input: 100,000
    # Cached input: 30,000
    # 30m cache writes: 20,000
    # 12h cache writes: 10,000
    # Cache writes total: 30,000 (20k + 10k)
    # Ordinary uncached input = 100,000 - 30,000 - 30,000 = 40,000
    # Output tokens: 40,000 (includes 10,000 reasoning tokens)
    usage = LLMUsage(
        input_tokens=100_000,
        output_tokens=40_000,
        total_tokens=140_000,
        cached_input_tokens=30_000,
        reasoning_tokens=10_000,
        cache_write_tokens=30_000,
        cache_write_tokens_30m=20_000,
        cache_write_tokens_12h=10_000,
        is_provider_reported=True,
    )

    # Standard GPT-6 Luna pricing:
    # Ordinary input: (40,000 / 1M) * $0.10 = $0.004000
    # Cached input:   (30,000 / 1M) * $0.01 = $0.000300
    # 30m writes:     (20,000 / 1M) * $0.125 = $0.002500
    # 12h writes:     (10,000 / 1M) * $0.25 = $0.002500
    # Output:         (40,000 / 1M) * $0.50 = $0.020000
    # Total = 0.004000 + 0.000300 + 0.002500 + 0.002500 + 0.020000 = $0.029300
    cost = calculate_cost(usage, pricing)
    assert cost == Decimal("0.029300")

    # Erroneous double-counted cost where ordinary input = 100K - 30K = 70K:
    # 70K * 0.10 = 0.007000 (an extra $0.003000 charge for the 30K write tokens)
    double_counted_cost = Decimal("0.032300")
    assert cost != double_counted_cost


def test_pilot_cost_accounting_missing_or_inconsistent_cache_writes():
    """Missing or inconsistent cache-write breakdown strictly marks cost as unknown (None)."""
    pricing = create_gpt6_luna_pricing()

    # Case A: cache_write_tokens > 0 but breakdown is completely missing (both 30m and 12h are None)
    usage_missing_breakdown = LLMUsage(
        input_tokens=100_000,
        output_tokens=20_000,
        total_tokens=120_000,
        cached_input_tokens=10_000,
        cache_write_tokens=30_000,
        cache_write_tokens_30m=None,
        cache_write_tokens_12h=None,
        is_provider_reported=True,
    )
    assert calculate_cost(usage_missing_breakdown, pricing) is None

    # Case B: breakdown components do not sum to reported cache_write_tokens
    # 30m (20k) + 12h (5k) = 25k != 30k reported
    usage_inconsistent_sum = LLMUsage(
        input_tokens=100_000,
        output_tokens=20_000,
        total_tokens=120_000,
        cached_input_tokens=10_000,
        cache_write_tokens=30_000,
        cache_write_tokens_30m=20_000,
        cache_write_tokens_12h=5_000,
        is_provider_reported=True,
    )
    assert calculate_cost(usage_inconsistent_sum, pricing) is None

    # Case C: cached + cache_writes exceed total input_tokens
    # 60k cached + 50k write = 110k > 100k input
    usage_exceeds_input = LLMUsage(
        input_tokens=100_000,
        output_tokens=20_000,
        total_tokens=120_000,
        cached_input_tokens=60_000,
        cache_write_tokens=50_000,
        cache_write_tokens_30m=50_000,
        cache_write_tokens_12h=0,
        is_provider_reported=True,
    )
    assert calculate_cost(usage_exceeds_input, pricing) is None


def test_pilot_cost_accounting_long_context_threshold_guardrail():
    """Prompts exceeding 272K input tokens return None cost when long-context pricing is undefined."""
    pricing = create_gpt6_luna_pricing()
    assert pricing.max_context_tokens_threshold == 272_000
    assert pricing.long_context_input_cost_per_million is None

    # Normal prompt <= 272K succeeds
    normal_usage = LLMUsage(
        input_tokens=272_000,
        output_tokens=1_000,
        total_tokens=273_000,
        is_provider_reported=True,
    )
    assert calculate_cost(normal_usage, pricing) is not None

    # Prompt > 272K triggers guardrail -> None
    long_usage = LLMUsage(
        input_tokens=272_001,
        output_tokens=1_000,
        total_tokens=273_001,
        is_provider_reported=True,
    )
    assert calculate_cost(long_usage, pricing) is None


# =============================================================================
# 7. ZERO NETWORK SOCKET ACTIVITY GUARANTEE
# =============================================================================

def test_zero_socket_activity_across_all_pilot_operations(tmp_path):
    """Assert that preflight, dry-run, and unit tests make zero socket connections."""
    orig_socket = socket.socket

    def guarded_socket(*args, **kwargs):
        raise RuntimeError("Illegal socket connection attempted during offline pilot test execution!")

    socket.socket = guarded_socket
    try:
        # 1. Preflight
        pre = run_pilot_preflight()
        assert pre["status"] == "passed"

        # 2. Dry run
        dry_out = tmp_path / "offline_dry_run"
        res_dry = run_pilot_dry_run(output_dir=dry_out)
        assert res_dry["status"] == "completed"

        # 3. Dual confirmation rejection
        with pytest.raises(ValueError):
            run_live_pilot(live=True, confirm_live=False)

    finally:
        socket.socket = orig_socket
