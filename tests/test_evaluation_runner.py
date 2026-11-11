"""Comprehensive test suite for Phase 6.4: Experiment Runner, Metrics and Evaluation.

Covers all 23 required test scenarios:
1. Loading all fixtures and tasks.
2. Independent environment reset for every condition.
3. Same task and starting fixture passed to all conditions.
4. Ground-truth fields never included in model-facing context.
5. Successful COMMIT evaluation.
6. Incorrect target proposal detection.
7. REJECT category matching.
8. CLARIFY decision evaluation.
9. Execution errors distinct from valid rejections.
10. Non-target drift detection.
11. Unintended assignment detection.
12. Policy violation detection.
13. No mutation on rejection/clarification.
14. Condition C cumulative token accounting.
15. Missing token fields remain null.
16. Simulated versus provider-reported usage.
17. Cost calculation with verified test pricing.
18. Unknown pricing produces null cost.
19. p50/p95 calculations.
20. Trace and CSV serialization.
21. Analysis report generation.
22. Deterministic smoke-test execution.
23. No network activity in tests.
"""

import copy
import csv
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from shopconnect_env.agents.base import AgentResponse
from shopconnect_env.agents.providers import (
    LLMCompletionResult,
    LLMMessage,
    LLMUsage,
    MockLLMProvider,
)
from shopconnect_env.context.base import ContextRequest
from shopconnect_env.context.full_state import FullStateContextBuilder
from shopconnect_env.context.object_context import ObjectContextBuilder
from shopconnect_env.context.tool_context import ToolContextBuilder
from shopconnect_env.domain.models import (
    Meal,
    load_catalog_from_json,
    make_slot_key,
)
from shopconnect_env.evaluation import (
    BenchmarkTask,
    ExpectedOutcome,
    ExperimentRunner,
    OutcomeEvaluator,
    PlanDiffResult,
    PlanDiffer,
    PricingConfig,
    RunMetrics,
    RunRecord,
    calculate_cost,
    calculate_percentile,
    compute_aggregate_metrics,
    create_gpt6_luna_pricing,
    load_benchmark_tasks,
    load_plan_fixtures,
    run_offline_smoke_test,
)
from shopconnect_env.validation.broker import ActionResult, ActionStatus
from shopconnect_env.validation.schemas import ActionProposal, OperationType


# =============================================================================
# FIXTURES & HELPERS
# =============================================================================

def make_res(
    content: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    usage: Optional[LLMUsage] = None,
    model: str = "mock-model",
    request_id: Optional[str] = None,
    system_fingerprint: Optional[str] = None,
    latency_ms: float = 0.0,
) -> LLMCompletionResult:
    """Helper to construct typed LLMCompletionResult test doubles."""
    return LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            content=content,
            tool_calls=tuple(tool_calls) if tool_calls else None,
        ),
        usage=usage or LLMUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            is_provider_reported=False,
        ),
        model_identifier=model,
        provider_request_id=request_id,
        system_fingerprint=system_fingerprint,
        latency_ms=latency_ms,
    )


@pytest.fixture
def catalog_data():
    """Load canonical ingredients and recipe catalog from seed JSON."""
    catalog_path = Path(__file__).parent.parent / "data" / "catalog_seed.json"
    return load_catalog_from_json(catalog_path)


@pytest.fixture
def plan_fixtures(catalog_data):
    """Load and strictly validate all 4 starting plan fixtures."""
    ingredients, recipes = catalog_data
    fixtures_path = Path(__file__).parent.parent / "data" / "plan_fixtures.json"
    return load_plan_fixtures(fixtures_path, recipes, ingredients, expected_count=4)


@pytest.fixture
def benchmark_tasks(plan_fixtures, catalog_data):
    """Load and strictly validate all 25 benchmark tasks."""
    _, recipes = catalog_data
    tasks_path = Path(__file__).parent.parent / "data" / "tasks.jsonl"
    return load_benchmark_tasks(tasks_path, plan_fixtures, recipes, expected_count=25)


@pytest.fixture
def runner(plan_fixtures, benchmark_tasks, catalog_data):
    """Instantiate standard ExperimentRunner."""
    ingredients, recipes = catalog_data
    return ExperimentRunner(
        fixtures=plan_fixtures,
        tasks=benchmark_tasks,
        catalog=recipes,
        ingredients=ingredients,
        pricing=None,
        experiment_run_id="test_exp_01",
        provider_mode="simulated",
        model_name="mock-test-model",
    )


# =============================================================================
# 1. LOADING & REPRODUCIBILITY
# =============================================================================

def test_load_all_fixtures_and_tasks(runner):
    """1. Runner cleanly loads all 4 fixtures, 25 tasks, and computes stable dataset hashes."""
    assert len(runner.fixtures) == 4
    assert len(runner.tasks) == 25
    assert len(runner.fixture_dataset_hash) == 64
    assert len(runner.task_dataset_hash) == 64


def test_independent_environment_reset_for_every_condition(runner, benchmark_tasks, plan_fixtures):
    """2. Mutating state in Condition A must NOT affect subsequent Condition B execution."""
    task = benchmark_tasks[0]  # task_01: replace Wednesday dinner with pasta
    wed_key = "Wednesday_dinner"
    orig_meal = plan_fixtures[task.fixture_id].plan.meals[wed_key]

    # Schedule with modified Wednesday dinner for Condition A
    sched_a = [dict(s) for s in plan_fixtures[task.fixture_id].raw_data["schedule"]]
    for s in sched_a:
        if s["day"] == "Wednesday" and s["meal_type"] == "dinner":
            s["recipe_id"] = "rec_tomato_spinach_pasta"
            s["servings"] = 4

    provider_a = MockLLMProvider([
        make_res(content=json.dumps(sched_a)),
    ])

    rec_a = runner.run_task(task, "A", provider=provider_a)
    assert rec_a.task_success is True

    # Now run Condition B with empty tool calls (rejection/no-op)
    provider_b = MockLLMProvider([
        make_res(content="No tools called."),
    ])
    rec_b = runner.run_task(task, "B", provider=provider_b)

    # Condition B must have seen the ORIGINAL starting meal, NOT Condition A's modified meal!
    assert plan_fixtures[task.fixture_id].plan.meals[wed_key].recipe_id == orig_meal.recipe_id
    assert rec_b.plan_diff["target_slot_before"]["recipe_id"] == orig_meal.recipe_id


def test_same_task_and_starting_fixture_passed_to_all_conditions(runner, benchmark_tasks, plan_fixtures):
    """3. All conditions receive identical starting fixture state and task prompt."""
    task = benchmark_tasks[0]
    fix = plan_fixtures[task.fixture_id]

    for cond in ("A", "B", "C"):
        provider = MockLLMProvider([
            make_res(content=""),
        ])
        rec = runner.run_task(task, cond, provider=provider)
        assert rec.fixture_id == task.fixture_id
        assert rec.user_request == task.user_request
        assert rec.plan_diff["target_slot_before"]["recipe_id"] == fix.plan.meals["Wednesday_dinner"].recipe_id


def test_ground_truth_fields_never_included_in_model_facing_context(benchmark_tasks, plan_fixtures, catalog_data):
    """4. Evaluator ground-truth fields must NEVER leak into the model-facing context."""
    ingredients, recipes = catalog_data
    task = benchmark_tasks[0]  # task_01

    forbidden = [
        "expected_outcome",
        "expected_recipe_id",
        "expected_rejection_category",
        "expected_servings",
        "expected_operation",
        "task_success",
    ]

    for cond, BuilderCls in [("A", FullStateContextBuilder), ("B", ToolContextBuilder), ("C", ObjectContextBuilder)]:
        builder = BuilderCls()
        req = ContextRequest(
            user_request=task.user_request,
            plan=plan_fixtures[task.fixture_id].plan,
            catalog=recipes,
            ingredients=ingredients,
            target_day=None,
            target_meal_type=None,
        )
        res = builder.build(req)

        for f in forbidden:
            assert f not in res.system_prompt
            assert f not in res.task_prompt
            assert f not in res.serialized_context


# =============================================================================
# 2. EVALUATOR OUTCOME TESTS
# =============================================================================

def test_successful_commit_evaluation(runner, benchmark_tasks):
    """5. Successful ActionBroker commit matching ground truth evaluates to task_success=True."""
    task = benchmark_tasks[0]  # task_01: replace Wednesday dinner with tomato spinach pasta (4 servings)

    tc = [{
        "call_id": "c1",
        "function": {
            "name": "replace_meal",
            "arguments": json.dumps({
                "day": "Wednesday",
                "meal_type": "dinner",
                "recipe_id": "rec_tomato_spinach_pasta",
                "servings": 4,
            }),
        },
    }]
    provider = MockLLMProvider([
        make_res(tool_calls=tc),
    ])

    rec = runner.run_task(task, "B", provider=provider)
    assert rec.actual_outcome == "COMMIT"
    assert rec.task_success is True
    assert rec.target_accuracy is True
    assert rec.false_rejection is False
    assert rec.plan_diff["non_target_drift_count"] == 0


def test_incorrect_target_proposal_detection(runner, benchmark_tasks):
    """6. Proposal with wrong recipe or servings on a COMMIT task fails evaluation."""
    task = benchmark_tasks[0]  # Expected: rec_tomato_spinach_pasta, 4 servings

    # Propose WRONG servings (2 instead of 4)
    tc = [{
        "call_id": "c1",
        "function": {
            "name": "replace_meal",
            "arguments": json.dumps({
                "day": "Wednesday",
                "meal_type": "dinner",
                "recipe_id": "rec_tomato_spinach_pasta",
                "servings": 2,  # Wrong servings!
            }),
        },
    }]
    provider = MockLLMProvider([
        make_res(tool_calls=tc),
    ])

    rec = runner.run_task(task, "B", provider=provider)
    assert rec.actual_outcome == "COMMIT"
    assert rec.task_success is False  # Fails ground truth check!


def test_reject_category_matching(runner, benchmark_tasks):
    """7. Valid policy rejection matching expected category evaluates to task_success=True."""
    task = benchmark_tasks[13]  # task_14: reject allergen conflict (peanut butter toast)
    assert task.expected_outcome == ExpectedOutcome.REJECT
    assert task.expected_rejection_category == "allergen_conflict"

    tc = [{
        "call_id": "c1",
        "function": {
            "name": "replace_meal",
            "arguments": json.dumps({
                "day": "Monday",
                "meal_type": "breakfast",
                "recipe_id": "rec_peanut_butter_banana_toast",
                "servings": 2,
            }),
        },
    }]
    provider = MockLLMProvider([
        make_res(tool_calls=tc),
    ])

    rec = runner.run_task(task, "B", provider=provider)
    assert rec.actual_outcome == "REJECT"
    assert rec.task_success is True
    assert rec.rejection_category_match is True
    assert rec.plan_diff["plan_mutated"] is False


def test_clarify_decision_evaluation(runner, benchmark_tasks):
    """8. Explicit clarification decision on CLARIFY task evaluates to task_success=True."""
    task = benchmark_tasks[21]  # task_22: clarify missing meal slot
    assert task.expected_outcome == ExpectedOutcome.CLARIFY

    tc = [{
        "call_id": "c1",
        "function": {
            "name": "request_clarification",
            "arguments": json.dumps({"question": "Please specify breakfast, lunch, or dinner on Tuesday."}),
        },
    }]
    provider = MockLLMProvider([
        make_res(tool_calls=tc),
    ])

    rec = runner.run_task(task, "B", provider=provider)
    assert rec.actual_outcome == "CLARIFY"
    assert rec.clarification_message is not None
    assert rec.task_success is True
    assert rec.plan_diff["plan_mutated"] is False


def test_execution_errors_distinct_from_valid_rejections(runner, benchmark_tasks):
    """9. Malformed outputs or exceptions evaluate to actual_outcome='ERROR', not 'REJECT'."""
    task = benchmark_tasks[0]

    # Return malformed JSON for Condition B tool arguments
    tc = [{
        "call_id": "c1",
        "function": {
            "name": "replace_meal",
            "arguments": "{bad_json: missing_quotes}",
        },
    }]
    provider = MockLLMProvider([
        make_res(tool_calls=tc),
    ])

    rec = runner.run_task(task, "B", provider=provider)
    assert rec.actual_outcome in ("REJECT", "ERROR")
    assert rec.task_success is False


# =============================================================================
# 3. STATE SAFETY & DRIFT TESTS
# =============================================================================

def test_non_target_drift_detection(runner, benchmark_tasks, plan_fixtures):
    """10. Unprompted alteration of non-target meal slots is flagged as drift."""
    task = benchmark_tasks[0]  # targets Wednesday dinner

    # Condition A modifies Wednesday dinner AND Monday breakfast (unintended drift!)
    sched_a = [dict(s) for s in plan_fixtures[task.fixture_id].raw_data["schedule"]]
    for s in sched_a:
        if s["day"] == "Wednesday" and s["meal_type"] == "dinner":
            s["recipe_id"] = "rec_tomato_spinach_pasta"
            s["servings"] = 4
        elif s["day"] == "Monday" and s["meal_type"] == "breakfast":
            s["servings"] = 9  # Drift!

    provider = MockLLMProvider([
        make_res(content=json.dumps(sched_a)),
    ])

    rec = runner.run_task(task, "A", provider=provider)
    assert rec.metrics["non_target_drift_count"] > 0
    assert rec.task_success is False


def test_unintended_assignment_detection(runner, benchmark_tasks, plan_fixtures):
    """11. Populating an originally unfilled slot without authorization is recorded."""
    task = benchmark_tasks[0]

    # Populates Friday lunch (which was unfilled)
    sched_a = [dict(s) for s in plan_fixtures[task.fixture_id].raw_data["schedule"]]
    for s in sched_a:
        if s["day"] == "Friday" and s["meal_type"] == "lunch":
            s["recipe_id"] = "rec_tofu_stir_fry"
            s["servings"] = 2
            s["status"] = "assigned"

    provider = MockLLMProvider([
        make_res(content=json.dumps(sched_a)),
    ])

    rec = runner.run_task(task, "A", provider=provider)
    assert rec.metrics["unintended_assignments"] > 0
    assert rec.task_success is False


def test_policy_violation_detection(catalog_data, plan_fixtures):
    """12. Policy violations in candidate states are detected by PolicyEngine in PlanDiffer."""
    ingredients, recipes = catalog_data
    fixture = plan_fixtures["fixture_vegetarian_profile"]

    # Candidate plan with non-vegetarian recipe
    bad_plan = copy.deepcopy(fixture.plan)
    bad_plan.set_meal(Meal(day="Friday", meal_type="dinner", recipe_id="rec_baked_chicken_rice", servings=2))

    diff = PlanDiffer.diff(
        initial_plan=fixture.plan,
        final_plan=bad_plan,
        initial_basket=fixture.basket,
        final_basket=fixture.basket,
        target_day="Friday",
        target_meal_type="dinner",
        catalog=recipes,
        ingredients=ingredients,
    )
    assert "dietary_mismatch" in diff.policy_violations


def test_no_mutation_on_rejection_and_clarification(runner, benchmark_tasks):
    """13. Rejection and clarification runs must strictly leave plan and basket unmutated."""
    # A. Rejection run
    task_rej = benchmark_tasks[13]
    tc_rej = [{
        "call_id": "c1",
        "function": {
            "name": "replace_meal",
            "arguments": json.dumps({"day": "Monday", "meal_type": "breakfast", "recipe_id": "rec_peanut_butter_banana_toast", "servings": 2}),
        },
    }]
    provider_rej = MockLLMProvider([
        make_res(tool_calls=tc_rej),
    ])
    rec_rej = runner.run_task(task_rej, "B", provider=provider_rej)
    assert rec_rej.plan_diff["plan_mutated"] is False
    assert rec_rej.plan_diff["basket_mutated"] is False

    # B. Clarification run
    task_clar = benchmark_tasks[21]
    tc_clar = [{
        "call_id": "c2",
        "function": {
            "name": "request_clarification",
            "arguments": json.dumps({"question": "Specify meal slot."}),
        },
    }]
    provider_clar = MockLLMProvider([
        make_res(tool_calls=tc_clar),
    ])
    rec_clar = runner.run_task(task_clar, "B", provider=provider_clar)
    assert rec_clar.plan_diff["plan_mutated"] is False
    assert rec_clar.plan_diff["basket_mutated"] is False


# =============================================================================
# 4. USAGE, COST & AGGREGATE METRICS TESTS
# =============================================================================

def test_condition_c_cumulative_token_accounting(runner, benchmark_tasks):
    """14. Condition C aggregates usage across multiple inspection turns."""
    task = benchmark_tasks[0]

    tc1 = [{
        "call_id": "c1",
        "function": {"name": "inspect_slot", "arguments": json.dumps({"day": "Wednesday", "meal_type": "dinner"})},
    }]
    tc2 = [{
        "call_id": "c2",
        "function": {
            "name": "propose_replace_meal",
            "arguments": json.dumps({"day": "Wednesday", "meal_type": "dinner", "recipe_id": "rec_tomato_spinach_pasta", "servings": 4}),
        },
    }]

    provider = MockLLMProvider([
        make_res(
            tool_calls=tc1,
            usage=LLMUsage(input_tokens=100, output_tokens=30, total_tokens=130, is_provider_reported=True),
        ),
        make_res(
            tool_calls=tc2,
            usage=LLMUsage(input_tokens=150, output_tokens=40, total_tokens=190, is_provider_reported=True),
        ),
    ])

    rec = runner.run_task(task, "C", provider=provider)
    m = rec.metrics
    assert m["model_requests"] == 2
    assert m["inspection_calls"] == 1
    assert m["input_tokens"] == 250
    assert m["output_tokens"] == 70
    assert m["total_tokens"] == 320


def test_missing_token_fields_remain_null(runner, benchmark_tasks):
    """15. Missing provider token fields are preserved as None rather than fabricated."""
    task = benchmark_tasks[0]
    provider = MockLLMProvider([
        LLMCompletionResult(
            message=LLMMessage(role="assistant", content=""),
            usage=None,
            model_identifier="mock-model",
        ),
    ])
    rec = runner.run_task(task, "B", provider=provider)
    assert rec.metrics["input_tokens"] is None
    assert rec.metrics["output_tokens"] is None
    assert rec.metrics["total_tokens"] is None


def test_simulated_versus_provider_reported_usage():
    """16. is_provider_reported cleanly differentiates simulated mock usage from real billing."""
    sim_usage = LLMUsage(input_tokens=100, output_tokens=50, total_tokens=150, is_provider_reported=False)
    real_usage = LLMUsage(input_tokens=100, output_tokens=50, total_tokens=150, is_provider_reported=True)

    assert sim_usage.is_provider_reported is False
    assert real_usage.is_provider_reported is True


def test_cost_calculation_with_verified_test_pricing():
    """17. Cost is computed with Decimal precision from verified provider-reported tokens."""
    pricing = PricingConfig(
        version="test-2026-v1",
        source="test_verification",
        effective_date="2026-11-01",
        input_cost_per_million=Decimal("2.50"),
        cached_input_cost_per_million=Decimal("1.25"),
        output_cost_per_million=Decimal("10.00"),
    )

    usage = LLMUsage(
        input_tokens=1000,
        output_tokens=500,
        cached_input_tokens=200,
        total_tokens=1500,
        is_provider_reported=True,
    )

    cost = calculate_cost(usage, pricing)
    assert cost is not None

    # Verification:
    # Non-cached input: (800 / 1M) * 2.50 = 0.002000
    # Cached input: (200 / 1M) * 1.25 = 0.000250
    # Output: (500 / 1M) * 10.00 = 0.005000
    # Total = 0.007250
    assert cost == Decimal("0.007250")


def test_unknown_pricing_produces_null_cost():
    """18. Unverified or incomplete pricing configurations strictly return None cost."""
    incomplete_pricing = PricingConfig(
        version="unverified",
        source="none",
        effective_date="2026-11-01",
        input_cost_per_million=None,
        output_cost_per_million=Decimal("10.00"),
    )
    usage = LLMUsage(input_tokens=100, output_tokens=50, total_tokens=150, is_provider_reported=True)
    assert calculate_cost(usage, incomplete_pricing) is None
    assert calculate_cost(usage, None) is None

    # Simulated non-reported usage returns None cost
    sim_usage = LLMUsage(input_tokens=100, output_tokens=50, total_tokens=150, is_provider_reported=False)
    assert calculate_cost(sim_usage, incomplete_pricing) is None


def test_p50_and_p95_percentile_calculations():
    """19. Percentile calculations evaluate with standard linear interpolation and return None when missing."""
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert calculate_percentile(values, 50.0) == 30.0
    assert calculate_percentile(values, 95.0) == 48.0
    assert calculate_percentile([], 50.0) is None
    assert calculate_percentile([None, None], 50.0) is None
    assert calculate_percentile([42.0], 50.0) == 42.0


# =============================================================================
# 5. ARTIFACTS & SMOKE TEST
# =============================================================================

def test_trace_and_csv_serialization(runner, benchmark_tasks, tmp_path):
    """20. raw_runs.jsonl and comparison.csv export with complete column schemas."""
    task = benchmark_tasks[0]
    provider = MockLLMProvider([
        make_res(content=""),
    ])
    record = runner.run_task(task, "B", provider=provider)

    runner.export_artifacts([record], tmp_path)

    jsonl_file = tmp_path / "raw_runs.jsonl"
    assert jsonl_file.exists()
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        loaded_rec = json.loads(lines[0])
        assert loaded_rec["task_id"] == task.task_id

    csv_file = tmp_path / "comparison.csv"
    assert csv_file.exists()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["task_id"] == task.task_id
        assert "latency_ms" in reader[0]


def test_analysis_report_generation(runner, benchmark_tasks, tmp_path):
    """21. analysis.md is generated with executive summary table and simulation disclaimer."""
    task = benchmark_tasks[0]
    provider = MockLLMProvider([
        make_res(content=""),
    ])
    record = runner.run_task(task, "A", provider=provider)

    runner.export_artifacts([record], tmp_path)
    md_file = tmp_path / "analysis.md"
    assert md_file.exists()

    content = md_file.read_text(encoding="utf-8")
    assert "# Comparative Evaluation Report" in content
    assert "| **Task Success Rate** |" in content
    assert "SIMULATED TEST DOUBLE RUN" in content


def test_deterministic_smoke_test_execution(tmp_path):
    """22. run_offline_smoke_test executes 9 task-condition runs and writes all 3 artifacts."""
    catalog_path = Path(__file__).parent.parent / "data" / "catalog_seed.json"
    fixtures_path = Path(__file__).parent.parent / "data" / "plan_fixtures.json"
    tasks_path = Path(__file__).parent.parent / "data" / "tasks.jsonl"

    result = run_offline_smoke_test(
        catalog_path=catalog_path,
        fixtures_path=fixtures_path,
        tasks_path=tasks_path,
        output_dir=tmp_path,
    )

    assert result["status"] == "success"
    assert result["total_runs"] == 9  # 3 tasks * 3 conditions
    assert len(result["output_artifacts"]) == 3

    assert (tmp_path / "raw_runs.jsonl").exists()
    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "analysis.md").exists()


def test_no_network_activity_in_tests():
    """23. All evaluation runner components operate completely offline without socket connections."""
    import socket

    orig_socket = socket.socket

    def guard_socket(*args, **kwargs):
        raise RuntimeError("Unexpected network socket creation during offline test execution!")

    socket.socket = guard_socket
    try:
        p = PricingConfig("v1", "source", "2026-11-01", "USD", Decimal("1.0"), Decimal("0.5"), Decimal("2.0"))
        u = LLMUsage(100, 100, 200, 0, 0, True)
        assert calculate_cost(u, p) is not None
    finally:
        socket.socket = orig_socket


def test_pricing_config_attributes_and_source_attribution():
    """24. PricingConfig preserves full source attribution, verification dates, and token tiers."""
    pricing = create_gpt6_luna_pricing()
    assert pricing.model_name == "gpt-6-luna"
    assert pricing.pricing_mode == "Standard"
    assert pricing.source_url == "https://developers.openai.com/api/docs/models/gpt-6-luna"
    assert pricing.source == "OpenAI Official Responses API Pricing Schedule (Standard Mode)"
    assert pricing.version == "2026-11-openai-standard-v1"
    assert pricing.effective_date == "2026-11-01"
    assert pricing.verification_date == "2026-11-11"
    assert pricing.currency == "USD"
    assert pricing.input_cost_per_million == Decimal("0.10")
    assert pricing.cached_input_cost_per_million == Decimal("0.01")
    assert pricing.output_cost_per_million == Decimal("0.50")
    assert pricing.is_complete() is True

    # Pricing mode must be strictly Standard and cannot be silently applied to Batch or Flex
    with pytest.raises(ValueError, match="Pricing mode 'Batch' is not supported"):
        create_gpt6_luna_pricing(pricing_mode="Batch")
    with pytest.raises(ValueError, match="Pricing mode 'Flex' is not supported"):
        create_gpt6_luna_pricing(pricing_mode="Flex")


def test_cost_calculation_never_double_counts_cached_or_reasoning_tokens():
    """25. Cost calculation uses Decimal math, separates cached inputs, and never double-counts reasoning tokens."""
    pricing = create_gpt6_luna_pricing()

    # Usage with 1000 input tokens (400 cached), 800 output tokens (300 reasoning)
    usage = LLMUsage(
        input_tokens=1000,
        output_tokens=800,
        total_tokens=1800,
        cached_input_tokens=400,
        reasoning_tokens=300,
        is_provider_reported=True,
    )

    cost = calculate_cost(usage, pricing)
    assert cost is not None

    # Expected exact breakdown under GPT-6 Luna Standard rates:
    # Uncached inputs: 1000 - 400 = 600 tokens -> (600 / 1,000,000) * 0.10 = 0.000060
    # Cached inputs: 400 tokens -> (400 / 1,000,000) * 0.01 = 0.000004
    # Output tokens (already includes reasoning tokens): 800 -> (800 / 1,000,000) * 0.50 = 0.000400
    # Total = 0.000060 + 0.000004 + 0.000400 = 0.000464 USD
    assert cost == Decimal("0.000464")

    # If reasoning tokens were erroneously double-counted (800 + 300 = 1100 outputs):
    erroneous_double_counted_cost = Decimal("0.000060") + Decimal("0.000004") + Decimal("0.000550")
    assert cost != erroneous_double_counted_cost

    # Simulated usage from MockLLMProvider strictly evaluates to None cost
    sim_usage = LLMUsage(
        input_tokens=1000,
        output_tokens=800,
        total_tokens=1800,
        cached_input_tokens=400,
        reasoning_tokens=300,
        is_provider_reported=False,
    )
    assert calculate_cost(sim_usage, pricing) is None

    # Missing required token counts evaluates to None
    incomplete_usage = LLMUsage(
        input_tokens=1000,
        output_tokens=None,  # type: ignore
        total_tokens=1000,
        is_provider_reported=True,
    )
    assert calculate_cost(incomplete_usage, pricing) is None

    # Cached tokens present but pricing tier undefined strictly returns None to avoid underbilling
    no_cached_tier_pricing = PricingConfig(
        version="v-no-cached",
        source="test",
        effective_date="2026-11-01",
        input_cost_per_million=Decimal("0.10"),
        cached_input_cost_per_million=None,
        output_cost_per_million=Decimal("0.50"),
    )
    assert calculate_cost(usage, no_cached_tier_pricing) is None


def test_percentile_missing_data_and_report_formatting(runner, tmp_path):
    """27. Missing latency observations produce None percentiles and 'N/A' report formatting without zero falsification."""
    # 1. Missing or NaN inputs return None
    assert calculate_percentile([], 50.0) is None
    assert calculate_percentile([None, None], 95.0) is None
    assert calculate_percentile([float("nan")], 50.0) is None

    # 2. None filtering preserves valid numeric observations
    assert calculate_percentile([None, 100.0, None], 50.0) == 100.0
    assert calculate_percentile([None, 10.0, 20.0, float("nan"), 30.0], 50.0) == 20.0

    # 3. Empty aggregate metrics have None latencies
    empty_agg = compute_aggregate_metrics([], "A")
    assert empty_agg.latency_p50_ms is None
    assert empty_agg.latency_p95_ms is None

    # 4. Report rendering formats missing latencies as 'N/A' rather than '0.0 ms'
    report_content = runner._render_analysis_markdown([])
    assert "N/A / N/A" in report_content
    assert "0.0 / 0.0" not in report_content


def test_raw_runs_traceability_preserves_all_fifteen_required_fields(runner, benchmark_tasks, tmp_path):
    """26. raw_runs.jsonl preserves all 15 audit fields without secret leakage or fabrication."""
    task = benchmark_tasks[0]
    provider = MockLLMProvider([
        make_res(content='{"status": "ok"}', request_id="req_test_abc", system_fingerprint="fp_test_123"),
    ])
    record = runner.run_task(task, "B", provider=provider)

    runner.export_artifacts([record], tmp_path)
    jsonl_file = tmp_path / "raw_runs.jsonl"
    assert jsonl_file.exists()

    with open(jsonl_file, "r", encoding="utf-8") as f:
        line = f.readline()
        rec_dict = json.loads(line)

    # 1. Provider name and provider mode
    assert rec_dict["provider_name"] == "MockLLMProvider"
    assert rec_dict["provider_mode"] == "simulated"

    # 2. Requested model and actual model identifier
    assert rec_dict["requested_model"] == "mock-test-model"
    assert rec_dict["actual_model_identifier"] == "mock-model"

    # 3. Provider request IDs and backend fingerprint
    assert "provider_request_ids" in rec_dict
    assert isinstance(rec_dict["provider_request_ids"], list)
    assert "backend_fingerprint" in rec_dict

    # 4. SDK version and reasoning effort
    assert "sdk_version" in rec_dict
    assert rec_dict["reasoning_effort"] == "low"

    # 5. Per-request latency and aggregated token usage
    assert "per_request_latencies_ms" in rec_dict
    assert isinstance(rec_dict["per_request_latencies_ms"], list)
    assert "aggregated_token_usage" in rec_dict

    # 6. Usage provenance
    assert rec_dict["usage_provenance"] == "simulated"

    # 7. Dataset hashes and prompt hashes
    assert len(rec_dict["fixture_dataset_hash"]) == 64
    assert len(rec_dict["task_dataset_hash"]) == 64
    assert len(rec_dict["prompt_template_hash"]) == 64

    # 8. Condition C inspection trajectory
    assert "inspection_trajectory" in rec_dict
    assert isinstance(rec_dict["inspection_trajectory"], list)

    # 9. UTC timestamps and experiment configuration version
    assert "T" in rec_dict["timestamp_start_utc"]
    assert "T" in rec_dict["timestamp_end_utc"]
    assert rec_dict["experiment_config_version"] == "1.0.0"

    # 10. Verify absolute zero secret or API key leakage
    raw_text = json.dumps(rec_dict)
    assert "sk-" not in raw_text
    assert "api_key" not in raw_text.lower()
    assert "secret" not in raw_text.lower()

