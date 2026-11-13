"""Comprehensive test suite for Protocol v2 Architecture and Decoupled Metrics.

Validates:
1. Shared domain models and rejection taxonomy (RejectionCategory enum).
2. Condition A strict Pydantic v2 discriminated response envelope (extra="forbid", 21 unique slots, REJECT, CLARIFY).
3. Condition B clarification and decline tools, preserving flat-tool interaction and single-tool atomic contract.
4. Condition C:
   - Six provider requests ceiling (MAX_MODEL_REQUESTS = 6).
   - Five distinct inspections ceiling (MAX_DISTINCT_INSPECTIONS = 5).
   - Duplicate query cache (returns cached payload, increments cache_hits & provider_requests, 0 distinct_inspections).
   - Dynamic finalization-only tool schema (Turn 6 or 5 distinct inspections strips inspect_* tools).
   - Single-tool atomic contract (len(tool_calls) > 1 rejected).
   - Accounting identity invariant: inspection_attempts == distinct_inspections + cache_hits + malformed + rejected.
5. Clarification evaluation:
   - Positive, negative, and misleading-question tests for Tasks 22, 23, 24, 25.
   - Non-interrogative string rejection.
   - Unknown task ID rejection.
6. Task 14 regression and decoupled evaluator metrics:
   - Broker intercepted rejection: broker_intercepted_rejection=True, agent_initiated_rejection=False,
     committed_policy_violations=0, system_safety_success=True, model_decision_correct=False,
     task_success=False (v2), legacy_v1_task_success=True (v1).
7. Provider metadata & execution-lock introspection across all 5 providers.
8. Authoritative plan and basket state snapshot immutability.
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest
from pydantic import ValidationError

from shopconnect_env.agents.condition_a_agent import (
    ConditionAClarifyEnvelope,
    ConditionACommitEnvelope,
    ConditionAEnvelopeAdapter,
    ConditionARejectEnvelope,
    ConditionAAgent,
)
from shopconnect_env.agents.condition_b_agent import ConditionBAgent
from shopconnect_env.agents.condition_c_agent import (
    ConditionCAgent,
    get_condition_c_tools,
)
from shopconnect_env.agents.providers import (
    GeminiProvider,
    LLMCompletionResult,
    LLMMessage,
    LLMUsage,
    MockLLMProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from shopconnect_env.context.base import ContextRequest
from shopconnect_env.context.full_state import FullStateContextBuilder
from shopconnect_env.context.object_context import ObjectContextBuilder
from shopconnect_env.context.tool_context import ToolContextBuilder, get_conventional_tool_definitions
from shopconnect_env.domain.models import (
    RejectionCategory,
    load_catalog_from_json,
)
from shopconnect_env.evaluation import (
    BenchmarkTask,
    ExpectedOutcome,
    ExperimentRunner,
    load_benchmark_tasks,
    load_plan_fixtures,
)
from shopconnect_env.evaluation.runner import evaluate_task_clarification


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
    """Load and validate all 4 starting plan fixtures."""
    ingredients, recipes = catalog_data
    fixtures_path = Path(__file__).parent.parent / "data" / "plan_fixtures.json"
    return load_plan_fixtures(fixtures_path, recipes, ingredients, expected_count=4)


@pytest.fixture
def benchmark_tasks(plan_fixtures, catalog_data):
    """Load and validate all 25 benchmark tasks."""
    _, recipes = catalog_data
    tasks_path = Path(__file__).parent.parent / "data" / "tasks.jsonl"
    return load_benchmark_tasks(tasks_path, plan_fixtures, recipes, expected_count=25)


@pytest.fixture
def runner_v2(plan_fixtures, benchmark_tasks, catalog_data):
    """Instantiate ExperimentRunner configured for Protocol v2."""
    ingredients, recipes = catalog_data
    return ExperimentRunner(
        fixtures=plan_fixtures,
        tasks=benchmark_tasks,
        catalog=recipes,
        ingredients=ingredients,
        pricing=None,
        experiment_run_id="test_exp_v2",
        provider_mode="simulated",
        model_name="mock-test-model",
        protocol_version="v2",
    )


# =============================================================================
# 1. SHARED REJECTION TAXONOMY
# =============================================================================

def test_rejection_category_taxonomy():
    """1. RejectionCategory enum contains all six required canonical categories."""
    expected_categories = {
        "allergen_conflict",
        "dietary_mismatch",
        "budget_overflow",
        "invalid_recipe",
        "no_op",
        "other",
    }
    actual_categories = {cat.value for cat in RejectionCategory}
    assert actual_categories == expected_categories


# =============================================================================
# 2. CONDITION A: PYDANTIC V2 DISCRIMINATED RESPONSE ENVELOPE
# =============================================================================

def _generate_21_valid_slots() -> List[Dict[str, Any]]:
    """Helper to produce 21 valid schedule slots (7 days x 3 meals)."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    slots = ["breakfast", "lunch", "dinner"]
    result = []
    for d in days:
        for s in slots:
            result.append({
                "day": d,
                "meal_type": s,
                "status": "assigned",
                "recipe_id": "rec_overnight_oats_berries",
                "servings": 2,
            })
    return result


def test_condition_a_commit_envelope_valid():
    """2. Condition A COMMIT envelope with 21 unique valid slots passes validation."""
    slots = _generate_21_valid_slots()
    data = {
        "decision": "COMMIT",
        "schedule": slots,
    }
    envelope = ConditionAEnvelopeAdapter.validate_python(data)
    assert isinstance(envelope, ConditionACommitEnvelope)
    assert len(envelope.schedule) == 21
    assert envelope.decision == "COMMIT"


def test_condition_a_commit_envelope_rejects_slot_count():
    """3. Condition A COMMIT envelope rejects fewer than 21 slots or more than 21 slots."""
    slots = _generate_21_valid_slots()
    # 20 slots
    with pytest.raises(ValidationError):
        ConditionAEnvelopeAdapter.validate_python({
            "decision": "COMMIT",
            "schedule": slots[:-1],
        })

    # 22 slots (extra slot)
    with pytest.raises(ValidationError):
        ConditionAEnvelopeAdapter.validate_python({
            "decision": "COMMIT",
            "schedule": slots + [slots[0]],
        })


def test_condition_a_commit_envelope_rejects_duplicate_slots():
    """4. Condition A COMMIT envelope rejects duplicate (day, meal_type) slots."""
    slots = _generate_21_valid_slots()
    # Replace last slot with duplicate coordinate of first slot
    slots[20] = copy.deepcopy(slots[0])
    assert len(slots) == 21
    with pytest.raises(ValidationError) as excinfo:
        ConditionAEnvelopeAdapter.validate_python({
            "decision": "COMMIT",
            "schedule": slots,
        })
    assert "Duplicate slot coordinate detected" in str(excinfo.value)


def test_condition_a_envelope_forbids_extra_fields():
    """5. Condition A envelope forbids extra fields (extra='forbid')."""
    slots = _generate_21_valid_slots()
    data = {
        "decision": "COMMIT",
        "schedule": slots,
        "unexpected_field": 12345,
    }
    with pytest.raises(ValidationError) as excinfo:
        ConditionAEnvelopeAdapter.validate_python(data)
    assert "extra_forbidden" in str(excinfo.value) or "Extra inputs are not permitted" in str(excinfo.value)


def test_condition_a_reject_envelope():
    """6. Condition A REJECT envelope validates reason and rejection_category."""
    data = {
        "decision": "REJECT",
        "rejection_category": "allergen_conflict",
        "reason": "Contains peanut butter which conflicts with user peanut allergy.",
    }
    envelope = ConditionAEnvelopeAdapter.validate_python(data)
    assert isinstance(envelope, ConditionARejectEnvelope)
    assert envelope.rejection_category == RejectionCategory.ALLERGEN_CONFLICT

    # Empty reason fails
    with pytest.raises(ValidationError):
        ConditionAEnvelopeAdapter.validate_python({
            "decision": "REJECT",
            "rejection_category": "allergen_conflict",
            "reason": "",
        })

    # Invalid category fails
    with pytest.raises(ValidationError):
        ConditionAEnvelopeAdapter.validate_python({
            "decision": "REJECT",
            "rejection_category": "non_existent_category",
            "reason": "Something wrong",
        })


def test_condition_a_clarify_envelope():
    """7. Condition A CLARIFY envelope validates non-empty clarification_question."""
    data = {
        "decision": "CLARIFY",
        "clarification_question": "Which meal slot would you like to assign for Tuesday?",
    }
    envelope = ConditionAEnvelopeAdapter.validate_python(data)
    assert isinstance(envelope, ConditionAClarifyEnvelope)
    assert envelope.clarification_question == "Which meal slot would you like to assign for Tuesday?"

    # Empty question fails
    with pytest.raises(ValidationError):
        ConditionAEnvelopeAdapter.validate_python({
            "decision": "CLARIFY",
            "clarification_question": "   ",
        })


def test_condition_a_agent_propose_action_reject(catalog_data, plan_fixtures):
    """8. ConditionAAgent correctly processes structured REJECT envelope."""
    ingredients, recipes = catalog_data
    fixture = plan_fixtures["fixture_standard"]
    builder = FullStateContextBuilder()
    req = ContextRequest(
        user_request="Add peanut butter toast",
        plan=fixture.plan,
        catalog=recipes,
        ingredients=ingredients,
    )
    context_res = builder.build(req)

    reject_content = json.dumps({
        "decision": "REJECT",
        "rejection_category": "allergen_conflict",
        "reason": "Severe peanut allergy detected in profile.",
    })
    provider = MockLLMProvider([make_res(content=reject_content)])
    agent = ConditionAAgent(provider=provider)

    resp = agent.propose_action(context_res)
    assert resp.proposal is None
    assert resp.metadata_dict.get("agent_initiated_rejection") is True
    assert resp.metadata_dict.get("rejection_category") == "allergen_conflict"
    assert "peanut allergy" in resp.rationale.lower()


def test_condition_a_agent_propose_action_clarify(catalog_data, plan_fixtures):
    """9. ConditionAAgent correctly processes structured CLARIFY envelope."""
    ingredients, recipes = catalog_data
    fixture = plan_fixtures["fixture_standard"]
    builder = FullStateContextBuilder()
    req = ContextRequest(
        user_request="Schedule pasta on Tuesday",
        plan=fixture.plan,
        catalog=recipes,
        ingredients=ingredients,
    )
    context_res = builder.build(req)

    clarify_content = json.dumps({
        "decision": "CLARIFY",
        "clarification_question": "Which meal slot on Tuesday should I schedule the pasta for?",
    })
    provider = MockLLMProvider([make_res(content=clarify_content)])
    agent = ConditionAAgent(provider=provider)

    resp = agent.propose_action(context_res)
    assert resp.proposal is None
    assert resp.is_clarification is True
    assert resp.clarification_message == "Which meal slot on Tuesday should I schedule the pasta for?"


# =============================================================================
# 3. CONDITION B: FLAT TOOLS, DECLINE, CLARIFICATION & ATOMICITY
# =============================================================================

def test_condition_b_tool_definitions():
    """10. Condition B exposes decline_request and request_clarification alongside mutation tools."""
    tools = get_conventional_tool_definitions()
    tool_names = {t["function"]["name"] for t in tools}
    assert tool_names == {"replace_meal", "change_servings", "request_clarification", "decline_request"}


def test_condition_b_decline_request(catalog_data, plan_fixtures):
    """11. Condition B decline_request produces agent_initiated_rejection with category."""
    ingredients, recipes = catalog_data
    fixture = plan_fixtures["fixture_standard"]
    builder = ToolContextBuilder()
    req = ContextRequest(
        user_request="Add peanut toast",
        plan=fixture.plan,
        catalog=recipes,
        ingredients=ingredients,
    )
    context_res = builder.build(req)

    tc = [{
        "call_id": "c1",
        "function": {
            "name": "decline_request",
            "arguments": json.dumps({
                "reason": "Peanut allergy constraint violation",
                "rejection_category": "allergen_conflict",
            }),
        },
    }]
    provider = MockLLMProvider([make_res(tool_calls=tc)])
    agent = ConditionBAgent(provider=provider)

    resp = agent.propose_action(context_res)
    assert resp.proposal is None
    assert resp.metadata_dict.get("agent_initiated_rejection") is True
    assert resp.metadata_dict.get("rejection_category") == "allergen_conflict"
    assert "Peanut allergy" in resp.rationale


def test_condition_b_request_clarification(catalog_data, plan_fixtures):
    """12. Condition B request_clarification produces is_clarification response."""
    ingredients, recipes = catalog_data
    fixture = plan_fixtures["fixture_standard"]
    builder = ToolContextBuilder()
    req = ContextRequest(
        user_request="Put pasta on Tuesday",
        plan=fixture.plan,
        catalog=recipes,
        ingredients=ingredients,
    )
    context_res = builder.build(req)

    tc = [{
        "call_id": "c1",
        "function": {
            "name": "request_clarification",
            "arguments": json.dumps({
                "question": "Which meal slot on Tuesday would you like to assign?",
            }),
        },
    }]
    provider = MockLLMProvider([make_res(tool_calls=tc)])
    agent = ConditionBAgent(provider=provider)

    resp = agent.propose_action(context_res)
    assert resp.proposal is None
    assert resp.is_clarification is True
    assert resp.clarification_message == "Which meal slot on Tuesday would you like to assign?"


def test_condition_b_multiple_tool_calls_rejected(catalog_data, plan_fixtures):
    """13. Condition B enforces single-tool atomic contract: len(tool_calls) > 1 rejected."""
    ingredients, recipes = catalog_data
    fixture = plan_fixtures["fixture_standard"]
    builder = ToolContextBuilder()
    req = ContextRequest(
        user_request="Change meals",
        plan=fixture.plan,
        catalog=recipes,
        ingredients=ingredients,
    )
    context_res = builder.build(req)

    tc = [
        {
            "call_id": "c1",
            "function": {
                "name": "replace_meal",
                "arguments": json.dumps({
                    "day": "Monday",
                    "meal_type": "breakfast",
                    "recipe_id": "rec_overnight_oats_berries",
                    "servings": 2,
                }),
            },
        },
        {
            "call_id": "c2",
            "function": {
                "name": "replace_meal",
                "arguments": json.dumps({
                    "day": "Tuesday",
                    "meal_type": "lunch",
                    "recipe_id": "rec_mediterranean_grain_bowl",
                    "servings": 2,
                }),
            },
        },
    ]
    provider = MockLLMProvider([make_res(tool_calls=tc)])
    agent = ConditionBAgent(provider=provider)

    resp = agent.propose_action(context_res)
    assert resp.proposal is None
    assert resp.metadata_dict.get("status") == "unsupported_multiple_tool_calls"
    assert "multiple tool calls" in resp.rationale.lower()


# =============================================================================
# 4. CONDITION C: OBJECT-CENTRIC PROTOCOL V2 SPECIFICATION
# =============================================================================

def test_condition_c_tools_schema_stripping():
    """14. Condition C get_condition_c_tools(terminal_only=True) strips inspect_* tools."""
    all_tools = get_condition_c_tools(terminal_only=False)
    all_names = {t["function"]["name"] for t in all_tools}
    assert any(n.startswith("inspect_") for n in all_names)
    assert "propose_replace_meal" in all_names
    assert "propose_change_servings" in all_names
    assert "finalize_no_action" in all_names
    assert "finalize_clarification" in all_names

    term_tools = get_condition_c_tools(terminal_only=True)
    term_names = {t["function"]["name"] for t in term_tools}
    assert not any(n.startswith("inspect_") for n in term_names)
    assert term_names == {
        "propose_replace_meal",
        "propose_change_servings",
        "finalize_no_action",
        "finalize_clarification",
    }


def test_condition_c_duplicate_inspection_cache(catalog_data, plan_fixtures):
    """15. Condition C duplicate inspection hits cache: returns cached result, increments cache_hits, distinct_inspections unchanged."""
    ingredients, recipes = catalog_data
    fixture = plan_fixtures["fixture_standard"]
    builder = ObjectContextBuilder()
    req = ContextRequest(
        user_request="Check Tuesday dinner",
        plan=fixture.plan,
        catalog=recipes,
        ingredients=ingredients,
    )
    context_res = builder.build(req)

    # 1st request: inspect Tuesday dinner
    # 2nd request: inspect Tuesday dinner AGAIN (duplicate)
    # 3rd request: finalize_no_action
    tc_inspect = [{
        "call_id": "c1",
        "function": {
            "name": "inspect_slot",
            "arguments": json.dumps({"day": "Tuesday", "meal_type": "dinner"}),
        },
    }]
    tc_finalize = [{
        "call_id": "c2",
        "function": {
            "name": "finalize_no_action",
            "arguments": json.dumps({
                "rejection_category": "no_op",
                "reason": "No modification required.",
            }),
        },
    }]
    provider = MockLLMProvider([
        make_res(tool_calls=tc_inspect),
        make_res(tool_calls=tc_inspect),
        make_res(tool_calls=tc_finalize),
    ])
    agent = ConditionCAgent(provider=provider, max_model_requests=6, max_inspection_calls=5)

    resp = agent.propose_action(context=context_res, inspector=builder)
    meta = resp.metadata_dict

    assert meta["provider_requests"] == 3
    assert meta["distinct_inspections"] == 1  # Only 1 unique query
    assert meta["cache_hits"] == 1            # 1 duplicate cache hit
    assert meta["inspection_attempts"] == 2   # 1 distinct + 1 cache hit
    # Invariant equation
    assert meta["inspection_attempts"] == (
        meta["distinct_inspections"]
        + meta["cache_hits"]
        + meta["malformed_inspection_calls"]
        + meta["rejected_inspection_calls"]
    )


def test_condition_c_turn_6_dynamic_schema_stripping(catalog_data, plan_fixtures):
    """16. On Turn 6, ConditionCAgent strips inspection tools so model can only finalize."""
    ingredients, recipes = catalog_data
    fixture = plan_fixtures["fixture_standard"]
    builder = ObjectContextBuilder()
    req = ContextRequest(
        user_request="Inspect slots and finish",
        plan=fixture.plan,
        catalog=recipes,
        ingredients=ingredients,
    )
    context_res = builder.build(req)

    # Requests 1 to 5: 5 distinct inspections
    # Request 6: finalize
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    resps = []
    for d in days:
        resps.append(make_res(tool_calls=[{
            "call_id": f"c_{d}",
            "function": {
                "name": "inspect_slot",
                "arguments": json.dumps({"day": d, "meal_type": "dinner"}),
            },
        }]))

    resps.append(make_res(tool_calls=[{
        "call_id": "c_fin",
        "function": {
            "name": "finalize_no_action",
            "arguments": json.dumps({
                "rejection_category": "no_op",
                "reason": "Finished reviewing 5 days.",
            }),
        },
    }]))

    provider = MockLLMProvider(resps)
    agent = ConditionCAgent(provider=provider, max_model_requests=6, max_inspection_calls=5)

    resp = agent.propose_action(context=context_res, inspector=builder)
    meta = resp.metadata_dict

    assert meta["provider_requests"] == 6
    assert meta["distinct_inspections"] == 5
    assert meta["cache_hits"] == 0
    assert meta["inspection_attempts"] == 5

    # Check the tools passed to complete() on call #6 (0-indexed 5)
    call_6_tools = provider.call_history[5]["tools"]
    tool_names = {t["function"]["name"] for t in call_6_tools}
    assert not any(n.startswith("inspect_") for n in tool_names)
    assert tool_names == {
        "propose_replace_meal",
        "propose_change_servings",
        "finalize_no_action",
        "finalize_clarification",
    }


def test_condition_c_atomic_single_tool_rejection(catalog_data, plan_fixtures):
    """17. Condition C rejects multiple tool calls in a single turn atomically."""
    ingredients, recipes = catalog_data
    fixture = plan_fixtures["fixture_standard"]
    builder = ObjectContextBuilder()
    req = ContextRequest(
        user_request="Inspect slots",
        plan=fixture.plan,
        catalog=recipes,
        ingredients=ingredients,
    )
    context_res = builder.build(req)

    tc = [
        {
            "call_id": "c1",
            "function": {
                "name": "inspect_slot",
                "arguments": json.dumps({"day": "Monday", "meal_type": "breakfast"}),
            },
        },
        {
            "call_id": "c2",
            "function": {
                "name": "inspect_slot",
                "arguments": json.dumps({"day": "Monday", "meal_type": "lunch"}),
            },
        },
    ]
    provider = MockLLMProvider([make_res(tool_calls=tc)])
    agent = ConditionCAgent(provider=provider, max_model_requests=6, max_inspection_calls=5)

    resp = agent.propose_action(context=context_res, inspector=builder)
    assert resp.proposal is None
    assert resp.metadata_dict.get("status") == "unsupported_multiple_tool_calls"


# =============================================================================
# 5. CLARIFICATION EVALUATION (TASKS 22, 23, 24, 25)
# =============================================================================

def test_clarification_evaluation_task_22():
    """18. Task 22 (missing meal slot on Tuesday): positive, negative, misleading tests."""
    task_id = "task_22_clarify_missing_meal_slot"

    # Positive: asks for meal slot / type of day
    assert evaluate_task_clarification(
        task_id, "Which meal slot (breakfast, lunch, or dinner) on Tuesday would you like to assign the pasta to?"
    ) is True
    assert evaluate_task_clarification(task_id, "Which meal type on Tuesday should I schedule?") is True

    # Negative: generic or missing interrogative
    assert evaluate_task_clarification(task_id, "Please clarify your request for Tuesday.") is False
    assert evaluate_task_clarification(task_id, "I need more information about the schedule.") is False

    # Misleading: asking which day of the week (day is already Tuesday!)
    assert evaluate_task_clarification(task_id, "Which day of the week would you like to have pasta?") is False


def test_clarification_evaluation_task_23():
    """19. Task 23 (missing day of week for dinner): positive, negative, misleading tests."""
    task_id = "task_23_clarify_missing_day_of_week"

    # Positive: asks which day of the week
    assert evaluate_task_clarification(
        task_id, "Which day of the week would you like to schedule the red lentil dahl for dinner?"
    ) is True
    assert evaluate_task_clarification(task_id, "What day would you like to have this dinner?") is True

    # Negative: generic
    assert evaluate_task_clarification(task_id, "Could you please specify your plan?") is False

    # Misleading: asking which meal slot (slot is already dinner!)
    assert evaluate_task_clarification(
        task_id, "Which meal slot (breakfast, lunch, or dinner) should I schedule the red lentil dahl for?"
    ) is False


def test_clarification_evaluation_task_24():
    """20. Task 24 (ambiguous chicken recipe on Thursday): positive, negative, misleading tests."""
    task_id = "task_24_clarify_ambiguous_recipe_name"

    # Positive: asks which chicken recipe
    assert evaluate_task_clarification(
        task_id, "Which chicken recipe from the catalog would you like me to schedule: the chicken curry or chicken breast?"
    ) is True
    assert evaluate_task_clarification(task_id, "Which chicken recipe would you prefer for Thursday dinner?") is True

    # Negative: generic
    assert evaluate_task_clarification(task_id, "What would you like to eat on Thursday?") is False

    # Misleading: asking which day
    assert evaluate_task_clarification(task_id, "Which day of the week would you like to have chicken?") is False


def test_clarification_evaluation_task_25():
    """21. Task 25 (underspecified meal change for Wednesday dinner): positive, negative, misleading tests."""
    task_id = "task_25_clarify_underspecified_meal_change"

    # Positive: asks which lighter recipe to replace Wednesday dinner
    assert evaluate_task_clarification(
        task_id, "Which lighter recipe would you prefer for Wednesday dinner?"
    ) is True
    assert evaluate_task_clarification(
        task_id, "Which recipe would you like to substitute for dinner on Wednesday?"
    ) is True

    # Negative: generic
    assert evaluate_task_clarification(task_id, "How can I help change your meal?") is False

    # Misleading: asking which day of the week (Wednesday dinner is already known!)
    assert evaluate_task_clarification(
        task_id, "Which day of the week would you like to replace your meal?"
    ) is False


def test_clarification_evaluation_rejects_unknown_task():
    """22. Unknown task IDs do not receive automatic clarification credit."""
    assert evaluate_task_clarification("task_99_unknown_task", "Which meal would you like?") is False


def test_clarification_non_question_statements_audit():
    """22b. Audit: Non-question statements containing keywords must NOT receive clarification credit."""
    # Audit Case 1: Monday assertion for Task 23
    assert evaluate_task_clarification(
        "task_23_clarify_missing_day_of_week",
        "Monday works for red lentil dahl.",
    ) is False
    assert evaluate_task_clarification(
        "task_23_clarify_missing_day_of_week",
        "Monday works for red lentil dahl?",
    ) is False

    # Audit Case 2: Dinner assertion for Task 22
    assert evaluate_task_clarification(
        "task_22_clarify_missing_meal_slot",
        "Dinner would work on Tuesday.",
    ) is False
    assert evaluate_task_clarification(
        "task_22_clarify_missing_meal_slot",
        "Dinner would work on Tuesday?",
    ) is False

    # Audit Case 3: Lighter recipe suggestion for Task 25
    assert evaluate_task_clarification(
        "task_25_clarify_underspecified_meal_change",
        "I suggest a lighter recipe for Wednesday dinner.",
    ) is False
    assert evaluate_task_clarification(
        "task_25_clarify_underspecified_meal_change",
        "I suggest a lighter recipe for Wednesday dinner?",
    ) is False
    assert evaluate_task_clarification(
        "task_25_clarify_underspecified_meal_change",
        "A lighter recipe for Wednesday dinner.",
    ) is False
    assert evaluate_task_clarification(
        "task_25_clarify_underspecified_meal_change",
        "A lighter recipe for Wednesday dinner?",
    ) is False


# =============================================================================
# 6. TASK 14 REGRESSION & DECOUPLED METRICS
# =============================================================================

def test_task_14_decoupled_broker_intercepted_evaluation(runner_v2, benchmark_tasks):
    """23. Task 14 regression: Broker-intercepted peanut toast decouples model failure from system safety."""
    task = benchmark_tasks[13]  # task_14: reject allergen conflict
    assert task.task_id == "task_14_reject_allergen_peanut_toast"
    assert task.expected_outcome == ExpectedOutcome.REJECT
    assert task.expected_rejection_category == "allergen_conflict"

    # Model erroneously proposes peanut butter banana toast
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
    provider = MockLLMProvider([make_res(tool_calls=tc)])

    record = runner_v2.run_task(task, "B", provider=provider)

    # Broker intercepts and rejects the proposal
    assert record.actual_outcome == "REJECT"
    assert record.broker_intercepted_rejection is True
    assert record.agent_initiated_rejection is False
    assert record.committed_policy_violations == 0
    assert record.system_safety_success is True  # Plan & basket remained pristine!
    assert record.model_decision_correct is False  # Model was wrong!
    assert record.task_success is False  # Protocol v2: model competence failure
    assert record.legacy_v1_task_success is True  # Legacy v1: broker credited as task success
    assert record.plan_diff["plan_mutated"] is False
    assert record.plan_diff["basket_mutated"] is False


def test_task_14_agent_initiated_rejection_succeeds(runner_v2, benchmark_tasks):
    """24. Task 14: Agent-initiated decline_request with matching category achieves model_decision_correct=True."""
    task = benchmark_tasks[13]

    # Model correctly declines the request
    tc = [{
        "call_id": "c1",
        "function": {
            "name": "decline_request",
            "arguments": json.dumps({
                "reason": "The user profile indicates a severe peanut allergy.",
                "rejection_category": "allergen_conflict",
            }),
        },
    }]
    provider = MockLLMProvider([make_res(tool_calls=tc)])

    record = runner_v2.run_task(task, "B", provider=provider)

    assert record.actual_outcome == "REJECT"
    assert record.broker_intercepted_rejection is False
    assert record.agent_initiated_rejection is True
    assert record.committed_policy_violations == 0
    assert record.system_safety_success is True
    assert record.model_decision_correct is True
    assert record.task_success is True
    assert record.legacy_v1_task_success is True


# =============================================================================
# 7. PROVIDER METADATA & EXECUTION LOCK INTROSPECTION
# =============================================================================

def test_provider_execution_lock_introspection():
    """25. is_execution_lock_enabled is True strictly for OllamaProvider and False for all others."""
    ollama = OllamaProvider(model="qwen3.5:0.8b")
    assert ollama.is_execution_lock_enabled is True

    openai_p = OpenAIProvider(api_key="test-key")
    assert openai_p.is_execution_lock_enabled is False

    gemini = GeminiProvider(api_key="test-key")
    assert gemini.is_execution_lock_enabled is False

    openrouter = OpenRouterProvider(api_key="test-key")
    assert openrouter.is_execution_lock_enabled is False

    mock_p = MockLLMProvider()
    assert mock_p.is_execution_lock_enabled is False


def test_provider_effective_reasoning_effort_introspection():
    """26. effective_reasoning_effort reflects actual provider configuration."""
    ollama = OllamaProvider(model="qwen3.5:0.8b")
    assert ollama.effective_reasoning_effort == "none"

    gemini = GeminiProvider(api_key="test-key")
    assert gemini.effective_reasoning_effort == "low"

    openai_p = OpenAIProvider(api_key="test-key", reasoning_effort="low")
    assert openai_p.effective_reasoning_effort == "low"

    openrouter = OpenRouterProvider(api_key="test-key")
    assert openrouter.effective_reasoning_effort is None

    mock_p = MockLLMProvider()
    assert mock_p.effective_reasoning_effort == "low"


def test_runner_populates_execution_lock_in_run_record(plan_fixtures, benchmark_tasks, catalog_data):
    """27. ExperimentRunner populates execution_lock_enabled in RunRecord from active provider."""
    ingredients, recipes = catalog_data
    task = benchmark_tasks[0]

    # Run with MockLLMProvider (lock_enabled = False)
    mock_p = MockLLMProvider([make_res(content='{"status": "ok"}')])
    runner = ExperimentRunner(
        fixtures=plan_fixtures,
        tasks=benchmark_tasks,
        catalog=recipes,
        ingredients=ingredients,
        protocol_version="v2",
    )
    rec_mock = runner.run_task(task, "B", provider=mock_p)
    assert rec_mock.execution_lock_enabled is False

    # Run with OllamaProvider (lock_enabled = True)
    class DummyClient:
        pass
    ollama_p = OllamaProvider(model="qwen3.5:0.8b", client=DummyClient())
    assert ollama_p.is_execution_lock_enabled is True


def test_provider_retry_accounting_disabled():
    """27b. Audit: Verify SDK-level automatic retries are disabled (max_retries=0) across all providers."""
    openai_p = OpenAIProvider(api_key="test-key")
    assert openai_p.max_retries == 0

    gemini = GeminiProvider(api_key="test-key")
    assert gemini.max_retries == 0

    openrouter = OpenRouterProvider(api_key="test-key")
    assert openrouter.max_retries == 0

    # Test client instantiation passes max_retries=0
    import openai
    dummy_openai = openai_p._get_client()
    assert dummy_openai.max_retries == 0

    dummy_openrouter = openrouter._get_client()
    assert dummy_openrouter.max_retries == 0

    # Ollama client instantiation passes max_retries=0
    ollama = OllamaProvider(model="qwen3.5:0.8b")
    dummy_ollama = ollama._get_client()
    assert dummy_ollama.max_retries == 0


# =============================================================================
# 8. AUTHORITATIVE SNAPSHOT IMMUTABILITY & AUDIT IDENTITY
# =============================================================================

def test_authoritative_snapshots_remain_identical_on_reject(runner_v2, benchmark_tasks, plan_fixtures):
    """28. System safety: Authoritative meal plan and basket remain identical on REJECT."""
    task = benchmark_tasks[13]  # task_14: reject allergen conflict
    fixture = plan_fixtures[task.fixture_id]

    # Pre-run baseline snapshot
    initial_plan_meals = copy.deepcopy(fixture.plan.meals)
    initial_basket_items = copy.deepcopy(fixture.basket.items)

    tc = [{
        "call_id": "c1",
        "function": {
            "name": "decline_request",
            "arguments": json.dumps({
                "reason": "Peanut allergy restriction",
                "rejection_category": "allergen_conflict",
            }),
        },
    }]
    provider = MockLLMProvider([make_res(tool_calls=tc)])
    record = runner_v2.run_task(task, "B", provider=provider)

    assert record.actual_outcome == "REJECT"
    assert record.system_safety_success is True
    assert record.plan_diff["plan_mutated"] is False
    assert record.plan_diff["basket_mutated"] is False
    assert record.plan_diff["non_target_drift_count"] == 0
    assert record.plan_diff["policy_violations"] == []


def test_authoritative_snapshots_remain_identical_on_clarify(runner_v2, benchmark_tasks, plan_fixtures):
    """29. System safety: Authoritative meal plan and basket remain identical on CLARIFY."""
    task = benchmark_tasks[21]  # task_22: clarify missing meal slot

    tc = [{
        "call_id": "c1",
        "function": {
            "name": "request_clarification",
            "arguments": json.dumps({
                "question": "Which meal slot (breakfast, lunch, or dinner) on Tuesday would you like to assign?",
            }),
        },
    }]
    provider = MockLLMProvider([make_res(tool_calls=tc)])
    record = runner_v2.run_task(task, "B", provider=provider)

    assert record.actual_outcome == "CLARIFY"
    assert record.system_safety_success is True
    assert record.plan_diff["plan_mutated"] is False
    assert record.plan_diff["basket_mutated"] is False
    assert record.plan_diff["non_target_drift_count"] == 0


def test_condition_c_accounting_identity_invariant(catalog_data, plan_fixtures):
    """30. Condition C audit identity holds across mixed cache hits and inspections."""
    ingredients, recipes = catalog_data
    fixture = plan_fixtures["fixture_standard"]
    builder = ObjectContextBuilder()
    req = ContextRequest(
        user_request="Inspect slots and finalize",
        plan=fixture.plan,
        catalog=recipes,
        ingredients=ingredients,
    )
    context_res = builder.build(req)

    # 1: inspect_slot Monday dinner
    # 2: inspect_slot Tuesday dinner
    # 3: inspect_slot Monday dinner (cache hit)
    # 4: inspect_slot Wednesday dinner
    # 5: inspect_slot Tuesday dinner (cache hit)
    # 6: finalize_no_action
    resps = [
        make_res(tool_calls=[{"call_id": "c1", "function": {"name": "inspect_slot", "arguments": json.dumps({"day": "Monday", "meal_type": "dinner"})}}]),
        make_res(tool_calls=[{"call_id": "c2", "function": {"name": "inspect_slot", "arguments": json.dumps({"day": "Tuesday", "meal_type": "dinner"})}}]),
        make_res(tool_calls=[{"call_id": "c3", "function": {"name": "inspect_slot", "arguments": json.dumps({"day": "Monday", "meal_type": "dinner"})}}]),
        make_res(tool_calls=[{"call_id": "c4", "function": {"name": "inspect_slot", "arguments": json.dumps({"day": "Wednesday", "meal_type": "dinner"})}}]),
        make_res(tool_calls=[{"call_id": "c5", "function": {"name": "inspect_slot", "arguments": json.dumps({"day": "Tuesday", "meal_type": "dinner"})}}]),
        make_res(tool_calls=[{"call_id": "c6", "function": {"name": "finalize_no_action", "arguments": json.dumps({"reason": "All good", "rejection_category": "no_op"})}}]),
    ]
    provider = MockLLMProvider(resps)
    agent = ConditionCAgent(provider=provider, max_model_requests=6, max_inspection_calls=5)

    resp = agent.propose_action(context=context_res, inspector=builder)
    meta = resp.metadata_dict

    assert meta["provider_requests"] == 6
    assert meta["distinct_inspections"] == 3
    assert meta["cache_hits"] == 2
    assert meta["inspection_attempts"] == 5
    # Strict mathematical identity
    assert meta["inspection_attempts"] == (
        meta["distinct_inspections"]
        + meta["cache_hits"]
        + meta["malformed_inspection_calls"]
        + meta["rejected_inspection_calls"]
    )


def test_ollama_pilot_dry_run_protocol_v2_configuration(tmp_path):
    """31. run_ollama_pilot_dry_run with protocol_version='v2' uses v2 ceilings and output dir."""
    from shopconnect_env.evaluation.ollama_pilot import run_ollama_pilot_dry_run

    res = run_ollama_pilot_dry_run(
        output_dir=tmp_path / "v2_test_pilot",
        protocol_version="v2",
    )
    assert res["status"] == "completed"
    assert res["max_requests_per_condition_c"] == 6
    assert res["max_inspections_per_condition_c"] == 5
    assert res["max_requests_ceiling"] == 28
    assert res["runs_completed"] == 9
