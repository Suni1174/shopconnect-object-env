"""Comprehensive unit and integration tests for Phase 6.2 Condition Agents:
- ConditionAAgent (Full-State Regeneration)
- ConditionBAgent (Conventional Flat Tool Calling)
- ConditionCAgent (Object-Centric Progressive Disclosure)

Verifies:
- All 3 agents use the LLMProvider abstraction and run 100% offline with MockLLMProvider.
- Condition A 21-slot representation, pre-sanitization drift detection, and ActionBroker translation.
- Condition B tool parsing, argument validation, multiple tool calls, and error handling.
- Condition C multi-turn dialogue, ObjectInspector integration, inspection and model-request limits.
- Symmetrical rollback: rejected actions leave plan version, scheduled meals, and basket untouched.
"""

import json
from decimal import Decimal
from pathlib import Path
import pytest

from shopconnect_env.agents.condition_a_agent import ConditionAAgent
from shopconnect_env.agents.condition_b_agent import ConditionBAgent
from shopconnect_env.agents.condition_c_agent import ConditionCAgent
from shopconnect_env.agents.providers import LLMCompletionResult, LLMMessage, LLMUsage, MockLLMProvider
from shopconnect_env.context.base import ContextRequest
from shopconnect_env.context.full_state import FullStateContextBuilder
from shopconnect_env.context.object_context import ObjectContextBuilder
from shopconnect_env.context.tool_context import ToolContextBuilder
from shopconnect_env.domain.models import (
    DAYS_OF_WEEK,
    MEAL_TYPES,
    Meal,
    MealPlan,
    UserPreferences,
    load_catalog_from_json,
)
from shopconnect_env.validation.broker import ActionBroker, ActionStatus
from shopconnect_env.validation.schemas import ActionProposal, OperationType


# =============================================================================
# FIXTURES & TEST HELPERS
# =============================================================================

@pytest.fixture
def catalog_data():
    """Load canonical ingredients and recipe catalog from seed JSON."""
    catalog_path = Path(__file__).parent.parent / "data" / "catalog_seed.json"
    return load_catalog_from_json(catalog_path)


@pytest.fixture
def test_plan():
    """Deterministic weekly meal plan with 4 scheduled meals."""
    plan = MealPlan(
        plan_id="plan_test_week",
        plan_version=1,
        preferences=UserPreferences(
            diet_tags=["vegetarian"],
            allergies=["peanuts"],
            weekly_budget=Decimal("150.00"),
            household_size=2,
        ),
    )
    plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_berry_oatmeal", servings=2))
    plan.set_meal(Meal(day="Monday", meal_type="lunch", recipe_id="rec_mediterranean_chickpea_salad", servings=2))
    plan.set_meal(Meal(day="Wednesday", meal_type="dinner", recipe_id="rec_red_lentil_dahl", servings=4))
    plan.set_meal(Meal(day="Friday", meal_type="dinner", recipe_id="rec_hearty_veggie_soup", servings=4))
    return plan


@pytest.fixture
def broker(test_plan, catalog_data):
    ingredients, recipes = catalog_data
    return ActionBroker(initial_plan=test_plan, catalog=recipes, ingredients=ingredients)


def make_21_slot_schedule(assigned_overrides=None):
    """Helper to generate a canonical 21-slot schedule with optional slot overrides."""
    overrides = assigned_overrides or {}
    schedule = []
    for d in DAYS_OF_WEEK:
        for m in MEAL_TYPES:
            coord = (d, m)
            if coord in overrides:
                rec_id, serv = overrides[coord]
                schedule.append({
                    "day": d,
                    "meal_type": m,
                    "status": "assigned",
                    "recipe_id": rec_id,
                    "servings": serv,
                })
            elif coord == ("Monday", "breakfast"):
                schedule.append({"day": d, "meal_type": m, "status": "assigned", "recipe_id": "rec_berry_oatmeal", "servings": 2})
            elif coord == ("Monday", "lunch"):
                schedule.append({"day": d, "meal_type": m, "status": "assigned", "recipe_id": "rec_mediterranean_chickpea_salad", "servings": 2})
            elif coord == ("Wednesday", "dinner"):
                schedule.append({"day": d, "meal_type": m, "status": "assigned", "recipe_id": "rec_red_lentil_dahl", "servings": 4})
            elif coord == ("Friday", "dinner"):
                schedule.append({"day": d, "meal_type": m, "status": "assigned", "recipe_id": "rec_hearty_veggie_soup", "servings": 4})
            else:
                schedule.append({"day": d, "meal_type": m, "status": "unfilled", "recipe_id": None, "servings": 0})
    return schedule


# =============================================================================
# 1. CONDITION A TESTS (FULL-STATE REGENERATION)
# =============================================================================

def test_condition_a_valid_recipe_replacement(catalog_data, broker):
    """Condition A successfully translates valid target recipe replacement and commits."""
    ingredients, recipes = catalog_data
    builder = FullStateContextBuilder()
    req = ContextRequest(
        user_request="Replace Wednesday dinner with rec_hearty_veggie_soup",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # Regenerated schedule changes Wednesday dinner to rec_hearty_veggie_soup (servings=4)
    regen_schedule = make_21_slot_schedule({("Wednesday", "dinner"): ("rec_hearty_veggie_soup", 4)})
    mock_provider = MockLLMProvider(
        default_response=LLMCompletionResult(
            message=LLMMessage(role="assistant", content=json.dumps({"schedule": regen_schedule})),
            usage=LLMUsage(input_tokens=800, output_tokens=400, total_tokens=1200, is_provider_reported=False),
            model_identifier="mock-a",
        )
    )

    agent = ConditionAAgent(provider=mock_provider)
    response = agent.propose_action(context)

    assert response.has_proposal is True
    assert response.proposal.operation == OperationType.REPLACE_MEAL
    assert response.proposal.arguments["recipe_id"] == "rec_hearty_veggie_soup"
    assert response.proposal.arguments["day"] == "Wednesday"
    assert response.proposal.arguments["meal_type"] == "dinner"
    assert response.metadata["unintended_mutations_count"] == 0

    result = broker.execute(response.proposal)
    assert result.status == ActionStatus.COMMITTED
    assert broker.get_plan().get_meal("Wednesday", "dinner").recipe_id == "rec_hearty_veggie_soup"
    assert broker.get_plan_version() == 2


def test_condition_a_valid_servings_change(catalog_data, broker):
    """Condition A detects single target servings change and emits CHANGE_SERVINGS proposal."""
    ingredients, recipes = catalog_data
    builder = FullStateContextBuilder()
    req = ContextRequest(
        user_request="Change servings for Monday breakfast to 4",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # Scale Monday breakfast servings from 2 to 4
    regen_schedule = make_21_slot_schedule({("Monday", "breakfast"): ("rec_berry_oatmeal", 4)})
    mock_provider = MockLLMProvider(
        default_response=LLMCompletionResult(
            message=LLMMessage(role="assistant", content=json.dumps(regen_schedule)),
            usage=LLMUsage(input_tokens=800, output_tokens=400, total_tokens=1200, is_provider_reported=False),
            model_identifier="mock-a",
        )
    )

    agent = ConditionAAgent(provider=mock_provider)
    response = agent.propose_action(context)

    assert response.has_proposal is True
    assert response.proposal.operation == OperationType.CHANGE_SERVINGS
    assert response.proposal.arguments["servings"] == 4
    assert response.metadata["unintended_mutations_count"] == 0

    result = broker.execute(response.proposal)
    assert result.status == ActionStatus.COMMITTED
    assert broker.get_plan().get_meal("Monday", "breakfast").servings == 4


def test_condition_a_noop_regeneration(catalog_data, broker):
    """When regenerated schedule is identical to original, agent rejects as target_not_modified."""
    ingredients, recipes = catalog_data
    builder = FullStateContextBuilder()
    req = ContextRequest(
        user_request="Replace Wednesday dinner with rec_hearty_veggie_soup",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # Identical schedule
    regen_schedule = make_21_slot_schedule()
    mock_provider = MockLLMProvider(
        default_response=LLMCompletionResult(
            message=LLMMessage(role="assistant", content=json.dumps(regen_schedule)),
            usage=None,
            model_identifier="mock-a",
        )
    )

    agent = ConditionAAgent(provider=mock_provider)
    response = agent.propose_action(context)

    assert response.has_proposal is False
    assert response.metadata["status"] == "target_not_modified"


def test_condition_a_invalid_slot_count(catalog_data, broker):
    """Condition A rejects schedules with invalid slot counts (e.g. 20 slots)."""
    ingredients, recipes = catalog_data
    builder = FullStateContextBuilder()
    req = ContextRequest(
        user_request="Replace Wednesday dinner",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # Drop one slot (20 slots instead of 21)
    short_schedule = make_21_slot_schedule()[:-1]
    mock_provider = MockLLMProvider(
        default_response=LLMCompletionResult(
            message=LLMMessage(role="assistant", content=json.dumps(short_schedule)),
            usage=None,
            model_identifier="mock-a",
        )
    )

    agent = ConditionAAgent(provider=mock_provider)
    response = agent.propose_action(context)

    assert response.has_proposal is False
    assert response.metadata["status"] == "slot_validation_error"
    assert "expected 21 slots" in response.metadata["error"]


def test_condition_a_duplicate_slot_coordinates(catalog_data, broker):
    """Condition A rejects schedules containing duplicate slot coordinates."""
    ingredients, recipes = catalog_data
    builder = FullStateContextBuilder()
    req = ContextRequest(
        user_request="Replace Wednesday dinner",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    dup_schedule = make_21_slot_schedule()
    # Duplicate Monday breakfast over Monday lunch
    dup_schedule[1] = dict(dup_schedule[0])

    mock_provider = MockLLMProvider(
        default_response=LLMCompletionResult(
            message=LLMMessage(role="assistant", content=json.dumps(dup_schedule)),
            usage=None,
            model_identifier="mock-a",
        )
    )

    agent = ConditionAAgent(provider=mock_provider)
    response = agent.propose_action(context)

    assert response.has_proposal is False
    assert response.metadata["status"] == "slot_validation_error"
    assert "Duplicate slot coordinate" in response.metadata["error"]


def test_condition_a_invalid_recipe_id(catalog_data, broker):
    """Condition A rejects schedules containing nonexistent recipe IDs."""
    ingredients, recipes = catalog_data
    builder = FullStateContextBuilder()
    req = ContextRequest(
        user_request="Replace Wednesday dinner",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    invalid_rec_schedule = make_21_slot_schedule({("Wednesday", "dinner"): ("rec_nonexistent_xyz", 2)})
    mock_provider = MockLLMProvider(
        default_response=LLMCompletionResult(
            message=LLMMessage(role="assistant", content=json.dumps(invalid_rec_schedule)),
            usage=None,
            model_identifier="mock-a",
        )
    )

    agent = ConditionAAgent(provider=mock_provider)
    response = agent.propose_action(context)

    assert response.has_proposal is False
    assert response.metadata["status"] == "slot_validation_error"
    assert "does not exist in catalog" in response.metadata["error"]


def test_condition_a_non_target_drift_rejection(catalog_data, broker):
    """Condition A records raw drift metrics and rejects candidate when non-target slots mutate."""
    ingredients, recipes = catalog_data
    builder = FullStateContextBuilder()
    req = ContextRequest(
        user_request="Replace Wednesday dinner with rec_hearty_veggie_soup",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # Target (Wednesday dinner) changed, BUT non-target (Monday lunch) was also mutated!
    drift_schedule = make_21_slot_schedule({
        ("Wednesday", "dinner"): ("rec_hearty_veggie_soup", 4),
        ("Monday", "lunch"): ("rec_black_bean_quinoa_bowl", 2),
    })

    mock_provider = MockLLMProvider(
        default_response=LLMCompletionResult(
            message=LLMMessage(role="assistant", content=json.dumps(drift_schedule)),
            usage=None,
            model_identifier="mock-a",
        )
    )

    agent = ConditionAAgent(provider=mock_provider)
    response = agent.propose_action(context)

    # Must reject immediately due to drift before reaching broker
    assert response.has_proposal is False
    assert response.metadata["status"] == "unintended_drift"
    assert response.metadata["unintended_mutations_count"] == 1
    assert ("Monday", "lunch") in response.metadata["unintended_slots"]

    # Verify execution via agent helper returns REJECTED and broker remains untouched
    result = agent.execute(context, broker)
    assert result.status == ActionStatus.REJECTED
    assert "Unintended state drift detected" in result.rejection_reason


def test_condition_a_rejection_leaves_plan_and_basket_unchanged(catalog_data, broker):
    """Assert symmetrical zero-mutation invariant: rejected Condition A candidate leaves state intact."""
    plan_before = broker.get_plan()
    basket_before = broker.get_basket()
    version_before = broker.get_plan_version()

    ingredients, recipes = catalog_data
    builder = FullStateContextBuilder()
    req = ContextRequest(
        user_request="Replace Wednesday dinner with rec_hearty_veggie_soup",
        plan=plan_before,
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # Malformed JSON
    mock_provider = MockLLMProvider(
        default_response=LLMCompletionResult(
            message=LLMMessage(role="assistant", content="NOT VALID JSON"),
            usage=None,
            model_identifier="mock-a",
        )
    )

    agent = ConditionAAgent(provider=mock_provider)
    result = agent.execute(context, broker)

    assert result.status == ActionStatus.REJECTED
    assert broker.get_plan_version() == version_before
    assert broker.get_plan().meals == plan_before.meals
    assert broker.get_basket().items == basket_before.items

def test_condition_b_valid_function_call(catalog_data, broker):
    """Condition B parses valid tool call into ActionProposal and commits cleanly."""
    ingredients, recipes = catalog_data
    builder = ToolContextBuilder()
    req = ContextRequest(
        user_request="Replace Wednesday dinner with rec_hearty_veggie_soup",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    tool_call = {
        "id": "call_123",
        "call_id": "call_123",
        "name": "replace_meal",
        "function": {
            "name": "replace_meal",
            "arguments": json.dumps({
                "day": "Wednesday",
                "meal_type": "dinner",
                "recipe_id": "rec_hearty_veggie_soup",
            }),
        },
    }

    mock_provider = MockLLMProvider(
        default_response=LLMCompletionResult(
            message=LLMMessage(role="assistant", tool_calls=(tool_call,)),
            usage=LLMUsage(input_tokens=250, output_tokens=35, total_tokens=285, is_provider_reported=False),
            model_identifier="mock-b",
        )
    )

    agent = ConditionBAgent(provider=mock_provider)
    response = agent.propose_action(context)

    assert response.has_proposal is True
    assert response.proposal.operation == OperationType.REPLACE_MEAL
    assert response.proposal.arguments["recipe_id"] == "rec_hearty_veggie_soup"
    assert response.metadata["tool_name"] == "replace_meal"
    assert response.metadata["call_id"] == "call_123"

    result = broker.execute(response.proposal)
    assert result.status == ActionStatus.COMMITTED
    assert broker.get_plan().get_meal("Wednesday", "dinner").recipe_id == "rec_hearty_veggie_soup"


def test_condition_b_invalid_function_arguments(catalog_data, broker):
    """Pydantic validation failure on invalid tool arguments produces safe rejection."""
    ingredients, recipes = catalog_data
    builder = ToolContextBuilder()
    req = ContextRequest(
        user_request="Change servings",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # Invalid servings count: 0 (must be > 0 and <= 20)
    bad_tool_call = {
        "id": "call_bad_args",
        "call_id": "call_bad_args",
        "name": "change_servings",
        "function": {
            "name": "change_servings",
            "arguments": json.dumps({"day": "Monday", "meal_type": "breakfast", "servings": 0}),
        },
    }

    mock_provider = MockLLMProvider(
        default_response=LLMCompletionResult(
            message=LLMMessage(role="assistant", tool_calls=(bad_tool_call,)),
            usage=None,
            model_identifier="mock-b",
        )
    )

    agent = ConditionBAgent(provider=mock_provider)
    response = agent.propose_action(context)

    assert response.has_proposal is False
    assert response.metadata["status"] == "invalid_arguments"
    assert "change_servings" in response.rationale


def test_condition_b_multiple_function_calls(catalog_data, broker):
    """Condition B records all returned function calls in metadata without silently dropping any."""
    ingredients, recipes = catalog_data
    builder = ToolContextBuilder()
    req = ContextRequest(
        user_request="Multiple actions",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    call_1 = {
        "id": "call_01",
        "call_id": "call_01",
        "name": "replace_meal",
        "function": {"name": "replace_meal", "arguments": json.dumps({"day": "Wednesday", "meal_type": "dinner", "recipe_id": "rec_hearty_veggie_soup"})},
    }
    call_2 = {
        "id": "call_02",
        "call_id": "call_02",
        "name": "change_servings",
        "function": {"name": "change_servings", "arguments": json.dumps({"day": "Monday", "meal_type": "breakfast", "servings": 4})},
    }

    mock_provider = MockLLMProvider(
        default_response=LLMCompletionResult(
            message=LLMMessage(role="assistant", tool_calls=(call_1, call_2)),
            usage=None,
            model_identifier="mock-b",
        )
    )

    plan_before = broker.get_plan()
    basket_before = broker.get_basket()
    version_before = broker.get_plan_version()

    agent = ConditionBAgent(provider=mock_provider)
    response = agent.propose_action(context)

    # Multi-action responses are explicitly rejected under atomic broker contract
    assert response.has_proposal is False
    assert response.metadata["status"] == "unsupported_multiple_tool_calls"
    assert response.metadata["tool_calls_count"] == 2
    assert len(response.metadata["all_tool_calls"]) == 2
    assert response.metadata["all_tool_calls"][0]["call_id"] == "call_01"
    assert response.metadata["all_tool_calls"][1]["call_id"] == "call_02"
    assert "Multiple tool calls" in response.rationale

    # Verify execution explicitly returns REJECTED and broker state remains untouched
    result = agent.execute(context, broker)
    assert result.status == ActionStatus.REJECTED
    assert "Multi-action responses are unsupported" in result.rejection_reason
    assert broker.get_plan_version() == version_before
    assert broker.get_plan().meals == plan_before.meals
    assert broker.get_basket().items == basket_before.items


def test_condition_b_unknown_tool_name(catalog_data, broker):
    """Condition B rejects unauthorized or unknown tool names safely."""
    ingredients, recipes = catalog_data
    builder = ToolContextBuilder()
    req = ContextRequest(
        user_request="Unknown tool",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    unknown_call = {
        "id": "call_unk",
        "call_id": "call_unk",
        "name": "delete_all_recipes",
        "function": {"name": "delete_all_recipes", "arguments": "{}"},
    }

    mock_provider = MockLLMProvider(
        default_response=LLMCompletionResult(
            message=LLMMessage(role="assistant", tool_calls=(unknown_call,)),
            usage=None,
            model_identifier="mock-b",
        )
    )

    agent = ConditionBAgent(provider=mock_provider)
    response = agent.propose_action(context)

    assert response.has_proposal is False
    assert response.metadata["status"] == "unknown_tool"


def test_condition_b_missing_or_malformed_tool_call_data(catalog_data, broker):
    """Malformed tool argument string is rejected cleanly without crashing."""
    ingredients, recipes = catalog_data
    builder = ToolContextBuilder()
    req = ContextRequest(
        user_request="Bad json",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    malformed_call = {
        "id": "call_bad_json",
        "call_id": "call_bad_json",
        "name": "replace_meal",
        "function": {"name": "replace_meal", "arguments": "{INVALID_JSON_CONTENT"},
    }

    mock_provider = MockLLMProvider(
        default_response=LLMCompletionResult(
            message=LLMMessage(role="assistant", tool_calls=(malformed_call,)),
            usage=None,
            model_identifier="mock-b",
        )
    )

    agent = ConditionBAgent(provider=mock_provider)
    response = agent.propose_action(context)

    assert response.has_proposal is False
    assert response.metadata["status"] == "malformed_tool_arguments"


def test_condition_b_broker_policy_rejection(catalog_data, broker):
    """Valid tool call violating household allergy is rejected by PolicyEngine with state intact."""
    plan_before = broker.get_plan()
    basket_before = broker.get_basket()
    version_before = broker.get_plan_version()

    # User preferences explicitly forbid peanuts
    assert "peanuts" in plan_before.preferences.allergies

    ingredients, recipes = catalog_data
    builder = ToolContextBuilder()
    req = ContextRequest(
        user_request="Propose peanut meal",
        plan=plan_before,
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # Propose peanut butter toast
    allergen_call = {
        "id": "call_allergen",
        "call_id": "call_allergen",
        "name": "replace_meal",
        "function": {
            "name": "replace_meal",
            "arguments": json.dumps({
                "day": "Monday",
                "meal_type": "breakfast",
                "recipe_id": "rec_peanut_butter_banana_toast",
            }),
        },
    }

    mock_provider = MockLLMProvider(
        default_response=LLMCompletionResult(
            message=LLMMessage(role="assistant", tool_calls=(allergen_call,)),
            usage=None,
            model_identifier="mock-b",
        )
    )

    agent = ConditionBAgent(provider=mock_provider)
    result = agent.execute(context, broker)

    assert result.status == ActionStatus.REJECTED
    assert "peanuts" in result.rejection_reason.lower()
    # Zero-mutation assertion
    assert broker.get_plan_version() == version_before
    assert broker.get_plan().meals == plan_before.meals
    assert broker.get_basket().items == basket_before.items


def test_condition_c_successful_one_inspection_interaction(catalog_data, broker):
    """Condition C performs 1 selective inspection followed by a valid proposal commit."""
    ingredients, recipes = catalog_data
    builder = ObjectContextBuilder()
    req = ContextRequest(
        user_request="Replace Wednesday dinner with a healthy vegetable soup",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # Turn 1: Inspect recipe rec_hearty_veggie_soup
    turn_1 = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            tool_calls=(
                {
                    "id": "call_insp_01",
                    "call_id": "call_insp_01",
                    "name": "inspect_recipe",
                    "function": {
                        "name": "inspect_recipe",
                        "arguments": json.dumps({"recipe_id": "rec_hearty_veggie_soup"}),
                    },
                },
            ),
        ),
        usage=LLMUsage(input_tokens=100, output_tokens=30, total_tokens=130, is_provider_reported=False),
        model_identifier="mock-c",
    )

    # Turn 2: Final proposal propose_replace_meal
    turn_2 = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            tool_calls=(
                {
                    "id": "call_prop_01",
                    "call_id": "call_prop_01",
                    "name": "propose_replace_meal",
                    "function": {
                        "name": "propose_replace_meal",
                        "arguments": json.dumps({
                            "day": "Wednesday",
                            "meal_type": "dinner",
                            "recipe_id": "rec_hearty_veggie_soup",
                        }),
                    },
                },
            ),
        ),
        usage=LLMUsage(input_tokens=150, output_tokens=40, total_tokens=190, is_provider_reported=False),
        model_identifier="mock-c",
    )

    mock_provider = MockLLMProvider(responses=[turn_1, turn_2])
    agent = ConditionCAgent(provider=mock_provider)

    # Pass the bound builder instance as inspector to preserve audit log
    response = agent.propose_action(context, inspector=builder)

    assert response.has_proposal is True
    assert response.proposal.operation == OperationType.REPLACE_MEAL
    assert response.proposal.arguments["recipe_id"] == "rec_hearty_veggie_soup"
    assert response.inspection_calls_made == 1
    assert response.metadata["total_model_requests"] == 2
    assert response.metadata["inspection_calls_count"] == 1

    # Verify bound builder recorded disclosure
    assert builder.get_disclosure_count() == 1
    log = builder.get_disclosure_log()
    assert log[0]["object_type"] == "recipe"
    assert log[0]["object_id"] == "rec_hearty_veggie_soup"

    result = broker.execute(response.proposal)
    assert result.status == ActionStatus.COMMITTED
    assert broker.get_plan().get_meal("Wednesday", "dinner").recipe_id == "rec_hearty_veggie_soup"


def test_condition_c_multiple_progressive_inspections(catalog_data, broker):
    """Condition C performs multiple progressive inspections before proposing an action."""
    ingredients, recipes = catalog_data
    builder = ObjectContextBuilder()
    req = ContextRequest(
        user_request="Replace Tuesday lunch",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # Turn 1: Inspect preferences
    t1 = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            tool_calls=({
                "id": "c1",
                "call_id": "c1",
                "name": "inspect_preferences",
                "function": {"name": "inspect_preferences", "arguments": "{}"},
            },),
        ),
        usage=None,
        model_identifier="mock-c",
    )
    # Turn 2: Inspect slot
    t2 = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            tool_calls=({
                "id": "c2",
                "call_id": "c2",
                "name": "inspect_slot",
                "function": {"name": "inspect_slot", "arguments": json.dumps({"day": "Monday", "meal_type": "lunch"})},
            },),
        ),
        usage=None,
        model_identifier="mock-c",
    )
    # Turn 3: Final proposal
    t3 = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            tool_calls=({
                "id": "c3",
                "call_id": "c3",
                "name": "propose_change_servings",
                "function": {"name": "propose_change_servings", "arguments": json.dumps({"day": "Monday", "meal_type": "lunch", "servings": 4})},
            },),
        ),
        usage=None,
        model_identifier="mock-c",
    )

    agent = ConditionCAgent(provider=MockLLMProvider(responses=[t1, t2, t3]))
    response = agent.propose_action(context, inspector=builder)

    assert response.has_proposal is True
    assert response.proposal.operation == OperationType.CHANGE_SERVINGS
    assert response.metadata["total_model_requests"] == 3
    assert response.metadata["inspection_calls_count"] == 2
    assert response.inspection_calls_made == 2


def test_condition_c_inspection_limit_enforcement(catalog_data, broker):
    """Condition C caps inspection calls at MAX_INSPECTION_CALLS = 5."""
    ingredients, recipes = catalog_data
    builder = ObjectContextBuilder()
    req = ContextRequest(
        user_request="Deep inspection",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # 6 sequential inspection attempts in a row
    calls = []
    for i in range(6):
        calls.append(
            LLMCompletionResult(
                message=LLMMessage(
                    role="assistant",
                    tool_calls=({
                        "id": f"call_{i}",
                        "call_id": f"call_{i}",
                        "name": "inspect_preferences",
                        "function": {"name": "inspect_preferences", "arguments": "{}"},
                    },),
                ),
                usage=None,
                model_identifier="mock-c",
            )
        )

    agent = ConditionCAgent(provider=MockLLMProvider(responses=calls))
    response = agent.propose_action(context, inspector=builder)

    # Exactly 5 inspections dispatched to ObjectInspector
    assert response.inspection_calls_made == 5
    assert response.metadata["inspection_calls_count"] == 5
    assert builder.get_disclosure_count() == 5


def test_condition_c_model_request_limit_enforcement(catalog_data, broker):
    """Condition C terminates safely when reaching MAX_MODEL_REQUESTS = 6 without proposal."""
    ingredients, recipes = catalog_data
    builder = ObjectContextBuilder()
    req = ContextRequest(
        user_request="Unfinished task",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # 6 responses that never submit a final action
    calls = []
    for i in range(6):
        calls.append(
            LLMCompletionResult(
                message=LLMMessage(
                    role="assistant",
                    tool_calls=({
                        "id": f"call_{i}",
                        "call_id": f"call_{i}",
                        "name": "inspect_preferences",
                        "function": {"name": "inspect_preferences", "arguments": "{}"},
                    },),
                ),
                usage=None,
                model_identifier="mock-c",
            )
        )

    agent = ConditionCAgent(provider=MockLLMProvider(responses=calls))
    response = agent.propose_action(context, inspector=builder)

    assert response.has_proposal is False
    assert response.metadata["total_model_requests"] == 6
    assert "MAX_MODEL_REQUESTS_EXCEEDED" in response.rationale


def test_condition_c_unknown_object_inspection(catalog_data, broker):
    """Inspecting an unknown object ID returns safe dictionary and does not crash agent."""
    ingredients, recipes = catalog_data
    builder = ObjectContextBuilder()
    req = ContextRequest(
        user_request="Unknown recipe query",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    turn_1 = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            tool_calls=({
                "id": "c_unk",
                "call_id": "c_unk",
                "name": "inspect_recipe",
                "function": {"name": "inspect_recipe", "arguments": json.dumps({"recipe_id": "rec_unknown_999"})},
            },),
        ),
        usage=None,
        model_identifier="mock-c",
    )
    turn_2 = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            tool_calls=({
                "id": "c_fin",
                "call_id": "c_fin",
                "name": "finalize_no_action",
                "function": {"name": "finalize_no_action", "arguments": json.dumps({"reason": "Recipe not found"})},
            },),
        ),
        usage=None,
        model_identifier="mock-c",
    )

    agent = ConditionCAgent(provider=MockLLMProvider(responses=[turn_1, turn_2]))
    response = agent.propose_action(context, inspector=builder)

    assert response.has_proposal is False
    assert response.rationale == "Recipe not found"
    assert response.inspection_calls_made == 1


def test_condition_c_rejected_proposal_leaves_plan_and_basket_unchanged(catalog_data, broker):
    """When a proposal from Condition C is rejected, state and basket remain 100% untouched."""
    plan_before = broker.get_plan()
    basket_before = broker.get_basket()
    version_before = broker.get_plan_version()

    ingredients, recipes = catalog_data
    builder = ObjectContextBuilder()
    req = ContextRequest(
        user_request="Allergen test",
        plan=plan_before,
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # Propose peanut butter toast (user is allergic to peanuts)
    turn_1 = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            tool_calls=({
                "id": "c_all",
                "call_id": "c_all",
                "name": "propose_replace_meal",
                "function": {
                    "name": "propose_replace_meal",
                    "arguments": json.dumps({
                        "day": "Monday",
                        "meal_type": "breakfast",
                        "recipe_id": "rec_peanut_butter_banana_toast",
                    }),
                },
            },),
        ),
        usage=None,
        model_identifier="mock-c",
    )

    agent = ConditionCAgent(provider=MockLLMProvider(responses=[turn_1]))
    result = agent.execute(context, broker, inspector=builder)

    assert result.status == ActionStatus.REJECTED
    assert "peanuts" in result.rejection_reason.lower()
    # Zero-mutation assertion
    assert broker.get_plan_version() == version_before
    assert broker.get_plan().meals == plan_before.meals
    assert broker.get_basket().items == basket_before.items


def test_condition_c_multiple_inspections_in_single_response(catalog_data, broker):
    """Condition C handles multiple inspection tool calls emitted within a single model response."""
    ingredients, recipes = catalog_data
    builder = ObjectContextBuilder()
    req = ContextRequest(
        user_request="Replace Wednesday dinner with hearty veggie soup",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # Turn 1: Assistant emits TWO inspection calls in a single response
    call_1 = {
        "id": "c_ins_1",
        "call_id": "c_ins_1",
        "name": "inspect_recipe",
        "function": {
            "name": "inspect_recipe",
            "arguments": json.dumps({"recipe_id": "rec_hearty_veggie_soup"}),
        },
    }
    call_2 = {
        "id": "c_ins_2",
        "call_id": "c_ins_2",
        "name": "inspect_preferences",
        "function": {
            "name": "inspect_preferences",
            "arguments": "{}",
        },
    }
    turn_1 = LLMCompletionResult(
        message=LLMMessage(role="assistant", tool_calls=(call_1, call_2)),
        usage=LLMUsage(input_tokens=100, output_tokens=30, total_tokens=130, is_provider_reported=True),
        model_identifier="mock-c",
    )

    # Turn 2: Assistant submits the final proposal
    turn_2 = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            tool_calls=({
                "id": "c_prop",
                "call_id": "c_prop",
                "name": "propose_replace_meal",
                "function": {
                    "name": "propose_replace_meal",
                    "arguments": json.dumps({
                        "day": "Wednesday",
                        "meal_type": "dinner",
                        "recipe_id": "rec_hearty_veggie_soup",
                    }),
                },
            },),
        ),
        usage=LLMUsage(input_tokens=150, output_tokens=40, total_tokens=190, is_provider_reported=True),
        model_identifier="mock-c",
    )

    agent = ConditionCAgent(provider=MockLLMProvider(responses=[turn_1, turn_2]))
    response = agent.propose_action(context, inspector=builder)

    # Both inspections were executed in 1 provider request
    assert response.has_proposal is True
    assert response.proposal.operation == OperationType.REPLACE_MEAL
    assert response.metadata["total_model_requests"] == 2
    assert response.metadata["inspection_calls_count"] == 2
    assert response.inspection_calls_made == 2
    assert builder.get_disclosure_count() == 2
    assert len(response.metadata["inspection_steps"]) == 2

    # Usage correctly aggregated across requests
    assert response.metadata["usage"] is not None
    assert response.metadata["usage"]["input_tokens"] == 250
    assert response.metadata["usage"]["output_tokens"] == 70
    assert response.metadata["usage"]["total_tokens"] == 320
    assert response.metadata["usage"]["is_provider_reported"] is True


def test_condition_c_multiple_inspections_ceiling_in_single_turn(catalog_data, broker):
    """Condition C enforces the 5-inspection ceiling even when multiple calls arrive in one turn."""
    ingredients, recipes = catalog_data
    builder = ObjectContextBuilder()
    req = ContextRequest(
        user_request="Deep inspection hitting ceiling",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # Turn 1: 3 inspection calls
    t1_calls = tuple({
        "id": f"c_t1_{i}",
        "call_id": f"c_t1_{i}",
        "name": "inspect_preferences",
        "function": {"name": "inspect_preferences", "arguments": "{}"},
    } for i in range(3))
    turn_1 = LLMCompletionResult(
        message=LLMMessage(role="assistant", tool_calls=t1_calls),
        usage=None,
        model_identifier="mock-c",
    )

    # Turn 2: 3 inspection calls (should execute 2, and reject the 3rd to stay <= 5)
    t2_calls = tuple({
        "id": f"c_t2_{i}",
        "call_id": f"c_t2_{i}",
        "name": "inspect_preferences",
        "function": {"name": "inspect_preferences", "arguments": "{}"},
    } for i in range(3))
    turn_2 = LLMCompletionResult(
        message=LLMMessage(role="assistant", tool_calls=t2_calls),
        usage=None,
        model_identifier="mock-c",
    )

    # Turn 3: Final proposal
    turn_3 = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            tool_calls=({
                "id": "c_final",
                "call_id": "c_final",
                "name": "propose_change_servings",
                "function": {
                    "name": "propose_change_servings",
                    "arguments": json.dumps({"day": "Monday", "meal_type": "breakfast", "servings": 3}),
                },
            },),
        ),
        usage=None,
        model_identifier="mock-c",
    )

    agent = ConditionCAgent(provider=MockLLMProvider(responses=[turn_1, turn_2, turn_3]))
    response = agent.propose_action(context, inspector=builder)

    # Exactly 5 inspections executed against ObjectInspector
    assert response.has_proposal is True
    assert response.metadata["total_model_requests"] == 3
    assert response.metadata["inspection_calls_count"] == 5
    assert response.inspection_calls_made == 5
    assert builder.get_disclosure_count() == 5


def test_condition_c_token_accounting_aggregation_and_provenance(catalog_data, broker):
    """Condition C accurately aggregates token consumption, preserves unknown fields, and respects provenance."""
    ingredients, recipes = catalog_data
    builder = ObjectContextBuilder()
    req = ContextRequest(
        user_request="Token accounting verification",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # Turn 1: Has input, output, cached, and reasoning tokens
    turn_1 = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            tool_calls=({
                "id": "c_tok_1",
                "call_id": "c_tok_1",
                "name": "inspect_preferences",
                "function": {"name": "inspect_preferences", "arguments": "{}"},
            },),
        ),
        usage=LLMUsage(
            input_tokens=100,
            output_tokens=30,
            total_tokens=130,
            cached_input_tokens=20,
            reasoning_tokens=10,
            is_provider_reported=True,
        ),
        model_identifier="mock-c",
    )

    # Turn 2: Missing optional cached and reasoning tokens (None)
    turn_2 = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            tool_calls=({
                "id": "c_tok_2",
                "call_id": "c_tok_2",
                "name": "propose_change_servings",
                "function": {
                    "name": "propose_change_servings",
                    "arguments": json.dumps({"day": "Monday", "meal_type": "breakfast", "servings": 2}),
                },
            },),
        ),
        usage=LLMUsage(
            input_tokens=80,
            output_tokens=40,
            total_tokens=120,
            cached_input_tokens=None,
            reasoning_tokens=None,
            is_provider_reported=True,
        ),
        model_identifier="mock-c",
    )

    agent = ConditionCAgent(provider=MockLLMProvider(responses=[turn_1, turn_2]))
    response = agent.propose_action(context, inspector=builder)

    usage_dict = response.metadata["usage"]
    assert usage_dict is not None
    assert usage_dict["input_tokens"] == 180
    assert usage_dict["output_tokens"] == 70
    assert usage_dict["total_tokens"] == 250
    assert usage_dict["cached_input_tokens"] == 20  # 20 + None -> 20
    assert usage_dict["reasoning_tokens"] == 10     # 10 + None -> 10
    assert usage_dict["is_provider_reported"] is True


def test_stale_plan_version_rejection_across_conditions(catalog_data, broker):
    """All three conditions safely reject proposals built against stale plan versions."""
    ingredients, recipes = catalog_data

    # 1. Snapshot state at plan_version = 1
    plan_v1 = broker.get_plan()
    version_before = broker.get_plan_version()
    assert version_before == 1

    # 2. Build contexts for A, B, and C at version 1
    ctx_a = FullStateContextBuilder().build(ContextRequest("Replace Wednesday dinner", plan_v1, recipes, ingredients))
    ctx_b = ToolContextBuilder().build(ContextRequest("Replace Wednesday dinner", plan_v1, recipes, ingredients))
    ctx_c = ObjectContextBuilder().build(ContextRequest("Replace Wednesday dinner", plan_v1, recipes, ingredients))

    # 3. Simulate another transaction committing first, bumping broker plan version to 2
    legit_mutation = ActionProposal(
        operation=OperationType.CHANGE_SERVINGS,
        arguments={"day": "Monday", "meal_type": "lunch", "servings": 4},
        reason="Independent concurrent update",
    )
    commit_res = broker.execute(legit_mutation)
    assert commit_res.status == ActionStatus.COMMITTED
    assert broker.get_plan_version() == 2

    plan_v2 = broker.get_plan()
    basket_v2 = broker.get_basket()

    # 4. Test Condition A stale rejection
    regen_schedule = make_21_slot_schedule({("Wednesday", "dinner"): ("rec_hearty_veggie_soup", 4)})
    agent_a = ConditionAAgent(
        provider=MockLLMProvider(
            default_response=LLMCompletionResult(
                message=LLMMessage(role="assistant", content=json.dumps(regen_schedule)),
                usage=None,
                model_identifier="mock-a",
            )
        )
    )
    res_a = agent_a.execute(ctx_a, broker)
    assert res_a.status == ActionStatus.REJECTED
    assert "stale plan version" in res_a.rejection_reason.lower()
    assert broker.get_plan_version() == 2
    assert broker.get_plan().meals == plan_v2.meals
    assert broker.get_basket().items == basket_v2.items

    # 5. Test Condition B stale rejection
    call_b = {
        "id": "c_b_stale",
        "call_id": "c_b_stale",
        "name": "replace_meal",
        "function": {"name": "replace_meal", "arguments": json.dumps({"day": "Wednesday", "meal_type": "dinner", "recipe_id": "rec_hearty_veggie_soup"})},
    }
    agent_b = ConditionBAgent(
        provider=MockLLMProvider(
            default_response=LLMCompletionResult(
                message=LLMMessage(role="assistant", tool_calls=(call_b,)),
                usage=None,
                model_identifier="mock-b",
            )
        )
    )
    res_b = agent_b.execute(ctx_b, broker)
    assert res_b.status == ActionStatus.REJECTED
    assert "stale plan version" in res_b.rejection_reason.lower()
    assert broker.get_plan_version() == 2
    assert broker.get_plan().meals == plan_v2.meals
    assert broker.get_basket().items == basket_v2.items

    # 6. Test Condition C stale rejection
    call_c = {
        "id": "c_c_stale",
        "call_id": "c_c_stale",
        "name": "propose_replace_meal",
        "function": {"name": "propose_replace_meal", "arguments": json.dumps({"day": "Wednesday", "meal_type": "dinner", "recipe_id": "rec_hearty_veggie_soup"})},
    }
    agent_c = ConditionCAgent(
        provider=MockLLMProvider(
            responses=[
                LLMCompletionResult(
                    message=LLMMessage(role="assistant", tool_calls=(call_c,)),
                    usage=None,
                    model_identifier="mock-c",
                )
            ]
        )
    )
    res_c = agent_c.execute(ctx_c, broker, inspector=ObjectContextBuilder())
    assert res_c.status == ActionStatus.REJECTED
    assert "stale plan version" in res_c.rejection_reason.lower()
    assert broker.get_plan_version() == 2
    assert broker.get_plan().meals == plan_v2.meals
    assert broker.get_basket().items == basket_v2.items
