"""Integration tests for Stage 5 Offline MockAgent and Agent Interface pipeline.

Verifies end-to-end data flow:
ContextBuilder (Conditions A, B, C) -> MockAgent (BaseAgent) -> ActionBroker.
Confirms deterministic proposals, progressive inspection recording, strict version
safety, semantic/policy rejection boundaries, and zero state drift.
"""

from decimal import Decimal
import json
from pathlib import Path
import pytest

from shopconnect_env.agents.base import AgentResponse, BaseAgent, ObjectInspector
from shopconnect_env.agents.mock_agent import MockAgent
from shopconnect_env.context.base import ContextRequest, ContextResult
from shopconnect_env.context.full_state import FullStateContextBuilder
from shopconnect_env.context.object_context import ObjectContextBuilder
from shopconnect_env.context.tool_context import ToolContextBuilder
from shopconnect_env.domain.models import (
    Meal,
    MealPlan,
    UserPreferences,
    load_catalog_from_json,
)
from shopconnect_env.validation.broker import ActionBroker
from shopconnect_env.validation.schemas import (
    ActionProposal,
    ActionStatus,
    OperationType,
)


@pytest.fixture
def catalog_data():
    """Load canonical ingredients and recipe catalog from seed JSON."""
    catalog_path = Path(__file__).parent.parent / "data" / "catalog_seed.json"
    ingredients, recipes = load_catalog_from_json(catalog_path)
    return ingredients, recipes


@pytest.fixture
def base_plan():
    """Create a baseline weekly MealPlan with several pre-populated meals."""
    plan = MealPlan(
        plan_id="plan_research_week_1",
        plan_version=1,
        preferences=UserPreferences(
            diet_tags=["vegetarian"],
            allergies=["peanuts"],
            weekly_budget=Decimal("75.00"),
            household_size=2,
        ),
    )
    plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_berry_oatmeal", servings=2))
    plan.set_meal(Meal(day="Monday", meal_type="dinner", recipe_id="rec_red_lentil_dahl", servings=4))
    plan.set_meal(Meal(day="Wednesday", meal_type="dinner", recipe_id="rec_hearty_veggie_soup", servings=4))
    plan.set_meal(Meal(day="Friday", meal_type="lunch", recipe_id="rec_mediterranean_chickpea_salad", servings=2))
    return plan


@pytest.fixture
def broker(base_plan, catalog_data):
    """Instantiate a Stage 3 ActionBroker with the baseline plan and catalog."""
    ingredients, recipes = catalog_data
    return ActionBroker(
        initial_plan=base_plan,
        catalog=recipes,
        ingredients=ingredients,
    )


# =============================================================================
# 1. INTERFACE & INSPECTOR VERIFICATION
# =============================================================================


def test_inspector_not_used_in_conditions_a_and_b(base_plan, catalog_data):
    """Conditions A and B must execute cleanly with inspector=None."""
    ingredients, recipes = catalog_data
    agent = MockAgent()
    req = ContextRequest(
        user_request="replace Wednesday dinner with rec_red_lentil_dahl",
        plan=base_plan,
        catalog=recipes,
        ingredients=ingredients,
    )

    # Condition A: Full state
    builder_a = FullStateContextBuilder()
    context_a = builder_a.build(req)
    resp_a = agent.propose_action(context_a, inspector=None)
    assert resp_a.has_proposal is True
    assert resp_a.inspection_calls_made == 0
    assert resp_a.proposal.arguments["recipe_id"] == "rec_red_lentil_dahl"

    # Condition B: Tool context
    builder_b = ToolContextBuilder()
    context_b = builder_b.build(req)
    resp_b = agent.propose_action(context_b, inspector=None)
    assert resp_b.has_proposal is True
    assert resp_b.inspection_calls_made == 0
    assert resp_b.proposal.arguments["recipe_id"] == "rec_red_lentil_dahl"


def test_condition_c_shares_builder_as_inspector_and_records_audit(base_plan, catalog_data):
    """Condition C passes the bound builder as inspector and records selective disclosure."""
    ingredients, recipes = catalog_data
    agent = MockAgent()
    builder_c = ObjectContextBuilder()

    req = ContextRequest(
        user_request="replace Wednesday dinner with rec_red_lentil_dahl",
        plan=base_plan,
        catalog=recipes,
        ingredients=ingredients,
        target_day="Wednesday",
        target_meal_type="dinner",
    )
    context_c = builder_c.build(req)

    # Verify builder_c satisfies ObjectInspector Protocol
    assert isinstance(builder_c, ObjectInspector)
    assert builder_c.get_disclosure_count() == 0

    # Agent proposes action while querying inspector
    resp_c = agent.propose_action(context_c, inspector=builder_c)

    assert resp_c.has_proposal is True
    assert resp_c.inspection_calls_made == 1
    assert resp_c.proposal.arguments["recipe_id"] == "rec_red_lentil_dahl"

    # Verify the bound builder's audit log recorded the inspection
    assert builder_c.get_disclosure_count() == 1
    log = builder_c.get_disclosure_log()
    assert len(log) == 1
    assert log[0]["object_type"] == "recipe"
    assert log[0]["object_id"] == "rec_red_lentil_dahl"


def test_mock_agent_deterministic_output(base_plan, catalog_data):
    """Repeated calls with identical inputs must produce byte-identical results."""
    ingredients, recipes = catalog_data
    agent = MockAgent()
    req = ContextRequest(
        user_request="replace Wednesday dinner with rec_red_lentil_dahl",
        plan=base_plan,
        catalog=recipes,
        ingredients=ingredients,
    )
    builder_a = FullStateContextBuilder()
    context_a = builder_a.build(req)

    resp_1 = agent.propose_action(context_a, inspector=None)
    resp_2 = agent.propose_action(context_a, inspector=None)

    assert resp_1 == resp_2
    assert resp_1.raw_output == resp_2.raw_output
    assert resp_1.proposal.model_dump() == resp_2.proposal.model_dump()


# =============================================================================
# 2. END-TO-END PIPELINE EXECUTION (ALL 3 CONDITIONS)
# =============================================================================


def test_pipeline_condition_a_replace_meal(base_plan, catalog_data, broker):
    """Condition A context flows through MockAgent and commits cleanly in broker."""
    ingredients, recipes = catalog_data
    agent = MockAgent()
    builder = FullStateContextBuilder()

    req = ContextRequest(
        user_request="replace Tuesday lunch with vegetarian recipe",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    # Agent selects rec_black_bean_quinoa_bowl (first alphabetical vegetarian lunch recipe)
    response = agent.propose_action(context, inspector=None)
    assert response.has_proposal is True
    assert response.proposal.arguments["recipe_id"] == "rec_black_bean_quinoa_bowl"
    assert response.proposal.arguments["day"] == "Tuesday"
    assert response.proposal.arguments["meal_type"] == "lunch"

    # ActionBroker stages, validates policies, and commits atomically
    result = broker.execute(response.proposal)
    assert result.status == ActionStatus.COMMITTED
    assert result.plan_version_before == 1
    assert result.plan_version_after == 2
    assert broker.get_plan_version() == 2

    # Verify target slot updated
    updated_meal = broker.get_plan().get_meal("Tuesday", "lunch")
    assert updated_meal is not None
    assert updated_meal.recipe_id == "rec_black_bean_quinoa_bowl"

    # Verify non-target slots preserved
    assert broker.get_plan().get_meal("Monday", "breakfast").recipe_id == "rec_berry_oatmeal"
    assert broker.get_plan().get_meal("Monday", "dinner").recipe_id == "rec_red_lentil_dahl"

    # Verify grocery basket updated
    basket = broker.get_basket()
    assert basket.basket_version == 2
    assert "ing_canned_black_beans" in basket.items


def test_pipeline_condition_b_replace_meal(base_plan, catalog_data, broker):
    """Condition B tool context flows through MockAgent and commits cleanly in broker."""
    ingredients, recipes = catalog_data
    agent = MockAgent()
    builder = ToolContextBuilder()

    req = ContextRequest(
        user_request="replace Tuesday lunch with vegetarian recipe",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    response = agent.propose_action(context, inspector=None)
    assert response.has_proposal is True
    assert response.proposal.arguments["recipe_id"] == "rec_black_bean_quinoa_bowl"

    result = broker.execute(response.proposal)
    assert result.status == ActionStatus.COMMITTED
    assert broker.get_plan_version() == 2
    assert broker.get_plan().get_meal("Tuesday", "lunch").recipe_id == "rec_black_bean_quinoa_bowl"


def test_pipeline_condition_c_replace_meal(base_plan, catalog_data, broker):
    """Condition C progressive disclosure context flows through MockAgent and commits cleanly."""
    ingredients, recipes = catalog_data
    agent = MockAgent()
    builder = ObjectContextBuilder()

    req = ContextRequest(
        user_request="replace Wednesday dinner with rec_red_lentil_dahl",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
        target_day="Wednesday",
        target_meal_type="dinner",
    )
    context = builder.build(req)

    response = agent.propose_action(context, inspector=builder)
    assert response.has_proposal is True
    assert response.inspection_calls_made == 1

    result = broker.execute(response.proposal)
    assert result.status == ActionStatus.COMMITTED
    assert broker.get_plan_version() == 2
    assert broker.get_plan().get_meal("Wednesday", "dinner").recipe_id == "rec_red_lentil_dahl"
    assert builder.get_disclosure_count() == 1


def test_pipeline_change_servings_all_conditions(base_plan, catalog_data, broker):
    """Serving scaling commands execute cleanly across all three conditions."""
    ingredients, recipes = catalog_data
    agent = MockAgent()

    for builder_cls in [FullStateContextBuilder, ToolContextBuilder, ObjectContextBuilder]:
        builder = builder_cls()
        req = ContextRequest(
            user_request="change servings for Monday breakfast to 4",
            plan=broker.get_plan(),
            catalog=recipes,
            ingredients=ingredients,
            target_day="Monday",
            target_meal_type="breakfast",
        )
        context = builder.build(req)
        inspector = builder if isinstance(builder, ObjectInspector) else None

        response = agent.propose_action(context, inspector=inspector)
        assert response.has_proposal is True
        assert response.proposal.operation == OperationType.CHANGE_SERVINGS
        assert response.proposal.arguments["servings"] == 4

        version_before = broker.get_plan_version()
        result = broker.execute(response.proposal)
        assert result.status == ActionStatus.COMMITTED
        assert broker.get_plan_version() == version_before + 1
        assert broker.get_plan().get_meal("Monday", "breakfast").servings == 4


# =============================================================================
# 3. SEMANTIC & POLICY REJECTION SAFETY (ZERO STATE MUTATION)
# =============================================================================


def test_pipeline_rejects_nonexistent_recipe(base_plan, catalog_data, broker):
    """Nonexistent recipe proposal fails at broker staging; plan and basket untouched."""
    ingredients, recipes = catalog_data
    agent = MockAgent()
    builder = FullStateContextBuilder()

    req = ContextRequest(
        user_request="TEST_NONEXISTENT_RECIPE",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)
    response = agent.propose_action(context, inspector=None)

    assert response.has_proposal is True
    assert response.proposal.arguments["recipe_id"] == "rec_nonexistent"

    version_before = broker.get_plan_version()
    plan_before = broker.get_plan()
    basket_before = broker.get_basket()

    result = broker.execute(response.proposal)
    assert result.status == ActionStatus.REJECTED
    assert "does not exist in catalog" in result.rejection_reason.lower()

    # Zero mutation invariant: plan, version, and basket remain identical
    assert broker.get_plan_version() == version_before
    assert broker.get_plan().meals == plan_before.meals
    assert broker.get_basket().items == basket_before.items


def test_pipeline_rejects_allergen_conflict(base_plan, catalog_data, broker):
    """Peanut recipe proposed to peanut-allergic profile is rejected by PolicyEngine."""
    ingredients, recipes = catalog_data
    agent = MockAgent()
    builder = FullStateContextBuilder()

    # User preferences explicitly forbid peanuts
    assert "peanuts" in broker.get_plan().preferences.allergies

    req = ContextRequest(
        user_request="TEST_ALLERGEN_CONFLICT",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)
    response = agent.propose_action(context, inspector=None)

    assert response.has_proposal is True
    assert response.proposal.arguments["recipe_id"] == "rec_peanut_butter_banana_toast"

    version_before = broker.get_plan_version()
    plan_before = broker.get_plan()
    basket_before = broker.get_basket()

    result = broker.execute(response.proposal)

    assert result.status == ActionStatus.REJECTED
    assert "peanuts" in result.rejection_reason.lower()

    # Explicit rollback assertions: plan version, scheduled meals, and basket remain unchanged
    assert broker.get_plan_version() == version_before
    assert broker.get_plan().meals == plan_before.meals
    assert broker.get_basket().items == basket_before.items


def test_pipeline_rejects_budget_overflow(catalog_data):
    """Proposal exceeding weekly budget ceiling is rejected by PolicyEngine."""
    ingredients, recipes = catalog_data

    # Controlled low budget ($10.00) with a single starting breakfast ($4.20)
    low_budget_plan = MealPlan(
        plan_id="plan_tight_budget",
        plan_version=1,
        preferences=UserPreferences(
            diet_tags=[],
            allergies=[],
            weekly_budget=Decimal("10.00"),
            household_size=2,
        ),
    )
    low_budget_plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_berry_oatmeal", servings=2))

    low_budget_broker = ActionBroker(
        initial_plan=low_budget_plan,
        catalog=recipes,
        ingredients=ingredients,
    )

    agent = MockAgent()
    builder = FullStateContextBuilder()

    # TEST_BUDGET_OVERFLOW proposes rec_pan_seared_salmon ($14.50), pushing total to $18.70 > $10.00
    req = ContextRequest(
        user_request="TEST_BUDGET_OVERFLOW",
        plan=low_budget_broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)
    response = agent.propose_action(context, inspector=None)

    assert response.has_proposal is True
    assert response.proposal.arguments["recipe_id"] == "rec_pan_seared_salmon"

    version_before = low_budget_broker.get_plan_version()
    plan_before = low_budget_broker.get_plan()
    basket_before = low_budget_broker.get_basket()

    result = low_budget_broker.execute(response.proposal)

    assert result.status == ActionStatus.REJECTED
    assert "budget" in result.rejection_reason.lower()

    # Explicit rollback assertions: plan version, scheduled meals, and basket remain unchanged
    assert low_budget_broker.get_plan_version() == version_before
    assert low_budget_broker.get_plan().meals == plan_before.meals
    assert low_budget_broker.get_basket().items == basket_before.items


def test_pipeline_rejects_stale_plan_version(base_plan, catalog_data, broker):
    """Proposal carrying a stale plan_version is rejected at broker Step 3."""
    ingredients, recipes = catalog_data
    agent = MockAgent()
    builder = FullStateContextBuilder()

    req = ContextRequest(
        user_request="TEST_STALE_VERSION",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)
    response = agent.propose_action(context, inspector=None)

    assert response.has_proposal is True
    assert response.proposal.expected_plan_version == broker.get_plan_version() + 999

    version_before = broker.get_plan_version()
    plan_before = broker.get_plan()
    basket_before = broker.get_basket()

    result = broker.execute(response.proposal)

    assert result.status == ActionStatus.REJECTED
    assert "stale" in result.rejection_reason.lower()

    # Explicit rollback assertions: plan version, scheduled meals, and basket remain unchanged
    assert broker.get_plan_version() == version_before
    assert broker.get_plan().meals == plan_before.meals
    assert broker.get_basket().items == basket_before.items


# =============================================================================
# 4. UNSUPPORTED TASKS & STRICT PLAN-VERSION VALIDATION
# =============================================================================


def test_pipeline_unsupported_task_produces_no_proposal(base_plan, catalog_data, broker):
    """Unrecognized natural language task produces has_proposal=False; broker is not invoked."""
    ingredients, recipes = catalog_data
    agent = MockAgent()
    builder = FullStateContextBuilder()

    req = ContextRequest(
        user_request="Order pizza for Friday night",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)
    response = agent.propose_action(context, inspector=None)

    assert response.has_proposal is False
    assert response.proposal is None
    assert "unsupported" in response.rationale.lower()
    assert broker.get_plan_version() == 1


def test_mock_agent_strict_plan_version_validation():
    """Agent refuses to propose state changes if plan_version is missing, boolean, negative, or string."""
    agent = MockAgent()

    test_cases = [
        '{"plan_id": "test_plan"}',                                  # Missing plan_version
        '{"plan_id": "test_plan", "plan_version": true}',             # Boolean plan_version
        '{"plan_id": "test_plan", "plan_version": -1}',               # Negative plan_version
        '{"plan_id": "test_plan", "plan_version": 0}',                # Zero plan_version
        '{"plan_id": "test_plan", "plan_version": "1"}',              # String plan_version
        '{"plan_id": "test_plan", "plan_version": 1.5}',              # Float plan_version
        'invalid json format',                                         # Malformed JSON
    ]

    for bad_json in test_cases:
        context = ContextResult(
            condition="Test Condition",
            condition_id="T",
            system_prompt="",
            task_prompt="replace Monday breakfast with rec_berry_oatmeal",
            serialized_context=bad_json,
        )
        response = agent.propose_action(context, inspector=None)
        assert response.has_proposal is False
        assert response.proposal is None
        assert response.metadata.get("status") == "invalid_plan_version"


def test_pipeline_custom_canned_fixtures(base_plan, catalog_data, broker):
    """MockAgent correctly supports registering custom canned task fixtures."""
    ingredients, recipes = catalog_data

    custom_canned = {
        "Custom Scenario 42: Swap Friday dinner": ActionProposal(
            operation=OperationType.REPLACE_MEAL,
            arguments={
                "day": "Friday",
                "meal_type": "dinner",
                "recipe_id": "rec_red_lentil_dahl",
            },
            reason="Custom research scenario replacement.",
        )
    }

    agent = MockAgent(canned_tasks=custom_canned)
    builder = FullStateContextBuilder()

    req = ContextRequest(
        user_request="Custom Scenario 42: Swap Friday dinner",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)
    response = agent.propose_action(context, inspector=None)

    assert response.has_proposal is True
    assert response.proposal.arguments["recipe_id"] == "rec_red_lentil_dahl"
    assert response.proposal.expected_plan_version == broker.get_plan_version()

    result = broker.execute(response.proposal)
    assert result.status == ActionStatus.COMMITTED
    assert broker.get_plan_version() == 2
    assert broker.get_plan().get_meal("Friday", "dinner").recipe_id == "rec_red_lentil_dahl"


def test_agent_response_metadata_immutability():
    """AgentResponse metadata must be recursively immutable and immune to external mutations."""
    from types import MappingProxyType

    initial_meta = {
        "scalar_str": "original_value",
        "scalar_int": 42,
        "nested_dict": {"inner_key": "inner_val"},
        "nested_list": [1, 2, {"deep_dict": "deep_val"}],
    }
    response = AgentResponse(
        proposal=None,
        rationale="Testing recursive immutability.",
        metadata=initial_meta,
    )

    # 1. Type checks for recursive freezing
    assert isinstance(response.metadata, MappingProxyType)
    assert response.metadata["scalar_str"] == "original_value"
    assert response.metadata["scalar_int"] == 42
    assert isinstance(response.metadata["nested_dict"], MappingProxyType)
    assert response.metadata["nested_dict"]["inner_key"] == "inner_val"
    assert isinstance(response.metadata["nested_list"], tuple)
    assert isinstance(response.metadata["nested_list"][2], MappingProxyType)
    assert response.metadata["nested_list"][2]["deep_dict"] == "deep_val"

    # 2. Rejection of top-level item mutation
    with pytest.raises(TypeError):
        response.metadata["scalar_str"] = "hacked"

    # 3. Rejection of top-level new key assignment
    with pytest.raises(TypeError):
        response.metadata["new_key"] = "forbidden"

    # 4. Rejection of nested mapping mutation
    with pytest.raises(TypeError):
        response.metadata["nested_dict"]["inner_key"] = "hacked_inner"

    # 5. Rejection of nested sequence mutation (tuple has no append)
    with pytest.raises(AttributeError):
        response.metadata["nested_list"].append(999)

    # 6. Rejection of deep dict mutation inside sequence
    with pytest.raises(TypeError):
        response.metadata["nested_list"][2]["deep_dict"] = "hacked_deep"

    # 7. Defensive copying: mutating original dict/list does not mutate response metadata
    initial_meta["scalar_str"] = "mutated_outside"
    initial_meta["nested_dict"]["inner_key"] = "mutated_inner_outside"
    initial_meta["nested_list"].append("outside_append")
    assert response.metadata["scalar_str"] == "original_value"
    assert response.metadata["nested_dict"]["inner_key"] == "inner_val"
    assert len(response.metadata["nested_list"]) == 3


def test_agent_response_serialization_with_frozen_metadata():
    """AgentResponse must serialize to JSON cleanly via to_dict() and metadata_dict."""
    import json

    prop = ActionProposal(
        operation=OperationType.CHANGE_SERVINGS,
        arguments={"day": "Tuesday", "meal_type": "lunch", "servings": 4},
        reason="Testing serialization.",
        expected_plan_version=1,
    )
    response = AgentResponse(
        proposal=prop,
        rationale="Valid proposal.",
        raw_output=prop.model_dump_json(),
        inspection_calls_made=0,
        metadata={
            "status": "ok",
            "nested_config": {"retries": 3, "tags": ["quick", "lunch"]},
        },
    )

    # 1. metadata_dict returns mutable standard dict/list
    meta_dict = response.metadata_dict
    assert isinstance(meta_dict, dict)
    assert isinstance(meta_dict["nested_config"], dict)
    assert isinstance(meta_dict["nested_config"]["tags"], list)
    assert json.dumps(meta_dict) == '{"status": "ok", "nested_config": {"retries": 3, "tags": ["quick", "lunch"]}}'

    # 2. to_dict produces completely JSON-serializable representation
    serialized_dict = response.to_dict()
    assert isinstance(serialized_dict, dict)
    assert serialized_dict["proposal"]["operation"] == "change_servings"
    assert serialized_dict["proposal"]["arguments"]["servings"] == 4

    dumped_json = json.dumps(serialized_dict)
    reloaded = json.loads(dumped_json)
    assert reloaded["proposal"]["operation"] == "change_servings"
    assert reloaded["metadata"]["nested_config"]["tags"] == ["quick", "lunch"]


def test_mock_agent_metadata_strictly_scalar_across_all_paths(base_plan, catalog_data, broker):
    """Verify MockAgent metadata contains strictly immutable scalar values across all paths."""
    import json
    from types import MappingProxyType

    ingredients, recipes = catalog_data
    builder_a = FullStateContextBuilder()
    builder_c = ObjectContextBuilder()
    agent = MockAgent()

    test_scenarios = [
        # 1. Invalid plan version (pass empty context JSON)
        ContextResult(
            condition="Condition A: Full State",
            condition_id="A",
            task_prompt="replace Monday breakfast with rec_berry_oatmeal",
            system_prompt="sys",
            serialized_context="{}",
        ),
        # 2. Canned task fixture
        ContextResult(
            condition="Condition A: Full State",
            condition_id="A",
            task_prompt="TEST_CANNED",
            system_prompt="sys",
            serialized_context=json.dumps({"plan_version": 1}),
        ),
        # 3-6. Semantic test fixtures
        builder_a.build(ContextRequest("TEST_NONEXISTENT_RECIPE", broker.get_plan(), recipes, ingredients)),
        builder_a.build(ContextRequest("TEST_ALLERGEN_CONFLICT", broker.get_plan(), recipes, ingredients)),
        builder_a.build(ContextRequest("TEST_BUDGET_OVERFLOW", broker.get_plan(), recipes, ingredients)),
        builder_a.build(ContextRequest("TEST_STALE_VERSION", broker.get_plan(), recipes, ingredients)),
        # 7. Change servings grammar
        builder_a.build(ContextRequest("change servings for Monday breakfast to 4", broker.get_plan(), recipes, ingredients)),
        # 8. Replace exact grammar
        builder_a.build(ContextRequest("replace Monday breakfast with rec_berry_oatmeal", broker.get_plan(), recipes, ingredients)),
        # 9. Replace tag grammar (Condition A)
        builder_a.build(ContextRequest("replace Tuesday lunch with vegetarian recipe", broker.get_plan(), recipes, ingredients)),
        # 10. Replace tag grammar rejected in Condition C
        builder_c.build(ContextRequest("replace Tuesday lunch with vegetarian recipe", broker.get_plan(), recipes, ingredients)),
        # 11. Invalid schedule coordinates (replace)
        builder_a.build(ContextRequest("replace Funday breakfast with rec_berry_oatmeal", broker.get_plan(), recipes, ingredients)),
        # 12. Invalid schedule coordinates (servings)
        builder_a.build(ContextRequest("change servings for Funday brunch to 4", broker.get_plan(), recipes, ingredients)),
        # 13. No matching recipes for tag
        builder_a.build(ContextRequest("replace Monday breakfast with nonexistenttagxyz recipe", broker.get_plan(), recipes, ingredients)),
        # 14. Unrecognized command grammar
        builder_a.build(ContextRequest("please make something delicious", broker.get_plan(), recipes, ingredients)),
    ]

    agent_with_canned = MockAgent(canned_tasks={
        "TEST_CANNED": ActionProposal(
            operation=OperationType.REPLACE_MEAL,
            arguments={"day": "Monday", "meal_type": "breakfast", "recipe_id": "rec_berry_oatmeal"},
            reason="Canned test.",
        )
    })

    responses = []
    for ctx in test_scenarios:
        act_agent = agent_with_canned if ctx.task_prompt == "TEST_CANNED" else agent
        resp = act_agent.propose_action(ctx, inspector=builder_c if ctx.condition_id == "C" else None)
        responses.append(resp)

    assert len(responses) == len(test_scenarios)

    for resp in responses:
        # Every response must produce a frozen MappingProxyType metadata mapping
        assert isinstance(resp.metadata, MappingProxyType)
        assert len(resp.metadata) > 0

        # Verify MockAgent metadata invariant: every value is strictly a scalar
        for k, v in resp.metadata.items():
            assert isinstance(k, str), f"Metadata key must be str: {k!r}"
            assert isinstance(v, (str, int, float, bool, type(None))), (
                f"MockAgent metadata value for '{k}' must be an immutable scalar, got {type(v)}: {v!r}"
            )
            assert not isinstance(v, (dict, list, set, tuple, MappingProxyType)), (
                f"MockAgent metadata value for '{k}' is a nested container: {v!r}"
            )

        # Verify serialization succeeds without errors
        serialized_meta = json.dumps(resp.metadata_dict)
        assert isinstance(serialized_meta, str)
        serialized_full = json.dumps(resp.to_dict())
        assert isinstance(serialized_full, str)


def test_canned_proposal_dynamically_receives_context_plan_version(base_plan, catalog_data, broker):
    """Ordinary canned proposals with stale template versions must dynamically receive context version."""
    ingredients, recipes = catalog_data

    # Template has a stale version (999) in its static definition
    canned = {
        "Swap Friday dinner": ActionProposal(
            operation=OperationType.REPLACE_MEAL,
            arguments={"day": "Friday", "meal_type": "dinner", "recipe_id": "rec_red_lentil_dahl"},
            reason="Swap Friday dinner.",
            expected_plan_version=999,
        )
    }

    agent = MockAgent(canned_tasks=canned)
    builder = FullStateContextBuilder()
    req = ContextRequest(
        user_request="Swap Friday dinner",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    response = agent.propose_action(context, inspector=None)
    assert response.has_proposal is True
    # Dynamically updated to match authoritative context version (1, not 999)
    assert response.proposal.expected_plan_version == broker.get_plan_version()

    result = broker.execute(response.proposal)
    assert result.status == ActionStatus.COMMITTED
    assert broker.get_plan_version() == 2


def test_canned_stale_version_fixture_retains_mismatched_version(base_plan, catalog_data, broker):
    """Only the explicit TEST_STALE_VERSION fixture intentionally uses a mismatched version."""
    ingredients, recipes = catalog_data

    canned = {
        "TEST_STALE_VERSION": ActionProposal(
            operation=OperationType.REPLACE_MEAL,
            arguments={"day": "Wednesday", "meal_type": "dinner", "recipe_id": "rec_red_lentil_dahl"},
            reason="Intentional stale test.",
            expected_plan_version=777,
        )
    }

    agent = MockAgent(canned_tasks=canned)
    builder = FullStateContextBuilder()
    req = ContextRequest(
        user_request="TEST_STALE_VERSION",
        plan=broker.get_plan(),
        catalog=recipes,
        ingredients=ingredients,
    )
    context = builder.build(req)

    response = agent.propose_action(context, inspector=None)
    assert response.has_proposal is True
    # Explicit stale fixture preserves mismatched version
    assert response.proposal.expected_plan_version == 777

    result = broker.execute(response.proposal)
    assert result.status == ActionStatus.REJECTED
    assert "stale" in result.rejection_reason.lower()
    assert broker.get_plan_version() == 1
