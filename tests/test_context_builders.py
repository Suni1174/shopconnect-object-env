"""Unit tests for Stage 4 Context Strategies (Conditions A, B, and C)."""

import copy
from decimal import Decimal
import json
from pathlib import Path
import pytest

from shopconnect_env.context.base import (
    ConditionType,
    ContextRequest,
    ContextResult,
    deterministic_json,
    serialize_all_21_slots,
    serialize_decimal,
    serialize_grocery_basket,
    serialize_ingredient,
    serialize_meal,
    serialize_recipe,
    serialize_user_preferences,
)
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
from shopconnect_env.operations.basket_calculator import calculate_grocery_basket
from shopconnect_env.validation.schemas import ChangeServingsArgs, ReplaceMealArgs


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
    # Pre-populate specific meals
    plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_berry_oatmeal", servings=2))
    plan.set_meal(Meal(day="Monday", meal_type="dinner", recipe_id="rec_red_lentil_dahl", servings=4))
    plan.set_meal(Meal(day="Wednesday", meal_type="dinner", recipe_id="rec_hearty_veggie_soup", servings=4))
    plan.set_meal(Meal(day="Friday", meal_type="lunch", recipe_id="rec_mediterranean_chickpea_salad", servings=2))
    return plan


@pytest.fixture
def context_request(base_plan, catalog_data):
    """Create a standardized ContextRequest."""
    ingredients, recipes = catalog_data
    return ContextRequest(
        user_request="Replace Wednesday dinner with a vegetarian pasta dish under $15.",
        plan=base_plan,
        catalog=recipes,
        ingredients=ingredients,
        target_day="Wednesday",
        target_meal_type="dinner",
    )


# =============================================================================
# 1. SHARED BEHAVIOR & SERIALIZATION TESTS
# =============================================================================


def test_shared_context_request_auto_derives_basket(base_plan, catalog_data):
    """ContextRequest should automatically derive a GroceryBasket if not explicitly provided."""
    ingredients, recipes = catalog_data
    req = ContextRequest(
        user_request="Review my current meals.",
        plan=base_plan,
        catalog=recipes,
        ingredients=ingredients,
        basket=None,
    )
    assert req.basket is not None
    assert req.basket.basket_version == base_plan.plan_version
    assert len(req.basket.items) > 0


def test_all_builders_accept_equivalent_request(context_request):
    """All three context builders must consume the exact same ContextRequest input."""
    builder_a = FullStateContextBuilder()
    builder_b = ToolContextBuilder()
    builder_c = ObjectContextBuilder()

    res_a = builder_a.build(context_request)
    res_b = builder_b.build(context_request)
    res_c = builder_c.build(context_request)

    assert isinstance(res_a, ContextResult)
    assert isinstance(res_b, ContextResult)
    assert isinstance(res_c, ContextResult)

    assert res_a.condition_id == ConditionType.CONDITION_A.value
    assert res_b.condition_id == ConditionType.CONDITION_B.value
    assert res_c.condition_id == ConditionType.CONDITION_C.value


def test_builders_do_not_mutate_source_plan_or_catalogs(context_request):
    """Context building must be strictly read-only and never mutate input objects."""
    plan_before = copy.deepcopy(context_request.plan)
    catalog_len_before = len(context_request.catalog)
    ing_len_before = len(context_request.ingredients)
    basket_before = copy.deepcopy(context_request.basket)

    builder_a = FullStateContextBuilder()
    builder_b = ToolContextBuilder()
    builder_c = ObjectContextBuilder()

    _ = builder_a.build(context_request)
    _ = builder_b.build(context_request)
    _ = builder_c.build(context_request)

    # Verify authoritative state remains unaltered
    assert context_request.plan.plan_version == plan_before.plan_version
    assert len(context_request.plan.meals) == len(plan_before.meals)
    assert set(context_request.plan.meals.keys()) == set(plan_before.meals.keys())
    assert len(context_request.catalog) == catalog_len_before
    assert len(context_request.ingredients) == ing_len_before
    assert context_request.basket == basket_before


def test_deterministic_serialization_across_repeated_runs(context_request):
    """Serialization outputs must be byte-for-byte identical across repeated executions."""
    builder_a = FullStateContextBuilder()
    builder_b = ToolContextBuilder()
    builder_c = ObjectContextBuilder()

    runs_a = [builder_a.build(context_request).serialized_context for _ in range(3)]
    runs_b = [builder_b.build(context_request).serialized_context for _ in range(3)]
    runs_c = [builder_c.build(context_request).serialized_context for _ in range(3)]

    assert len(set(runs_a)) == 1, "Condition A produced non-deterministic serialized output."
    assert len(set(runs_b)) == 1, "Condition B produced non-deterministic serialized output."
    assert len(set(runs_c)) == 1, "Condition C produced non-deterministic serialized output."


def test_decimal_serialization_exactness():
    """Decimal values must be serialized as exact strings without floating-point artifacts."""
    val = Decimal("12.50")
    serialized = serialize_decimal(val)
    assert serialized == "12.50"
    assert isinstance(serialized, str)

    # Ensure json output contains string numbers
    payload = {"price": val}
    json_str = deterministic_json({"cost": serialize_decimal(val)})
    assert '"cost": "12.50"' in json_str


def test_all_21_slots_explicitly_represented(base_plan, catalog_data):
    """All 21 slots (Monday-Sunday, 3 meals/day) must be explicitly listed with status."""
    _, recipes = catalog_data
    slots = serialize_all_21_slots(base_plan, recipes)

    assert len(slots) == 21
    # Check ordering
    expected_first = "Monday_breakfast"
    expected_last = "Sunday_dinner"
    assert slots[0]["slot_key"] == expected_first
    assert slots[-1]["slot_key"] == expected_last

    # Check filled vs unfilled slots
    assigned = [s for s in slots if s["status"] == "assigned"]
    unfilled = [s for s in slots if s["status"] == "unfilled"]

    assert len(assigned) == 4
    assert len(unfilled) == 17

    # Check structure of unfilled slot
    for u in unfilled:
        assert u["recipe_id"] is None
        assert u["servings"] == 0
        assert u["estimated_cost"] == "0.00"


# =============================================================================
# 2. CONDITION A: FULL-STATE REGENERATION TESTS
# =============================================================================


def test_condition_a_contains_complete_state(context_request):
    """Condition A context must include schedule, recipes, ingredients, and basket."""
    builder = FullStateContextBuilder()
    result = builder.build(context_request)

    assert result.condition_id == "A"
    assert result.signatures is None  # No tool/method signatures
    assert result.metadata["total_slots"] == 21
    assert result.metadata["assigned_slots"] == 4
    assert result.metadata["unfilled_slots"] == 17
    assert result.metadata["disclosure_count"] == 0

    parsed = json.loads(result.serialized_context)
    assert parsed["plan_id"] == "plan_research_week_1"
    assert parsed["plan_version"] == 1
    assert len(parsed["schedule"]) == 21
    assert "recipes_in_plan" in parsed
    assert "available_recipes" in parsed
    assert "ingredient_catalog" in parsed
    assert "grocery_basket" in parsed


def test_condition_a_recipe_ingredient_requirements_included(context_request):
    """Condition A must embed complete ingredient requirements for assigned recipes."""
    builder = FullStateContextBuilder()
    result = builder.build(context_request)
    parsed = json.loads(result.serialized_context)

    # Monday breakfast is 'rec_berry_oatmeal'
    recipes_in_plan = parsed["recipes_in_plan"]
    assert "rec_berry_oatmeal" in recipes_in_plan
    oatmeal = recipes_in_plan["rec_berry_oatmeal"]
    assert "ingredient_requirements" in oatmeal
    assert len(oatmeal["ingredient_requirements"]) > 0

    first_req = oatmeal["ingredient_requirements"][0]
    assert "ingredient_id" in first_req
    assert "quantity" in first_req
    assert "unit" in first_req


# =============================================================================
# 3. CONDITION B: CONVENTIONAL FLAT TOOL-CALLING TESTS
# =============================================================================


def test_condition_b_tool_definitions_match_pydantic_schemas(context_request):
    """Condition B must provide flat tool specifications matching Pydantic schemas."""
    builder = ToolContextBuilder()
    result = builder.build(context_request)

    assert result.condition_id == "B"
    assert isinstance(result.signatures, list)
    assert len(result.signatures) == 2

    tool_names = [t["function"]["name"] for t in result.signatures]
    assert "replace_meal" in tool_names
    assert "change_servings" in tool_names

    # Check parameters match Pydantic schemas
    replace_tool = next(t for t in result.signatures if t["function"]["name"] == "replace_meal")
    assert replace_tool["function"]["parameters"] == ReplaceMealArgs.model_json_schema()

    change_tool = next(t for t in result.signatures if t["function"]["name"] == "change_servings")
    assert change_tool["function"]["parameters"] == ChangeServingsArgs.model_json_schema()


def test_condition_b_state_snapshot_content(context_request):
    """Condition B state snapshot must contain schedule summary and basket summary."""
    builder = ToolContextBuilder()
    result = builder.build(context_request)
    parsed = json.loads(result.serialized_context)

    assert "plan_id" in parsed
    assert "plan_version" in parsed
    assert "preferences" in parsed
    assert "current_schedule" in parsed
    assert "available_recipes" in parsed
    assert "basket_summary" in parsed

    # Should only contain active meals, not 21 unfilled slots
    assert len(parsed["current_schedule"]) == 4

    # Available recipes should contain high-level metadata without raw ingredient table
    for r in parsed["available_recipes"]:
        assert "recipe_id" in r
        assert "name" in r
        assert "tags" in r
        assert "ingredient_requirements" not in r  # Kept high-level in B


def test_condition_b_does_not_expose_progressive_inspection(context_request):
    """Condition B must not expose Condition C's progressive inspection operations."""
    builder = ToolContextBuilder()
    result = builder.build(context_request)

    tool_names = [t["function"]["name"] for t in result.signatures]
    assert "inspect_slot" not in tool_names
    assert "inspect_recipe" not in tool_names
    assert "inspect_ingredient" not in tool_names


# =============================================================================
# 4. CONDITION C: OBJECT-CENTRIC PROGRESSIVE DISCLOSURE TESTS
# =============================================================================


def test_condition_c_initial_context_is_compact(context_request):
    """Condition C initial context must be substantially smaller than Condition A."""
    builder_a = FullStateContextBuilder()
    builder_c = ObjectContextBuilder()

    res_a = builder_a.build(context_request)
    res_c = builder_c.build(context_request)

    # Condition C serialized context should be noticeably more compact than full state
    assert len(res_c.serialized_context) < len(res_a.serialized_context)

    parsed_c = json.loads(res_c.serialized_context)
    assert "schedule" not in parsed_c  # No 21 slots
    assert "available_recipes" not in parsed_c  # No full catalog dump
    assert "ingredient_catalog" not in parsed_c  # No ingredient dump
    assert "grocery_basket" not in parsed_c  # No basket dump


def test_condition_c_includes_class_signatures(context_request):
    """Condition C must supply compact class and method signatures."""
    builder = ObjectContextBuilder()
    result = builder.build(context_request)

    assert result.condition_id == "C"
    assert isinstance(result.signatures, str)
    assert "class MealPlan:" in result.signatures
    assert "class Recipe:" in result.signatures
    assert "class Ingredient:" in result.signatures
    assert "class UserPreferences:" in result.signatures
    assert "ActionBroker" in result.signatures
    assert 'Operation: "replace_meal"' in result.signatures
    assert 'Operation: "change_servings"' in result.signatures
    assert "inspect_slot" in result.signatures


def test_compact_signatures_distinguish_domain_models_from_broker_proposals(context_request):
    """Signatures must not represent mutating operations as direct methods on MealPlan.

    Verifies:
    1. MealPlan has no replace_meal or change_servings methods in Python.
    2. MealPlan in COMPACT_CLASS_SIGNATURES does not advertise def replace_meal( or def change_servings(.
    3. Mutating operations are explicitly categorized as ActionProposal actions executed via ActionBroker.
    4. MealPlan advertises only its actual read-only accessors (get_meal, calculate_total_cost, is_complete).
    5. Inspection operations match actual ObjectContextBuilder read-only inspection methods.
    """
    builder = ObjectContextBuilder()
    result = builder.build(context_request)
    signatures = result.signatures

    # 1. Authoritative domain model truth
    assert not hasattr(MealPlan, "replace_meal")
    assert not hasattr(MealPlan, "change_servings")
    assert hasattr(MealPlan, "get_meal")
    assert hasattr(MealPlan, "calculate_total_cost")
    assert hasattr(MealPlan, "is_complete")

    # 2. Signatures must not claim replace_meal or change_servings are methods on MealPlan
    assert "def replace_meal(" not in signatures
    assert "def change_servings(" not in signatures

    # 3. Read-only domain accessors ARE advertised on MealPlan
    assert "def get_meal(self, day: str, meal_type: str)" in signatures
    assert "def calculate_total_cost(self, catalog: Dict[str, Recipe])" in signatures
    assert "def is_complete(self)" in signatures

    # 4. Mutating operations are explicitly documented as ActionBroker proposals
    assert "ActionBroker.execute(proposal)" in signatures
    assert 'Operation: "replace_meal"' in signatures
    assert 'Operation: "change_servings"' in signatures

    # 5. Read-only progressive inspection methods are explicitly listed
    for fn in [
        "inspect_slot",
        "inspect_recipe",
        "inspect_recipe_ingredients",
        "inspect_ingredient",
        "inspect_plan",
        "inspect_preferences",
    ]:
        assert fn in signatures
        assert hasattr(builder, fn)


def test_condition_c_target_slot_summary_when_specified(context_request):
    """Condition C includes target slot summary if coordinates are in request."""
    builder = ObjectContextBuilder()
    result = builder.build(context_request)
    parsed = json.loads(result.serialized_context)

    assert parsed["target_slot"] is not None
    assert parsed["target_slot"]["slot_key"] == "Wednesday_dinner"
    assert parsed["target_slot"]["status"] == "assigned"
    assert parsed["target_slot"]["recipe_id"] == "rec_hearty_veggie_soup"


def test_condition_c_target_slot_unfilled_representation(base_plan, catalog_data):
    """Condition C represents an unfilled target slot safely."""
    ingredients, recipes = catalog_data
    req = ContextRequest(
        user_request="Add breakfast for Tuesday.",
        plan=base_plan,
        catalog=recipes,
        ingredients=ingredients,
        target_day="Tuesday",
        target_meal_type="breakfast",
    )
    builder = ObjectContextBuilder()
    result = builder.build(req)
    parsed = json.loads(result.serialized_context)

    assert parsed["target_slot"] is not None
    assert parsed["target_slot"]["slot_key"] == "Tuesday_breakfast"
    assert parsed["target_slot"]["status"] == "unfilled"
    assert parsed["target_slot"]["recipe_id"] is None


def test_condition_c_progressive_slot_inspection(context_request):
    """Condition C inspect_slot retrieves assigned or unfilled slot on demand."""
    builder = ObjectContextBuilder()
    builder.build(context_request)

    # Inspect assigned slot
    slot_info = builder.inspect_slot("Monday", "breakfast")
    assert slot_info["status"] == "assigned"
    assert slot_info["recipe_id"] == "rec_berry_oatmeal"

    # Inspect unfilled slot
    empty_slot = builder.inspect_slot("Tuesday", "lunch")
    assert empty_slot["status"] == "unfilled"
    assert empty_slot["recipe_id"] is None

    assert builder.get_disclosure_count() == 2


def test_condition_c_progressive_recipe_and_ingredients_inspection(context_request):
    """Condition C inspect_recipe and inspect_recipe_ingredients retrieve details on demand."""
    builder = ObjectContextBuilder()
    builder.build(context_request)

    # 1. Inspect recipe metadata
    rec_info = builder.inspect_recipe("rec_mediterranean_chickpea_salad")
    assert rec_info["status"] == "found"
    assert rec_info["name"] == "Mediterranean Chickpea Salad"
    assert "vegetarian" in rec_info["tags"]
    assert "ingredient_count" in rec_info

    # 2. Inspect recipe ingredients
    ing_info = builder.inspect_recipe_ingredients("rec_mediterranean_chickpea_salad")
    assert ing_info["status"] == "found"
    assert len(ing_info["ingredient_requirements"]) > 0
    first_req = ing_info["ingredient_requirements"][0]
    assert "ingredient_id" in first_req
    assert "quantity" in first_req

    # 3. Inspect specific canonical ingredient
    ing_spec = builder.inspect_ingredient("ing_olive_oil")
    assert ing_spec["status"] == "found"
    assert ing_spec["name"] == "Extra Virgin Olive Oil"
    assert ing_spec["standard_unit"] == "ml"

    assert builder.get_disclosure_count() == 3


def test_condition_c_inspect_unknown_objects_handled_safely(context_request):
    """Unknown recipes, ingredients, or slots must return structured error results."""
    builder = ObjectContextBuilder()
    builder.build(context_request)

    # Unknown recipe
    rec_err = builder.inspect_recipe("rec_non_existent")
    assert rec_err["status"] == "error"
    assert rec_err["error"] == "not_found"

    # Unknown ingredient
    ing_err = builder.inspect_ingredient("ing_mystery_dust")
    assert ing_err["status"] == "error"
    assert ing_err["error"] == "not_found"

    # Invalid day
    slot_err = builder.inspect_slot("Funday", "dinner")
    assert slot_err["status"] == "error"
    assert slot_err["error"] == "invalid_day"


def test_condition_c_unified_inspection_dispatcher(context_request):
    """Unified inspect() dispatcher routes queries correctly across all target types."""
    builder = ObjectContextBuilder()
    builder.build(context_request)

    res_slot = builder.inspect("slot", target_id="Monday.dinner")
    assert res_slot["status"] == "assigned"
    assert res_slot["recipe_id"] == "rec_red_lentil_dahl"

    res_rec = builder.inspect("recipe", target_id="rec_red_lentil_dahl")
    assert res_rec["status"] == "found"
    assert res_rec["name"] == "Golden Red Lentil Dahl"

    res_reqs = builder.inspect("recipe_ingredients", target_id="rec_red_lentil_dahl")
    assert res_reqs["status"] == "found"
    assert len(res_reqs["ingredient_requirements"]) > 0

    res_ing = builder.inspect("ingredient", target_id="ing_red_lentils")
    assert res_ing["status"] == "found"
    assert res_ing["name"] == "Dry Red Lentils"

    res_plan = builder.inspect("plan")
    assert res_plan["status"] == "found"
    assert res_plan["plan_version"] == 1

    res_pref = builder.inspect("preferences")
    assert res_pref["status"] == "found"
    assert "vegetarian" in res_pref["diet_tags"]

    res_invalid = builder.inspect("unknown_category")
    assert res_invalid["status"] == "error"


def test_condition_c_inspection_does_not_mutate_state(context_request):
    """Mutating returned inspection dictionaries must not alter builder internal state."""
    builder = ObjectContextBuilder()
    builder.build(context_request)

    rec = builder.inspect_recipe("rec_mediterranean_chickpea_salad")
    rec["tags"].append("mutated_tag")
    rec["name"] = "Hacked Salad"

    # Inspect again to verify internal state remained pristine
    fresh = builder.inspect_recipe("rec_mediterranean_chickpea_salad")
    assert fresh["name"] == "Mediterranean Chickpea Salad"
    assert "mutated_tag" not in fresh["tags"]


def test_condition_c_disclosure_audit_logging(context_request):
    """Successful inspections must be tracked in disclosure audit log."""
    builder = ObjectContextBuilder()
    builder.build(context_request)

    assert builder.get_disclosure_count() == 0
    assert len(builder.get_disclosure_log()) == 0

    builder.inspect_slot("Monday", "breakfast")
    builder.inspect_recipe("rec_berry_oatmeal")
    builder.inspect_ingredient("ing_rolled_oats")

    assert builder.get_disclosure_count() == 3
    log = builder.get_disclosure_log()
    assert len(log) == 3

    assert log[0]["object_type"] == "slot"
    assert log[0]["object_id"] == "Monday_breakfast"
    assert log[1]["object_type"] == "recipe"
    assert log[1]["object_id"] == "rec_berry_oatmeal"
    assert log[2]["object_type"] == "ingredient"
    assert log[2]["object_id"] == "ing_rolled_oats"

    # Reset log
    builder.reset_disclosure_log()
    assert builder.get_disclosure_count() == 0
    assert len(builder.get_disclosure_log()) == 0


def test_empty_user_request_raises_value_error(base_plan, catalog_data):
    """ContextRequest must reject empty or whitespace-only user requests."""
    ingredients, recipes = catalog_data
    with pytest.raises(ValueError, match="user_request cannot be empty"):
        ContextRequest(
            user_request="   ",
            plan=base_plan,
            catalog=recipes,
            ingredients=ingredients,
        )


def test_condition_c_without_target_slot_in_request(base_plan, catalog_data):
    """Condition C handles requests without target coordinates gracefully."""
    ingredients, recipes = catalog_data
    req = ContextRequest(
        user_request="Plan meals for next week.",
        plan=base_plan,
        catalog=recipes,
        ingredients=ingredients,
        target_day=None,
        target_meal_type=None,
    )
    builder = ObjectContextBuilder()
    result = builder.build(req)
    parsed = json.loads(result.serialized_context)

    assert parsed["target_slot"] is None
    assert result.metadata["target_slot_specified"] is False


def test_condition_c_inspect_plan_and_preferences_direct_methods(context_request):
    """Condition C inspect_plan and inspect_preferences direct methods work cleanly."""
    builder = ObjectContextBuilder()
    builder.build(context_request)

    plan_res = builder.inspect_plan()
    assert plan_res["status"] == "found"
    assert plan_res["plan_id"] == "plan_research_week_1"
    assert plan_res["plan_version"] == 1
    assert plan_res["total_assigned_meals"] == 4

    pref_res = builder.inspect_preferences()
    assert pref_res["status"] == "found"
    assert "vegetarian" in pref_res["diet_tags"]
    assert "peanuts" in pref_res["allergies"]


def test_condition_c_cumulative_character_tracking(context_request):
    """Cumulative revealed characters track payload sizes and distinguish from initial context."""
    builder = ObjectContextBuilder()
    initial_res = builder.build(context_request)

    init_chars = initial_res.metadata["initial_context_character_count"]
    assert init_chars > 0
    assert builder.get_cumulative_revealed_characters() == 0

    builder.inspect_recipe("rec_berry_oatmeal")
    builder.inspect_ingredient("ing_rolled_oats")

    revealed = builder.get_cumulative_revealed_characters()
    assert revealed > 0
    total = builder.get_total_context_characters(initial_res)
    assert total == init_chars + revealed
