"""Unit tests for Stage 3 Action Broker."""

import copy
from decimal import Decimal
from pathlib import Path
import pytest

from shopconnect_env.domain.models import (
    Ingredient,
    IngredientRequirement,
    Meal,
    MealPlan,
    Recipe,
    UserPreferences,
    load_catalog_from_json,
)
from shopconnect_env.validation.broker import ActionBroker
from shopconnect_env.validation.schemas import (
    ActionProposal,
    ActionResult,
    ActionStatus,
)


@pytest.fixture
def catalog_data():
    """Load canonical ingredients and recipe catalog from seed JSON."""
    catalog_path = Path(__file__).parent.parent / "data" / "catalog_seed.json"
    ingredients, recipes = load_catalog_from_json(catalog_path)
    return ingredients, recipes


@pytest.fixture
def base_plan():
    """Create a baseline MealPlan."""
    plan = MealPlan(
        plan_id="plan_test_week",
        plan_version=1,
        preferences=UserPreferences(
            diet_tags=["vegetarian"],
            allergies=["peanuts"],
            weekly_budget=Decimal("50.00"),
        ),
    )
    # Pre-populate Monday breakfast & dinner
    plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_berry_oatmeal", servings=2))
    plan.set_meal(Meal(day="Monday", meal_type="dinner", recipe_id="rec_red_lentil_dahl", servings=4))
    # Pre-populate Wednesday dinner
    plan.set_meal(Meal(day="Wednesday", meal_type="dinner", recipe_id="rec_hearty_veggie_soup", servings=4))
    return plan


@pytest.fixture
def broker(base_plan, catalog_data):
    """Create an ActionBroker instance."""
    ingredients, recipes = catalog_data
    return ActionBroker(
        initial_plan=base_plan,
        catalog=recipes,
        ingredients=ingredients,
    )


def test_successful_meal_replacement(broker):
    """Verify that a compliant replace_meal action commits and updates basket."""
    # Replace Wednesday dinner with Tomato Spinach Pasta (vegetarian, peanut-free, cheap)
    proposal = {
        "operation": "replace_meal",
        "arguments": {
            "day": "Wednesday",
            "meal_type": "dinner",
            "recipe_id": "rec_tomato_spinach_pasta",
            "servings": 2,
        },
        "reason": "Swap soup for pasta",
        "expected_plan_version": 1,
    }

    result = broker.execute(proposal)
    assert result.status == ActionStatus.COMMITTED
    assert result.plan_version_before == 1
    assert result.plan_version_after == 2
    assert broker.get_plan_version() == 2

    # Check committed plan
    committed_plan = broker.get_plan()
    wed_dinner = committed_plan.get_meal("Wednesday", "dinner")
    assert wed_dinner is not None
    assert wed_dinner.recipe_id == "rec_tomato_spinach_pasta"
    assert wed_dinner.servings == 2

    # Check committed basket
    committed_basket = broker.get_basket()
    assert committed_basket.basket_version == 2
    assert "ing_whole_wheat_pasta" in committed_basket.items


def test_successful_serving_update(broker):
    """Verify that changing servings commits, updates version, and recalculates basket."""
    proposal = {
        "operation": "change_servings",
        "arguments": {
            "day": "Monday",
            "meal_type": "breakfast",
            "servings": 4,
        },
        "reason": "Double breakfast portions",
        "expected_plan_version": 1,
    }

    result = broker.execute(proposal)
    assert result.status == ActionStatus.COMMITTED
    assert result.plan_version_before == 1
    assert result.plan_version_after == 2

    committed_plan = broker.get_plan()
    assert committed_plan.get_meal("Monday", "breakfast").servings == 4

    committed_basket = broker.get_basket()
    # Oats for 2 servings is 100g -> for 4 servings is 200g
    assert committed_basket.items["ing_rolled_oats"].quantity == Decimal("200.0")


def test_non_target_meals_preserved_on_commit(broker):
    """Verify that unrelated meal slots are identical after commit."""
    orig_mon_dinner = broker.get_plan().get_meal("Monday", "dinner")

    proposal = {
        "operation": "replace_meal",
        "arguments": {
            "day": "Wednesday",
            "meal_type": "dinner",
            "recipe_id": "rec_tomato_spinach_pasta",
            "servings": 2,
        },
        "reason": "Swap Wednesday dinner",
    }
    result = broker.execute(proposal)
    assert result.status == ActionStatus.COMMITTED

    new_mon_dinner = broker.get_plan().get_meal("Monday", "dinner")
    assert new_mon_dinner == orig_mon_dinner


def test_malformed_proposals_rejected(broker):
    """Verify rejection of syntactically invalid proposals with zero state mutation."""
    version_before = broker.get_plan_version()
    plan_snapshot = broker.get_plan()
    basket_snapshot = broker.get_basket()

    # Missing arguments
    result = broker.execute({"operation": "replace_meal", "reason": "No args"})
    assert result.status == ActionStatus.REJECTED
    assert "validation error" in result.message.lower()

    # Malformed JSON string
    result2 = broker.execute('{"bad json": ...')
    assert result2.status == ActionStatus.REJECTED

    # Authoritative state remains completely untouched
    assert broker.get_plan_version() == version_before
    assert broker.get_plan() == plan_snapshot
    assert broker.get_basket() == basket_snapshot


def test_unsupported_operations_rejected(broker):
    """Verify rejection of non-allowlisted operations and read-only actions."""
    version_before = broker.get_plan_version()

    # Read-only inspect_object rejected by mutation broker
    result = broker.execute({
        "operation": "inspect_object",
        "arguments": {"target_type": "recipe", "target_id": "rec_01"},
        "reason": "Inspect object",
    })
    assert result.status == ActionStatus.REJECTED
    assert "read-only" in result.message

    assert broker.get_plan_version() == version_before


def test_dietary_violations_rejected(broker):
    """Verify rejection when candidate violates user dietary tag (vegetarian)."""
    version_before = broker.get_plan_version()
    plan_snapshot = broker.get_plan()

    # Try assigning chicken to a vegetarian user
    proposal = {
        "operation": "replace_meal",
        "arguments": {
            "day": "Wednesday",
            "meal_type": "dinner",
            "recipe_id": "rec_baked_chicken_rice",
            "servings": 2,
        },
        "reason": "Trying to sneak in chicken",
    }

    result = broker.execute(proposal)
    assert result.status == ActionStatus.REJECTED
    assert "Policy rejection (dietary_restrictions)" in result.message
    assert "vegetarian" in result.message

    # State unchanged
    assert broker.get_plan_version() == version_before
    assert broker.get_plan() == plan_snapshot


def test_allergen_violations_rejected(broker):
    """Verify rejection when candidate violates user allergy (peanuts)."""
    version_before = broker.get_plan_version()
    plan_snapshot = broker.get_plan()

    # rec_peanut_butter_banana_toast contains peanuts
    proposal = {
        "operation": "replace_meal",
        "arguments": {
            "day": "Monday",
            "meal_type": "breakfast",
            "recipe_id": "rec_peanut_butter_banana_toast",
            "servings": 2,
        },
        "reason": "Peanut toast",
    }

    result = broker.execute(proposal)
    assert result.status == ActionStatus.REJECTED
    assert "Policy rejection (allergen_safety)" in result.message
    assert "peanuts" in result.message

    # State unchanged
    assert broker.get_plan_version() == version_before
    assert broker.get_plan() == plan_snapshot


def test_budget_violations_rejected(broker):
    """Verify rejection when candidate pushes total cost over weekly budget."""
    # Set a tiny weekly budget of $10.00
    broker._state.plan.preferences.weekly_budget = Decimal("10.00")
    version_before = broker.get_plan_version()
    plan_snapshot = broker.get_plan()

    # Current meals (berry oatmeal $4.20 + lentil dahl $7.80 = $12.00) already exceeds or is near
    # Replacing Wednesday dinner with another expensive meal will exceed $10 cap
    proposal = {
        "operation": "replace_meal",
        "arguments": {
            "day": "Wednesday",
            "meal_type": "dinner",
            "recipe_id": "rec_tofu_stir_fry",
            "servings": 2,
        },
        "reason": "Tofu stir fry",
    }

    result = broker.execute(proposal)
    assert result.status == ActionStatus.REJECTED
    assert "weekly_budget" in result.message

    # State unchanged
    assert broker.get_plan_version() == version_before
    assert broker.get_plan() == plan_snapshot


def test_stale_proposals_rejected(broker):
    """Verify stale-state protection: proposal with outdated expected_plan_version is rejected."""
    version_before = broker.get_plan_version()  # Version 1

    proposal = {
        "operation": "replace_meal",
        "arguments": {
            "day": "Wednesday",
            "meal_type": "dinner",
            "recipe_id": "rec_tomato_spinach_pasta",
            "servings": 2,
        },
        "reason": "Stale proposal",
        "expected_plan_version": 99,  # Mismatch!
    }

    result = broker.execute(proposal)
    assert result.status == ActionStatus.REJECTED
    assert "Stale plan version" in result.message
    assert broker.get_plan_version() == version_before


def test_recalculation_failure_does_not_partially_commit(broker):
    """Verify that an error during derived basket calculation aborts transaction cleanly."""
    version_before = broker.get_plan_version()
    plan_snapshot = broker.get_plan()
    basket_snapshot = broker.get_basket()

    # Create a recipe with a unit conflict (ing_spinach in 'ml' instead of standard 'g')
    # This will pass policy checks (diet and allergen) but fail during basket aggregation!
    unit_conflict_recipe = Recipe(
        recipe_id="rec_unit_conflict",
        name="Unit Conflict Recipe",
        tags=["vegetarian"],
        default_servings=2,
        estimated_cost=Decimal("5.00"),
        requirements=[
            IngredientRequirement(ingredient_id="ing_spinach", quantity=Decimal("50.0"), unit="ml")  # 'ml' vs 'g'
        ],
    )
    broker.catalog["rec_unit_conflict"] = unit_conflict_recipe

    # Ensure another meal in the plan already uses ing_spinach in 'g' (e.g. Tomato Spinach Pasta on Monday)
    broker.catalog["rec_spinach_omelet"] = broker.catalog["rec_tomato_spinach_pasta"]

    proposal = {
        "operation": "replace_meal",
        "arguments": {
            "day": "Wednesday",
            "meal_type": "dinner",
            "recipe_id": "rec_unit_conflict",
            "servings": 2,
        },
        "reason": "Trigger unit conflict",
    }

    # Add a meal that uses ing_spinach in 'g' so aggregation sees a conflict
    spinach_meal = Meal(day="Tuesday", meal_type="dinner", recipe_id="rec_tomato_spinach_pasta", servings=2)
    broker._state.plan.set_meal(spinach_meal)
    plan_snapshot = broker.get_plan()
    version_before = broker.get_plan_version()

    result = broker.execute(proposal)
    assert result.status == ActionStatus.REJECTED
    assert "derived grocery basket calculation failed" in result.message.lower()

    # Authoritative state remains completely untouched
    assert broker.get_plan_version() == version_before
    assert broker.get_plan() == plan_snapshot
    assert broker.get_basket() == broker._state.basket


def test_authoritative_state_cannot_be_mutated_via_exposed_snapshots(broker):
    """Verify state encapsulation: modifying returned plan does not corrupt broker internal state."""
    # 1. Mutate returned plan
    exposed_plan = broker.get_plan()
    exposed_plan.meals.clear()
    exposed_plan.plan_version = 9999

    internal_plan = broker.get_plan()
    assert internal_plan.plan_version == 1
    assert len(internal_plan.meals) > 0

    # 2. Mutate returned basket items dictionary
    exposed_basket = broker.get_basket()
    exposed_basket.items.clear()

    internal_basket = broker.get_basket()
    assert len(internal_basket.items) > 0


def test_external_catalog_mutation_does_not_affect_broker(catalog_data, base_plan):
    """Verify that caller mutating their original catalog dict does not corrupt broker."""
    ingredients, recipes = catalog_data
    caller_catalog = dict(recipes)
    caller_ingredients = dict(ingredients)

    test_broker = ActionBroker(
        initial_plan=base_plan,
        catalog=caller_catalog,
        ingredients=caller_ingredients,
    )

    # Caller clears their catalog dictionary outside the broker
    caller_catalog.clear()
    caller_ingredients.clear()

    # Broker should still have all its catalog recipes and execute normally
    proposal = {
        "operation": "replace_meal",
        "arguments": {
            "day": "Wednesday",
            "meal_type": "dinner",
            "recipe_id": "rec_tomato_spinach_pasta",
            "servings": 2,
        },
        "reason": "Swap after catalog cleared outside",
    }
    result = test_broker.execute(proposal)
    assert result.status == ActionStatus.COMMITTED
    assert test_broker.get_plan_version() == 2


def test_postcondition_failure_aborts_transaction(broker, monkeypatch):
    """Verify that an artificial postcondition version mismatch aborts commit and preserves state."""
    version_before = broker.get_plan_version()
    plan_snapshot = broker.get_plan()
    basket_snapshot = broker.get_basket()

    # Monkeypatch calculate_grocery_basket to return an intentionally mismatched basket_version (999)
    from shopconnect_env.domain.models import GroceryBasket
    import shopconnect_env.validation.broker as broker_mod

    original_calc = broker_mod.calculate_grocery_basket

    def bad_calc(plan, catalog, ingredients):
        good = original_calc(plan, catalog, ingredients)
        return GroceryBasket(basket_version=999, items=good.items)

    monkeypatch.setattr(broker_mod, "calculate_grocery_basket", bad_calc)

    proposal = {
        "operation": "replace_meal",
        "arguments": {
            "day": "Wednesday",
            "meal_type": "dinner",
            "recipe_id": "rec_tomato_spinach_pasta",
            "servings": 2,
        },
        "reason": "Trigger postcondition mismatch",
    }

    result = broker.execute(proposal)
    assert result.status == ActionStatus.REJECTED
    assert "postcondition invariant broken" in result.message.lower()

    # Original state must be completely untouched
    assert broker.get_plan_version() == version_before
    assert broker.get_plan() == plan_snapshot
    assert broker.get_basket() == basket_snapshot


def test_empty_and_whitespace_json_rejected(broker):
    """Verify that empty strings, whitespace, or invalid non-dict payloads are rejected safely."""
    version_before = broker.get_plan_version()

    assert broker.execute("").status == ActionStatus.REJECTED
    assert broker.execute("   ").status == ActionStatus.REJECTED
    assert broker.execute("null").status == ActionStatus.REJECTED
    assert broker.execute("[]").status == ActionStatus.REJECTED
    assert broker.execute(12345).status == ActionStatus.REJECTED  # type: ignore

    assert broker.get_plan_version() == version_before


def test_boolean_and_float_servings_rejected_by_broker(broker):
    """Verify that boolean or float servings in proposal dictionaries are rejected."""
    version_before = broker.get_plan_version()

    # Boolean servings
    result_bool = broker.execute({
        "operation": "replace_meal",
        "arguments": {"day": "Wednesday", "meal_type": "dinner", "recipe_id": "rec_01", "servings": True},
        "reason": "Boolean serving test",
    })
    assert result_bool.status == ActionStatus.REJECTED
    assert "boolean" in result_bool.message

    # Float servings
    result_float = broker.execute({
        "operation": "change_servings",
        "arguments": {"day": "Monday", "meal_type": "breakfast", "servings": 3.5},
        "reason": "Float serving test",
    })
    assert result_float.status == ActionStatus.REJECTED
    assert "float" in result_float.message

    assert broker.get_plan_version() == version_before
