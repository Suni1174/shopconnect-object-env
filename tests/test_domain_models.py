"""Unit tests for Stage 1: Core Domain Models & Seed Data.

Verifies:
- Dataclass instantiation & validation.
- Decimal-based precision for money & quantities.
- 7-day x 3-meal slot schedule representation.
- Ingredient references & requirement scaling.
- Deterministic GroceryBasket derived calculation.
- Loading and structural validation of data/catalog_seed.json.
"""

from decimal import Decimal
from pathlib import Path
import pytest

from shopconnect_env.domain.models import (
    DAYS_OF_WEEK,
    MEAL_TYPES,
    GroceryBasket,
    GroceryItem,
    Ingredient,
    IngredientRequirement,
    Meal,
    MealPlan,
    Recipe,
    UserPreferences,
    load_catalog_from_json,
    make_slot_key,
)


def test_ingredient_creation():
    """Verify Ingredient instantiation and immutability."""
    ing = Ingredient(
        ingredient_id="ing_oats",
        name="Rolled Oats",
        category="Grains",
        standard_unit="g",
    )
    assert ing.ingredient_id == "ing_oats"
    assert ing.name == "Rolled Oats"
    assert ing.standard_unit == "g"

    # Verify frozen/immutable behavior
    with pytest.raises(Exception):
        ing.name = "Steel Cut Oats"  # type: ignore


def test_ingredient_requirement_and_scaling():
    """Verify IngredientRequirement creation, Decimal precision, and scaling."""
    req = IngredientRequirement(
        ingredient_id="ing_oats",
        quantity=Decimal("100.0"),
        unit="g",
    )
    assert isinstance(req.quantity, Decimal)
    assert req.quantity == Decimal("100.0")

    # Scaling factor of 1.5 (e.g. 3 servings instead of 2)
    scaled = req.scale(Decimal("1.5"))
    assert scaled.quantity == Decimal("150.0")
    assert scaled.ingredient_id == "ing_oats"
    assert scaled.unit == "g"

    # Scale factor must be positive
    with pytest.raises(ValueError):
        req.scale(Decimal("0"))


def test_recipe_cost_per_serving():
    """Verify Recipe cost calculation basis."""
    req = IngredientRequirement(
        ingredient_id="ing_eggs",
        quantity=Decimal("4"),
        unit="unit",
    )
    recipe = Recipe(
        recipe_id="rec_omelet",
        name="Spinach Omelet",
        tags=["vegetarian", "high-protein"],
        default_servings=2,
        estimated_cost=Decimal("6.50"),
        requirements=[req],
    )

    # 6.50 / 2 = 3.25 per serving
    assert recipe.cost_per_serving == Decimal("3.25")
    assert isinstance(recipe.cost_per_serving, Decimal)


def test_meal_slot_keys_and_validation():
    """Verify standard slot key generation and invalid slot protection."""
    key = make_slot_key("Wednesday", "dinner")
    assert key == "Wednesday_dinner"

    # Case insensitivity & normalization
    key_messy = make_slot_key(" wednesday ", " DINNER ")
    assert key_messy == "Wednesday_dinner"

    # Invalid day or meal type must raise ValueError
    with pytest.raises(ValueError, match="Invalid day"):
        make_slot_key("Funday", "dinner")

    with pytest.raises(ValueError, match="Invalid meal_type"):
        make_slot_key("Monday", "midnight_snack")


def test_meal_cost_calculation():
    """Verify Meal computes its cost using assigned recipe and servings."""
    recipe = Recipe(
        recipe_id="rec_soup",
        name="Lentil Soup",
        tags=["vegan"],
        default_servings=4,
        estimated_cost=Decimal("8.00"),  # $2.00 per serving
        requirements=[],
    )

    meal = Meal(
        day="Tuesday",
        meal_type="lunch",
        recipe_id="rec_soup",
        servings=3,
    )
    assert meal.slot_key == "Tuesday_lunch"

    # Cost for 3 servings at $2.00/serving = $6.00
    cost = meal.calculate_cost(recipe)
    assert cost == Decimal("6.00")
    assert isinstance(cost, Decimal)


def test_meal_plan_slot_representation():
    """Verify MealPlan holds 21 unique slots across 7 days."""
    plan = MealPlan(plan_id="plan_week_42", plan_version=1)

    expected_slots = MealPlan.all_expected_slot_keys()
    assert len(expected_slots) == 21
    assert "Monday_breakfast" in expected_slots
    assert "Sunday_dinner" in expected_slots
    assert not plan.is_complete()

    # Fill one slot
    meal = Meal(
        day="Monday",
        meal_type="breakfast",
        recipe_id="rec_berry_oatmeal",
        servings=2,
    )
    plan.set_meal(meal)

    retrieved = plan.get_meal("Monday", "breakfast")
    assert retrieved is not None
    assert retrieved.recipe_id == "rec_berry_oatmeal"
    assert retrieved.servings == 2
    assert plan.get_meal("Monday", "lunch") is None


def test_user_preferences_defaults():
    """Verify UserPreferences default values and constraints."""
    prefs = UserPreferences()
    assert prefs.weekly_budget == Decimal("150.00")
    assert prefs.household_size == 2

    # Negative budget should fail
    with pytest.raises(ValueError):
        UserPreferences(weekly_budget=Decimal("-10.00"))


def test_load_catalog_seed():
    """Verify loading and parsing the seed catalog from disk."""
    catalog_path = Path(__file__).parent.parent / "data" / "catalog_seed.json"
    assert catalog_path.exists(), f"Catalog seed file missing at {catalog_path}"

    ingredients, recipes = load_catalog_from_json(catalog_path)

    # Check that ingredients and recipes were parsed
    assert len(ingredients) >= 30
    assert len(recipes) >= 15

    # Check specific items
    assert "ing_spinach" in ingredients
    assert "rec_berry_oatmeal" in recipes

    # Check recipe integrity
    for rec_id, recipe in recipes.items():
        assert isinstance(recipe.estimated_cost, Decimal)
        assert recipe.default_servings > 0
        assert len(recipe.requirements) > 0
        # All requirements must point to existing canonical ingredients
        for req in recipe.requirements:
            assert req.ingredient_id in ingredients, (
                f"Recipe {rec_id} references unknown ingredient: {req.ingredient_id}"
            )
            assert isinstance(req.quantity, Decimal)


def test_derived_grocery_basket_calculation():
    """Verify that GroceryBasket is purely derived and scales correctly."""
    catalog_path = Path(__file__).parent.parent / "data" / "catalog_seed.json"
    ingredients, recipes = load_catalog_from_json(catalog_path)

    plan = MealPlan(plan_id="plan_test_01", plan_version=2)

    # Assign two meals that both use spinach:
    # 1. Spinach Omelet (default 2 servings, 80g spinach). We schedule 2 servings -> 80g.
    plan.set_meal(
        Meal(
            day="Monday",
            meal_type="breakfast",
            recipe_id="rec_spinach_omelet",
            servings=2,
        )
    )

    # 2. Grilled Chicken Wrap (default 2 servings, 100g spinach). We schedule 4 servings (2x) -> 200g.
    plan.set_meal(
        Meal(
            day="Monday",
            meal_type="lunch",
            recipe_id="rec_chicken_salad_wrap",
            servings=4,
        )
    )

    basket = GroceryBasket.from_meal_plan(plan, recipes, ingredients)

    assert basket.basket_version == 2
    # Spinach total should be 80g + 200g = 280g
    assert "ing_spinach" in basket.items
    spinach_item = basket.items["ing_spinach"]
    assert spinach_item.name == "Baby Spinach"
    assert spinach_item.quantity == Decimal("280.0")
    assert spinach_item.unit == "g"

    # Eggs: 4 units for 2 servings of omelet
    assert "ing_eggs" in basket.items
    assert basket.items["ing_eggs"].quantity == Decimal("4")

    # Immutability check: GroceryBasket is frozen, cannot be directly edited
    with pytest.raises(Exception):
        basket.basket_version = 99  # type: ignore
