"""Unit tests for Stage 2: Deterministic Operations and Pure Grocery Basket Aggregation.

Verifies:
1. Replacing Wednesday dinner.
2. Confirming other meal slots remain unchanged.
3. Changing serving counts.
4. Correct ingredient quantity scaling.
5. Aggregating shared ingredients across recipes.
6. Handling missing recipe IDs.
7. Handling unknown meal slots.
8. Handling invalid serving counts.
9. Confirming the original plan is unchanged when an operation fails.
10. Recalculating the basket after a meal replacement.
"""

import copy
from decimal import Decimal
from pathlib import Path
import pytest

from shopconnect_env.domain.models import (
    DAYS_OF_WEEK,
    MEAL_TYPES,
    GroceryBasket,
    Ingredient,
    IngredientRequirement,
    Meal,
    MealPlan,
    Recipe,
    UserPreferences,
    load_catalog_from_json,
    make_slot_key,
)
from shopconnect_env.operations.basket_calculator import calculate_grocery_basket
from shopconnect_env.operations.mutations import (
    apply_change_servings,
    apply_replace_meal,
)


@pytest.fixture
def catalog_data():
    """Load canonical ingredients and recipe catalog from seed JSON."""
    catalog_path = Path(__file__).parent.parent / "data" / "catalog_seed.json"
    ingredients, recipes = load_catalog_from_json(catalog_path)
    return ingredients, recipes


@pytest.fixture
def sample_plan(catalog_data):
    """Create a sample 7-day meal plan with several pre-populated slots."""
    ingredients, recipes = catalog_data
    plan = MealPlan(plan_id="plan_sample_week", plan_version=1)

    # Monday breakfast & dinner
    plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_berry_oatmeal", servings=2))
    plan.set_meal(Meal(day="Monday", meal_type="dinner", recipe_id="rec_red_lentil_dahl", servings=4))

    # Tuesday lunch
    plan.set_meal(Meal(day="Tuesday", meal_type="lunch", recipe_id="rec_tofu_stir_fry", servings=2))

    # Wednesday dinner (the one we will target in tests)
    plan.set_meal(Meal(day="Wednesday", meal_type="dinner", recipe_id="rec_baked_chicken_rice", servings=2))

    # Friday dinner
    plan.set_meal(Meal(day="Friday", meal_type="dinner", recipe_id="rec_veggie_fried_rice", servings=2))

    return plan


# 1. Replacing Wednesday dinner
def test_replace_wednesday_dinner(sample_plan, catalog_data):
    """Test replacing Wednesday dinner with a new recipe."""
    ingredients, recipes = catalog_data

    # Wednesday dinner starts as baked chicken
    original_wed_dinner = sample_plan.get_meal("Wednesday", "dinner")
    assert original_wed_dinner is not None
    assert original_wed_dinner.recipe_id == "rec_baked_chicken_rice"

    # Replace with pan-seared salmon
    updated_plan = apply_replace_meal(
        plan=sample_plan,
        day="Wednesday",
        meal_type="dinner",
        recipe_id="rec_pan_seared_salmon",
        catalog=recipes,
        servings=2,
    )

    new_wed_dinner = updated_plan.get_meal("Wednesday", "dinner")
    assert new_wed_dinner is not None
    assert new_wed_dinner.recipe_id == "rec_pan_seared_salmon"
    assert new_wed_dinner.servings == 2
    assert updated_plan.plan_version == sample_plan.plan_version + 1


# 2. Confirming other meal slots remain unchanged
def test_other_slots_remain_unchanged(sample_plan, catalog_data):
    """Confirm that modifying one slot does not alter any other slot."""
    ingredients, recipes = catalog_data

    updated_plan = apply_replace_meal(
        plan=sample_plan,
        day="Wednesday",
        meal_type="dinner",
        recipe_id="rec_pan_seared_salmon",
        catalog=recipes,
    )

    # Check that Monday breakfast, Monday dinner, Tuesday lunch, and Friday dinner are untouched
    for slot_key, original_meal in sample_plan.meals.items():
        if slot_key == "Wednesday_dinner":
            continue
        updated_meal = updated_plan.meals[slot_key]
        assert updated_meal.recipe_id == original_meal.recipe_id
        assert updated_meal.servings == original_meal.servings
        assert updated_meal.day == original_meal.day
        assert updated_meal.meal_type == original_meal.meal_type


# 3. Changing serving counts
def test_change_servings(sample_plan):
    """Test modifying the serving count of an existing scheduled meal."""
    original_tue_lunch = sample_plan.get_meal("Tuesday", "lunch")
    assert original_tue_lunch.servings == 2

    updated_plan = apply_change_servings(
        plan=sample_plan,
        day="Tuesday",
        meal_type="lunch",
        servings=4,
    )

    updated_tue_lunch = updated_plan.get_meal("Tuesday", "lunch")
    assert updated_tue_lunch.servings == 4
    assert updated_tue_lunch.recipe_id == "rec_tofu_stir_fry"
    assert updated_plan.plan_version == sample_plan.plan_version + 1


# 4. Correct ingredient quantity scaling
def test_ingredient_quantity_scaling(catalog_data):
    """Test that doubling servings doubles the required ingredient quantities."""
    ingredients, recipes = catalog_data

    plan_2_servings = MealPlan(plan_id="plan_scaling", plan_version=1)
    # rec_spinach_omelet default_servings is 2, requiring 80g spinach
    plan_2_servings.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_spinach_omelet", servings=2))

    basket_2 = calculate_grocery_basket(plan_2_servings, recipes, ingredients)
    assert basket_2.items["ing_spinach"].quantity == Decimal("80.0")

    # Scale to 4 servings
    plan_4_servings = apply_change_servings(plan_2_servings, "Monday", "breakfast", servings=4)
    basket_4 = calculate_grocery_basket(plan_4_servings, recipes, ingredients)

    # 80g * (4 / 2) = 160g
    assert basket_4.items["ing_spinach"].quantity == Decimal("160.0")


# 5. Aggregating shared ingredients across recipes
def test_aggregating_shared_ingredients(catalog_data):
    """Test that two different meals sharing an ingredient are aggregated into one item."""
    ingredients, recipes = catalog_data

    plan = MealPlan(plan_id="plan_shared", plan_version=1)
    # Meal 1: Spinach Omelet (2 servings -> 80g spinach)
    plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_spinach_omelet", servings=2))
    # Meal 2: Tomato Spinach Pasta (2 servings -> 100g spinach)
    plan.set_meal(Meal(day="Monday", meal_type="dinner", recipe_id="rec_tomato_spinach_pasta", servings=2))

    basket = calculate_grocery_basket(plan, recipes, ingredients)

    # Spinach total should be 80g + 100g = 180g
    assert "ing_spinach" in basket.items
    assert basket.items["ing_spinach"].quantity == Decimal("180.0")
    assert basket.items["ing_spinach"].unit == "g"


# 6. Handling missing recipe IDs
def test_missing_recipe_id_error(sample_plan, catalog_data):
    """Test that referencing a non-existent recipe raises KeyError."""
    ingredients, recipes = catalog_data

    with pytest.raises(KeyError, match="does not exist in catalog"):
        apply_replace_meal(
            plan=sample_plan,
            day="Wednesday",
            meal_type="dinner",
            recipe_id="rec_phantom_recipe_999",
            catalog=recipes,
        )


# 7. Handling unknown meal slots
def test_unknown_meal_slot_error(sample_plan, catalog_data):
    """Test that passing an invalid day or meal slot raises ValueError."""
    ingredients, recipes = catalog_data

    with pytest.raises(ValueError, match="Invalid day"):
        apply_replace_meal(
            plan=sample_plan,
            day="Funday",
            meal_type="dinner",
            recipe_id="rec_pan_seared_salmon",
            catalog=recipes,
        )

    with pytest.raises(ValueError, match="Invalid meal_type"):
        apply_replace_meal(
            plan=sample_plan,
            day="Wednesday",
            meal_type="fourth_meal",
            recipe_id="rec_pan_seared_salmon",
            catalog=recipes,
        )


# 8. Handling invalid serving counts
def test_invalid_serving_counts(sample_plan, catalog_data):
    """Test that zero, negative, or non-integer serving counts raise ValueError."""
    ingredients, recipes = catalog_data

    # Zero servings
    with pytest.raises(ValueError, match="positive integer"):
        apply_replace_meal(
            plan=sample_plan,
            day="Wednesday",
            meal_type="dinner",
            recipe_id="rec_pan_seared_salmon",
            catalog=recipes,
            servings=0,
        )

    # Negative servings
    with pytest.raises(ValueError, match="positive integer"):
        apply_change_servings(sample_plan, "Wednesday", "dinner", servings=-3)

    # Boolean or float servings
    with pytest.raises(ValueError, match="positive integer"):
        apply_change_servings(sample_plan, "Wednesday", "dinner", servings=True)  # type: ignore


# 9. Confirming original plan is unchanged when an operation fails
def test_original_plan_unchanged_on_failure(sample_plan, catalog_data):
    """Verify candidate state isolation: errors leave the original plan 100% untouched."""
    ingredients, recipes = catalog_data

    # Capture deepcopy snapshot of the original plan before attempting invalid action
    original_snapshot = copy.deepcopy(sample_plan)

    # Attempt invalid action
    try:
        apply_replace_meal(
            plan=sample_plan,
            day="Wednesday",
            meal_type="dinner",
            recipe_id="rec_nonexistent",
            catalog=recipes,
        )
    except KeyError:
        pass

    # Verify original plan is identical to snapshot
    assert sample_plan.plan_version == original_snapshot.plan_version
    assert len(sample_plan.meals) == len(original_snapshot.meals)
    for slot_key, meal in sample_plan.meals.items():
        snapshot_meal = original_snapshot.meals[slot_key]
        assert meal.recipe_id == snapshot_meal.recipe_id
        assert meal.servings == snapshot_meal.servings


# 10. Recalculating the basket after a meal replacement
def test_recalculate_basket_after_replacement(catalog_data):
    """Verify that replacing a meal updates the grocery basket deterministically."""
    ingredients, recipes = catalog_data

    # Start with a plan having only 1 meal: Spinach Omelet (uses eggs & spinach)
    initial_plan = MealPlan(plan_id="plan_swap", plan_version=1)
    initial_plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_spinach_omelet", servings=2))

    initial_basket = calculate_grocery_basket(initial_plan, recipes, ingredients)
    assert "ing_eggs" in initial_basket.items
    assert "ing_spinach" in initial_basket.items
    assert "ing_rolled_oats" not in initial_basket.items
    assert initial_basket.basket_version == 1

    # Replace Monday breakfast with Berry Oatmeal Bowl (uses rolled oats, almond milk, berries, chia seeds)
    updated_plan = apply_replace_meal(
        plan=initial_plan,
        day="Monday",
        meal_type="breakfast",
        recipe_id="rec_berry_oatmeal",
        catalog=recipes,
        servings=2,
    )

    updated_basket = calculate_grocery_basket(updated_plan, recipes, ingredients)

    # Eggs and spinach must be gone from the basket
    assert "ing_eggs" not in updated_basket.items
    assert "ing_spinach" not in updated_basket.items

    # Oatmeal ingredients must now be in the basket
    assert "ing_rolled_oats" in updated_basket.items
    assert updated_basket.items["ing_rolled_oats"].quantity == Decimal("100.0")
    assert "ing_berries" in updated_basket.items

    # Basket version must match the updated plan version (2)
    assert updated_basket.basket_version == updated_plan.plan_version
    assert updated_plan.plan_version == 2
