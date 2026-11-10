"""Unit tests for Stage 3 Policy Engine."""

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
from shopconnect_env.validation.policy_engine import (
    PolicyDecision,
    PolicyEngine,
    allergens_conflict,
    normalize_allergen,
)


@pytest.fixture
def catalog_data():
    """Load canonical ingredients and recipe catalog from seed JSON."""
    catalog_path = Path(__file__).parent.parent / "data" / "catalog_seed.json"
    ingredients, recipes = load_catalog_from_json(catalog_path)
    return ingredients, recipes


@pytest.fixture
def engine(catalog_data):
    """Instantiate a PolicyEngine with seed catalog."""
    ingredients, recipes = catalog_data
    return PolicyEngine(catalog=recipes, ingredients=ingredients)


def test_vegetarian_recipe_allowed(engine):
    """Verify that a vegetarian recipe is allowed for vegetarian preferences."""
    plan = MealPlan(
        plan_id="plan_veg",
        preferences=UserPreferences(diet_tags=["vegetarian"]),
    )
    # rec_red_lentil_dahl is vegetarian
    plan.set_meal(Meal(day="Monday", meal_type="dinner", recipe_id="rec_red_lentil_dahl", servings=2))

    decision = engine.validate_dietary_restrictions(plan)
    assert decision.allowed is True
    assert decision.rule_name == "dietary_restrictions"


def test_non_vegetarian_recipe_rejected(engine):
    """Verify that a meat recipe is rejected when user requires vegetarian."""
    plan = MealPlan(
        plan_id="plan_veg",
        preferences=UserPreferences(diet_tags=["vegetarian"]),
    )
    # rec_baked_chicken_rice is not vegetarian
    plan.set_meal(Meal(day="Monday", meal_type="dinner", recipe_id="rec_baked_chicken_rice", servings=2))

    decision = engine.validate_dietary_restrictions(plan)
    assert decision.allowed is False
    assert decision.rule_name == "dietary_restrictions"
    assert "violates dietary restriction" in decision.reason
    assert "vegetarian" in decision.reason


def test_multiple_dietary_constraints(engine):
    """Verify that multiple dietary tags must all be satisfied."""
    plan = MealPlan(
        plan_id="plan_multi",
        preferences=UserPreferences(diet_tags=["vegetarian", "dairy-free"]),
    )
    # rec_spinach_omelet is vegetarian, but contains parmesan cheese (dairy)
    plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_spinach_omelet", servings=2))

    decision = engine.validate_dietary_restrictions(plan)
    assert decision.allowed is False
    assert "dairy-free" in decision.reason

    # Replace with berry oatmeal, which is both vegetarian and dairy-free
    plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_berry_oatmeal", servings=2))
    decision = engine.validate_dietary_restrictions(plan)
    assert decision.allowed is True


def test_peanut_allergy_rejects_peanut_recipe(engine):
    """Verify that a recipe containing peanut butter is rejected for peanut allergy."""
    plan = MealPlan(
        plan_id="plan_allergy",
        preferences=UserPreferences(allergies=["peanuts"]),
    )
    # rec_peanut_butter_banana_toast contains ing_peanut_butter
    plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_peanut_butter_banana_toast", servings=2))

    decision = engine.validate_allergens(plan)
    assert decision.allowed is False
    assert decision.rule_name == "allergen_safety"
    assert "Allergen conflict detected" in decision.reason
    assert "peanuts" in decision.reason


def test_known_non_matching_allergen_allowed(engine):
    """Verify that an allergen-safe recipe is allowed."""
    plan = MealPlan(
        plan_id="plan_allergy",
        preferences=UserPreferences(allergies=["peanuts"]),
    )
    # rec_berry_oatmeal has no peanuts
    plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_berry_oatmeal", servings=2))

    decision = engine.validate_allergens(plan)
    assert decision.allowed is True
    assert decision.rule_name == "allergen_safety"


def test_unknown_allergen_metadata_conservatively_rejected(catalog_data):
    """Verify conservative safety rule: unknown allergen metadata (None) is rejected if user has allergies."""
    ingredients, recipes = catalog_data

    # Create a custom ingredient with allergens=None
    mystery_sauce = Ingredient(
        ingredient_id="ing_mystery_sauce",
        name="Chef Special Mystery Sauce",
        category="Condiments",
        standard_unit="ml",
        allergens=None,  # Unknown metadata
    )
    test_ingredients = dict(ingredients)
    test_ingredients[mystery_sauce.ingredient_id] = mystery_sauce

    # Create a recipe using the mystery sauce
    mystery_recipe = Recipe(
        recipe_id="rec_mystery_dish",
        name="Mystery Stir Fry",
        tags=["quick"],
        default_servings=2,
        estimated_cost=Decimal("5.00"),
        requirements=[
            IngredientRequirement(ingredient_id=mystery_sauce.ingredient_id, quantity=Decimal("20.0"), unit="ml")
        ],
    )
    test_recipes = dict(recipes)
    test_recipes[mystery_recipe.recipe_id] = mystery_recipe

    test_engine = PolicyEngine(catalog=test_recipes, ingredients=test_ingredients)

    plan = MealPlan(
        plan_id="plan_conservative",
        preferences=UserPreferences(allergies=["dairy"]),  # User has an active allergy
    )
    plan.set_meal(Meal(day="Friday", meal_type="dinner", recipe_id="rec_mystery_dish", servings=2))

    decision = test_engine.validate_allergens(plan)
    assert decision.allowed is False
    assert "Conservative allergen rejection" in decision.reason
    assert "unknown allergen metadata" in decision.reason


def test_budget_compliant_candidate(engine):
    """Verify that a plan within budget passes validation."""
    plan = MealPlan(
        plan_id="plan_budget",
        preferences=UserPreferences(weekly_budget=Decimal("50.00")),
    )
    # Cost is $4.20
    plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_berry_oatmeal", servings=2))

    decision = engine.validate_budget(plan)
    assert decision.allowed is True
    assert decision.rule_name == "weekly_budget"


def test_budget_exceeding_candidate_rejected(engine):
    """Verify that a plan exceeding the weekly budget is rejected."""
    plan = MealPlan(
        plan_id="plan_budget",
        preferences=UserPreferences(weekly_budget=Decimal("10.00")),  # Low budget cap
    )
    # Pan-seared salmon is $14.50 for 2 servings
    plan.set_meal(Meal(day="Wednesday", meal_type="dinner", recipe_id="rec_pan_seared_salmon", servings=2))

    decision = engine.validate_budget(plan)
    assert decision.allowed is False
    assert decision.rule_name == "weekly_budget"
    assert "exceeds configured weekly budget limit" in decision.reason


def test_missing_recipe_reference_in_plan(engine):
    """Verify handling when a scheduled meal references a non-existent recipe."""
    plan = MealPlan(plan_id="plan_missing")
    plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_phantom", servings=2))

    decision = engine.validate_budget(plan)
    assert decision.allowed is False
    assert "missing recipe" in decision.reason.lower()


def test_incomplete_plan_cost_semantics_documented(engine):
    """Verify budget validation details report is_complete_plan flag."""
    plan = MealPlan(
        plan_id="plan_partial",
        preferences=UserPreferences(weekly_budget=Decimal("100.00")),
    )
    plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_berry_oatmeal", servings=2))

    decision = engine.validate_budget(plan)
    assert decision.allowed is True
    assert decision.details is not None
    assert decision.details["is_complete_plan"] is False
    assert decision.details["scheduled_slots_count"] == 1


def test_invariant_validation_detects_unintended_changes(engine):
    """Verify that validate_invariants catches version discrepancies and slot tampering."""
    orig_plan = MealPlan(plan_id="p1", plan_version=1)
    orig_plan.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_berry_oatmeal", servings=2))
    orig_plan.set_meal(Meal(day="Monday", meal_type="dinner", recipe_id="rec_red_lentil_dahl", servings=4))

    # Candidate modifying Wednesday dinner (intended), but version did not increment
    bad_cand_1 = MealPlan(plan_id="p1", plan_version=1)  # Version didn't change!
    bad_cand_1.meals = dict(orig_plan.meals)
    bad_cand_1.set_meal(Meal(day="Wednesday", meal_type="dinner", recipe_id="rec_pan_seared_salmon", servings=2))

    dec_1 = engine.validate_invariants(orig_plan, bad_cand_1, target_slot_key="Wednesday_dinner")
    assert dec_1.allowed is False
    assert "Invalid plan version transition" in dec_1.reason

    # Candidate modified Monday breakfast even though target was Wednesday dinner
    bad_cand_2 = MealPlan(plan_id="p1", plan_version=2)
    bad_cand_2.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_spinach_omelet", servings=2))  # Drift!
    bad_cand_2.set_meal(Meal(day="Monday", meal_type="dinner", recipe_id="rec_red_lentil_dahl", servings=4))
    bad_cand_2.set_meal(Meal(day="Wednesday", meal_type="dinner", recipe_id="rec_pan_seared_salmon", servings=2))

    dec_2 = engine.validate_invariants(orig_plan, bad_cand_2, target_slot_key="Wednesday_dinner")
    assert dec_2.allowed is False
    assert "Non-target meal slot 'Monday_breakfast' was unexpectedly modified" in dec_2.reason


def test_peanuts_and_tree_nuts_are_distinct(engine):
    """Verify that peanut and tree nut classifications are never treated as identical."""
    # 1. Function-level verification
    assert allergens_conflict("peanuts", "peanuts") is True
    assert allergens_conflict("peanut", "peanuts") is True
    assert allergens_conflict("tree_nuts", "tree_nuts") is True
    assert allergens_conflict("peanuts", "tree_nuts") is False
    assert allergens_conflict("tree_nuts", "peanuts") is False

    # 2. Plan-level verification: Peanut-allergic user can eat almond milk (tree nut)
    plan_peanut_allergic = MealPlan(
        plan_id="plan_p_allergy",
        preferences=UserPreferences(allergies=["peanuts"]),
    )
    # rec_chia_pudding uses almond milk (tree_nuts), but NO peanuts
    plan_peanut_allergic.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_chia_pudding", servings=2))
    decision = engine.validate_allergens(plan_peanut_allergic)
    assert decision.allowed is True


def test_wheat_and_gluten_directional_safety(engine, catalog_data):
    """Verify that wheat and gluten have a strict directional relationship without false equivalence."""
    # 1. Function-level verification
    assert allergens_conflict("gluten", "wheat") is True   # Wheat contains gluten
    assert allergens_conflict("wheat", "wheat") is True     # Direct wheat match
    assert allergens_conflict("wheat", "gluten") is False   # Gluten alone is not claimed to be wheat
    assert allergens_conflict("gluten", "gluten") is True   # Direct gluten match

    # 2. Plan-level verification: User with wheat allergy avoids wheat bread
    plan_wheat = MealPlan(
        plan_id="plan_wheat",
        preferences=UserPreferences(allergies=["wheat"]),
    )
    # rec_avocado_toast uses whole wheat bread
    plan_wheat.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_avocado_toast", servings=2))
    assert engine.validate_allergens(plan_wheat).allowed is False

    # 3. Plan-level verification: User with gluten allergy also avoids wheat bread
    plan_gluten = MealPlan(
        plan_id="plan_gluten",
        preferences=UserPreferences(allergies=["gluten"]),
    )
    plan_gluten.set_meal(Meal(day="Monday", meal_type="breakfast", recipe_id="rec_avocado_toast", servings=2))
    assert engine.validate_allergens(plan_gluten).allowed is False


def test_unknown_allergen_allowed_when_user_has_no_allergies(catalog_data):
    """Verify that ingredients with unknown metadata (None) are permitted when user has NO allergies."""
    ingredients, recipes = catalog_data

    # Ingredient with unknown metadata
    mystery_ing = Ingredient(
        ingredient_id="ing_unknown_spice",
        name="Wild Forest Spice",
        category="Spices",
        standard_unit="g",
        allergens=None,
    )
    test_ing = dict(ingredients)
    test_ing[mystery_ing.ingredient_id] = mystery_ing

    test_rec = Recipe(
        recipe_id="rec_spice_dish",
        name="Spiced Rice",
        tags=["vegetarian"],
        default_servings=2,
        estimated_cost=Decimal("3.00"),
        requirements=[IngredientRequirement(ingredient_id="ing_unknown_spice", quantity=Decimal("5.0"), unit="g")],
    )
    test_recipes = dict(recipes)
    test_recipes[test_rec.recipe_id] = test_rec

    test_engine = PolicyEngine(catalog=test_recipes, ingredients=test_ing)

    # User with empty allergies list -> Should be allowed
    plan_safe = MealPlan(
        plan_id="plan_no_allergies",
        preferences=UserPreferences(allergies=[]),
    )
    plan_safe.set_meal(Meal(day="Monday", meal_type="dinner", recipe_id="rec_spice_dish", servings=2))
    decision = test_engine.validate_allergens(plan_safe)
    assert decision.allowed is True
