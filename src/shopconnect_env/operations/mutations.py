"""Deterministic state mutations for the ShopConnect MealPlan.

These functions provide pure, side-effect-free operations for updating meal plans.
They never mutate the input MealPlan in place; instead, they stage changes on a
deep copy and return the updated plan with an incremented plan_version.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

from shopconnect_env.domain.models import Meal, MealPlan, Recipe, make_slot_key


def apply_replace_meal(
    plan: MealPlan,
    day: str,
    meal_type: str,
    recipe_id: str,
    catalog: Dict[str, Recipe],
    servings: Optional[int] = None,
) -> MealPlan:
    """Replace the meal assigned to a specific schedule slot.
    
    Guarantees:
        - Only the specified (day, meal_type) slot is modified.
        - All other scheduled meals remain byte-identical.
        - Input `plan` is never mutated in-place (candidate staging).
        - Returns a new MealPlan instance with incremented plan_version.
        
    Args:
        plan: The authoritative current MealPlan.
        day: Day of the week (e.g. 'Wednesday').
        meal_type: Meal slot ('breakfast', 'lunch', or 'dinner').
        recipe_id: Unique recipe identifier to assign.
        catalog: Dictionary mapping recipe_id to Recipe instances.
        servings: Optional target serving count. If None, defaults to the
                  recipe's default_servings.
                  
    Returns:
        A new MealPlan with the replacement applied.
        
    Raises:
        ValueError: If day or meal_type is invalid, or if servings <= 0.
        KeyError: If recipe_id is not in catalog.
    """
    # 1. Validate slot coordinates (raises ValueError if invalid)
    slot_key = make_slot_key(day, meal_type)

    # 2. Validate recipe exists in catalog
    if recipe_id not in catalog:
        raise KeyError(
            f"Cannot replace meal: recipe '{recipe_id}' does not exist in catalog."
        )
    target_recipe = catalog[recipe_id]

    # 3. Resolve and validate servings
    resolved_servings = (
        target_recipe.default_servings if servings is None else servings
    )
    if (
        not isinstance(resolved_servings, int)
        or isinstance(resolved_servings, bool)
        or resolved_servings <= 0
    ):
        raise ValueError(
            f"Servings must be a positive integer, got: {resolved_servings!r}."
        )

    # 4. Clone state to prevent accidental mutation of the original plan
    staged_plan = copy.deepcopy(plan)

    # 5. Apply the replacement to the candidate clone
    new_meal = Meal(
        day=day,
        meal_type=meal_type,
        recipe_id=recipe_id,
        servings=resolved_servings,
    )
    staged_plan.set_meal(new_meal)

    # 6. Increment plan version
    staged_plan.plan_version = plan.plan_version + 1

    return staged_plan


def apply_change_servings(
    plan: MealPlan,
    day: str,
    meal_type: str,
    servings: int,
) -> MealPlan:
    """Change the serving count for an existing scheduled meal.
    
    Guarantees:
        - Only the specified meal's servings are modified.
        - The assigned recipe and all other scheduled meals remain unchanged.
        - Input `plan` is never mutated in-place.
        - Returns a new MealPlan instance with incremented plan_version.
        
    Args:
        plan: The authoritative current MealPlan.
        day: Day of the week (e.g. 'Tuesday').
        meal_type: Meal slot ('breakfast', 'lunch', or 'dinner').
        servings: New positive integer serving count.
        
    Returns:
        A new MealPlan with the scaled serving count.
        
    Raises:
        ValueError: If day or meal_type is invalid, or if servings <= 0.
        KeyError: If no meal is currently scheduled at the given slot.
    """
    # 1. Validate slot coordinates (raises ValueError if invalid)
    slot_key = make_slot_key(day, meal_type)

    # 2. Validate servings
    if (
        not isinstance(servings, int)
        or isinstance(servings, bool)
        or servings <= 0
    ):
        raise ValueError(
            f"Servings must be a positive integer, got: {servings!r}."
        )

    # 3. Verify a meal exists at this slot
    if slot_key not in plan.meals:
        raise KeyError(
            f"Cannot change servings: slot '{slot_key}' has no scheduled meal."
        )

    # 4. Clone state to prevent in-place mutation
    staged_plan = copy.deepcopy(plan)

    # 5. Update servings on the cloned meal
    existing_meal = staged_plan.meals[slot_key]
    staged_plan.meals[slot_key] = Meal(
        day=existing_meal.day,
        meal_type=existing_meal.meal_type,
        recipe_id=existing_meal.recipe_id,
        servings=servings,
    )

    # 6. Increment plan version
    staged_plan.plan_version = plan.plan_version + 1

    return staged_plan
