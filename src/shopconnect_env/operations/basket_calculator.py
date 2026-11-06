"""Deterministic GroceryBasket calculation engine.

Calculates an aggregated grocery shopping basket from an active MealPlan
and canonical recipe/ingredient catalogs.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict

from shopconnect_env.domain.models import (
    GroceryBasket,
    GroceryItem,
    Ingredient,
    MealPlan,
    Recipe,
)


def calculate_grocery_basket(
    plan: MealPlan,
    catalog: Dict[str, Recipe],
    ingredients: Dict[str, Ingredient],
) -> GroceryBasket:
    """Deterministically aggregate required groceries for all scheduled meals.
    
    Calculation Protocol:
        1. For each scheduled meal in the plan, look up its Recipe in `catalog`.
        2. Calculate the scaling factor: meal.servings / recipe.default_servings.
        3. For each ingredient requirement, scale quantity by the factor.
        4. Validate that ingredient_id exists in `ingredients`.
        5. Aggregate quantities for matching ingredient IDs. Verify matching units.
        6. Return an immutable GroceryBasket with items sorted by ingredient_id
           for 100% deterministic, reproducible output.
           
    Args:
        plan: The active MealPlan holding scheduled meals.
        catalog: Dictionary mapping recipe_id to Recipe.
        ingredients: Dictionary mapping ingredient_id to Ingredient.
        
    Returns:
        An immutable GroceryBasket whose basket_version matches plan.plan_version.
        
    Raises:
        KeyError: If a scheduled recipe_id is missing from catalog, or if
                  an ingredient_id is missing from ingredients.
        ValueError: If the same ingredient is required in conflicting units.
    """
    aggregated_quantities: Dict[str, Decimal] = {}
    units_map: Dict[str, str] = {}

    for meal in plan.meals.values():
        if meal.recipe_id not in catalog:
            raise KeyError(
                f"Cannot calculate grocery basket: recipe '{meal.recipe_id}' "
                f"scheduled for '{meal.slot_key}' is missing from catalog."
            )
        recipe = catalog[meal.recipe_id]

        # Exact Decimal scaling factor
        scale_factor = Decimal(meal.servings) / Decimal(recipe.default_servings)

        for req in recipe.requirements:
            if req.ingredient_id not in ingredients:
                raise KeyError(
                    f"Cannot calculate grocery basket: ingredient '{req.ingredient_id}' "
                    f"required by recipe '{recipe.recipe_id}' is missing from catalog."
                )

            scaled_qty = req.quantity * scale_factor

            if req.ingredient_id in aggregated_quantities:
                existing_unit = units_map[req.ingredient_id]
                if existing_unit != req.unit:
                    raise ValueError(
                        f"Unit conflict for ingredient '{req.ingredient_id}': "
                        f"found '{existing_unit}' and '{req.unit}'."
                    )
                aggregated_quantities[req.ingredient_id] += scaled_qty
            else:
                aggregated_quantities[req.ingredient_id] = scaled_qty
                units_map[req.ingredient_id] = req.unit

    # Construct GroceryItem instances sorted by ingredient_id for determinism
    sorted_items: Dict[str, GroceryItem] = {}
    for ing_id in sorted(aggregated_quantities.keys()):
        ing = ingredients[ing_id]
        sorted_items[ing_id] = GroceryItem(
            ingredient_id=ing_id,
            name=ing.name,
            category=ing.category,
            quantity=aggregated_quantities[ing_id],
            unit=units_map[ing_id],
        )

    return GroceryBasket(
        basket_version=plan.plan_version,
        items=sorted_items,
    )
