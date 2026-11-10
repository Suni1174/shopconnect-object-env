"""Shared data structures, serialization helpers, and base classes for context strategies.

This module defines the standardized input (ContextRequest) and output (ContextResult)
contracts consumed by all experimental context builders (Conditions A, B, and C).
It also provides deterministic, Decimal-safe serialization utilities to ensure
that all context builders produce stable, comparable representations of application state.
"""

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Tuple, Union

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
    make_slot_key,
)
from shopconnect_env.operations.basket_calculator import calculate_grocery_basket


class ConditionType(str, Enum):
    """Identifier for the experimental context condition."""

    CONDITION_A = "A"
    CONDITION_B = "B"
    CONDITION_C = "C"


def serialize_decimal(val: Union[Decimal, int, float, str]) -> str:
    """Format a monetary or numeric value as a deterministic string.

    Prevents floating-point representation drift (e.g. 0.10000000000000001).
    """
    if isinstance(val, Decimal):
        return str(val)
    if isinstance(val, (int, str)):
        return str(val)
    return f"{val:.2f}"


def serialize_ingredient(ingredient: Ingredient) -> Dict[str, Any]:
    """Serialize an Ingredient to a deterministic dictionary."""
    return {
        "ingredient_id": ingredient.ingredient_id,
        "name": ingredient.name,
        "category": ingredient.category,
        "standard_unit": ingredient.standard_unit,
        "allergens": sorted(list(ingredient.allergens)) if ingredient.allergens is not None else None,
    }


def serialize_ingredient_requirement(
    req: IngredientRequirement,
    ingredient: Optional[Ingredient] = None,
) -> Dict[str, Any]:
    """Serialize an IngredientRequirement to a deterministic dictionary."""
    return {
        "ingredient_id": req.ingredient_id,
        "name": ingredient.name if ingredient else req.ingredient_id,
        "quantity": serialize_decimal(req.quantity),
        "unit": req.unit,
    }


def serialize_recipe(
    recipe: Recipe,
    ingredients: Optional[Dict[str, Ingredient]] = None,
) -> Dict[str, Any]:
    """Serialize a Recipe to a deterministic dictionary."""
    sorted_reqs = sorted(recipe.requirements, key=lambda r: r.ingredient_id)
    reqs_serialized = [
        serialize_ingredient_requirement(
            req,
            ingredients.get(req.ingredient_id) if ingredients else None,
        )
        for req in sorted_reqs
    ]
    return {
        "recipe_id": recipe.recipe_id,
        "name": recipe.name,
        "tags": sorted(list(recipe.tags)),
        "default_servings": recipe.default_servings,
        "estimated_cost": serialize_decimal(recipe.estimated_cost),
        "cost_per_serving": serialize_decimal(recipe.cost_per_serving),
        "ingredient_requirements": reqs_serialized,
    }


def serialize_meal(meal: Meal, catalog: Dict[str, Recipe]) -> Dict[str, Any]:
    """Serialize a scheduled Meal to a deterministic dictionary."""
    recipe = catalog.get(meal.recipe_id)
    estimated_cost = (
        serialize_decimal(meal.calculate_cost(recipe))
        if recipe
        else "0.00"
    )
    return {
        "slot_key": meal.slot_key,
        "day": meal.day,
        "meal_type": meal.meal_type,
        "status": "assigned",
        "recipe_id": meal.recipe_id,
        "recipe_name": recipe.name if recipe else meal.recipe_id,
        "servings": meal.servings,
        "estimated_cost": estimated_cost,
    }


def serialize_unfilled_slot(day: str, meal_type: str) -> Dict[str, Any]:
    """Serialize an unfilled meal slot with explicit representation."""
    return {
        "slot_key": make_slot_key(day, meal_type),
        "day": day,
        "meal_type": meal_type,
        "status": "unfilled",
        "recipe_id": None,
        "recipe_name": None,
        "servings": 0,
        "estimated_cost": "0.00",
    }


def serialize_all_21_slots(
    plan: MealPlan,
    catalog: Dict[str, Recipe],
) -> List[Dict[str, Any]]:
    """Return all 21 slots (Monday through Sunday, 3 meals/day) in canonical order.

    Unfilled slots are explicitly marked with status='unfilled' to prevent
    ambiguity or state hallucination.
    """
    slots: List[Dict[str, Any]] = []
    for day in DAYS_OF_WEEK:
        for meal_type in MEAL_TYPES:
            slot_key = make_slot_key(day, meal_type)
            if slot_key in plan.meals:
                slots.append(serialize_meal(plan.meals[slot_key], catalog))
            else:
                slots.append(serialize_unfilled_slot(day, meal_type))
    return slots


def serialize_user_preferences(prefs: UserPreferences) -> Dict[str, Any]:
    """Serialize UserPreferences to a deterministic dictionary."""
    return {
        "diet_tags": sorted(list(prefs.diet_tags)),
        "allergies": sorted(list(prefs.allergies)),
        "weekly_budget": serialize_decimal(prefs.weekly_budget),
        "household_size": prefs.household_size,
    }


def serialize_grocery_item(item: GroceryItem) -> Dict[str, Any]:
    """Serialize a single GroceryItem to a deterministic dictionary."""
    return {
        "ingredient_id": item.ingredient_id,
        "name": item.name,
        "category": item.category,
        "quantity": serialize_decimal(item.quantity),
        "unit": item.unit,
    }


def serialize_grocery_basket(basket: GroceryBasket) -> Dict[str, Any]:
    """Serialize a complete GroceryBasket to a deterministic dictionary."""
    sorted_items = [
        serialize_grocery_item(basket.items[k])
        for k in sorted(basket.items.keys())
    ]
    return {
        "basket_version": basket.basket_version,
        "total_items": len(basket.items),
        "items": sorted_items,
    }


def deterministic_json(data: Any) -> str:
    """Format an arbitrary data structure as deterministic JSON with stable keys."""
    return json.dumps(data, indent=2, sort_keys=True)


@dataclass(frozen=True)
class ContextRequest:
    """Standardized input payload for context construction across all conditions.

    Contains the authoritative state components required to build context for an agent.
    If basket is not supplied, it is automatically computed deterministically from the plan.
    """

    user_request: str
    plan: MealPlan
    catalog: Dict[str, Recipe]
    ingredients: Dict[str, Ingredient]
    basket: Optional[GroceryBasket] = None
    target_day: Optional[str] = None
    target_meal_type: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.user_request.strip():
            raise ValueError("user_request cannot be empty.")
        if self.basket is None:
            # Deterministically derive the basket if not explicitly provided
            derived = calculate_grocery_basket(
                plan=self.plan,
                catalog=self.catalog,
                ingredients=self.ingredients,
            )
            object.__setattr__(self, "basket", derived)


@dataclass(frozen=True)
class ContextResult:
    """Standardized output payload from context builders.

    Attributes:
        condition: Identifier name of the experimental condition (A, B, or C).
        condition_id: Short condition code ("A", "B", "C").
        system_prompt: System-level instructions for the agent.
        task_prompt: Natural-language request text.
        serialized_context: Deterministic context string passed to the agent.
        signatures: Optional tool definitions or object signatures.
        metadata: Dictionary of metrics (e.g. character count, slot count, disclosure count).
    """

    condition: str
    condition_id: str
    system_prompt: str
    task_prompt: str
    serialized_context: str
    signatures: Optional[Union[str, List[Dict[str, Any]]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a serializable dictionary."""
        return {
            "condition": self.condition,
            "condition_id": self.condition_id,
            "system_prompt": self.system_prompt,
            "task_prompt": self.task_prompt,
            "signatures": self.signatures,
            "serialized_context": self.serialized_context,
            "metadata": copy.deepcopy(self.metadata),
        }

    def to_json(self) -> str:
        """Convert the result to a deterministic JSON string."""
        return deterministic_json(self.to_dict())


class BaseContextBuilder(ABC):
    """Abstract base class for experimental context builders."""

    @abstractmethod
    def build(self, request: ContextRequest) -> ContextResult:
        """Construct the experimental context result for the given request."""
        pass
