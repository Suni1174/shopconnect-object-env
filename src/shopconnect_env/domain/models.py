"""Domain models for the ShopConnect Object-Centric Intelligence Environment.

These models represent the authoritative state of the meal planning domain.
They are pure Python dataclasses and have zero dependence on any LLM or external framework.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Canonical constants for the 7-day, 3-meals-per-day schedule
DAYS_OF_WEEK: Tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

MEAL_TYPES: Tuple[str, ...] = (
    "breakfast",
    "lunch",
    "dinner",
)


def make_slot_key(day: str, meal_type: str) -> str:
    """Create a standardized unique identifier for a meal slot.
    
    Example:
        make_slot_key("Wednesday", "dinner") -> "Wednesday_dinner"
    """
    normalized_day = day.strip().capitalize()
    normalized_slot = meal_type.strip().lower()

    if normalized_day not in DAYS_OF_WEEK:
        raise ValueError(
            f"Invalid day: '{day}'. Must be one of {DAYS_OF_WEEK}."
        )
    if normalized_slot not in MEAL_TYPES:
        raise ValueError(
            f"Invalid meal_type: '{meal_type}'. Must be one of {MEAL_TYPES}."
        )

    return f"{normalized_day}_{normalized_slot}"


@dataclass(frozen=True)
class Ingredient:
    """A canonical ingredient stored in the system catalog.
    
    Attributes:
        ingredient_id: Unique string identifier (e.g. 'ing_rolled_oats').
        name: Human-readable display name (e.g. 'Rolled Oats').
        category: Broad category for organization (e.g. 'Grains', 'Produce', 'Dairy').
        standard_unit: Canonical measurement unit (e.g. 'g', 'ml', 'unit').
        allergens: Optional tuple of canonical allergen classifications (e.g. ('dairy', 'milk')).
                   None indicates unknown allergen metadata (treated conservatively).
                   An empty tuple () indicates explicitly verified allergen-free
                   among standard monitored classifications.
    """

    ingredient_id: str
    name: str
    category: str
    standard_unit: str
    allergens: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if not self.ingredient_id.strip():
            raise ValueError("ingredient_id cannot be empty.")
        if not self.name.strip():
            raise ValueError("name cannot be empty.")
        if not self.standard_unit.strip():
            raise ValueError("standard_unit cannot be empty.")
        if self.allergens is not None:
            # Normalize to an immutable tuple of stripped lowercase strings
            normalized = tuple(a.strip().lower() for a in self.allergens)
            object.__setattr__(self, "allergens", normalized)


@dataclass(frozen=True)
class IngredientRequirement:
    """The quantity and unit of an ingredient required by a recipe.
    
    Attributes:
        ingredient_id: Reference to canonical Ingredient.ingredient_id.
        quantity: Amount needed (using Decimal for exact math, e.g. Decimal('150.0')).
        unit: Measurement unit (e.g. 'g', 'ml', 'unit').
    """

    ingredient_id: str
    quantity: Decimal
    unit: str

    def __post_init__(self) -> None:
        if not self.ingredient_id.strip():
            raise ValueError("ingredient_id cannot be empty.")
        if self.quantity <= Decimal("0"):
            raise ValueError(f"Requirement quantity must be positive, got {self.quantity}.")
        if not self.unit.strip():
            raise ValueError("unit cannot be empty.")

    def scale(self, factor: Decimal) -> "IngredientRequirement":
        """Scale this ingredient requirement by a multiplier."""
        if factor <= Decimal("0"):
            raise ValueError(f"Scale factor must be positive, got {factor}.")
        return IngredientRequirement(
            ingredient_id=self.ingredient_id,
            quantity=self.quantity * factor,
            unit=self.unit,
        )


@dataclass
class Recipe:
    """A recipe blueprint.
    
    Cost Calculation Basis:
        - estimated_cost represents the baseline cost (in USD) for default_servings.
        - cost_per_serving = estimated_cost / default_servings.
    
    Attributes:
        recipe_id: Unique recipe identifier (e.g. 'rec_lentil_soup').
        name: Recipe title (e.g. 'Comforting Red Lentil Soup').
        tags: Dietary and meal metadata (e.g. ['vegetarian', 'vegan', 'high-protein']).
        default_servings: Number of servings this recipe yields by default (e.g. 4).
        estimated_cost: Total estimated ingredient cost in USD for default_servings.
        requirements: List of IngredientRequirement items needed.
    """

    recipe_id: str
    name: str
    tags: List[str]
    default_servings: int
    estimated_cost: Decimal
    requirements: List[IngredientRequirement] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.recipe_id.strip():
            raise ValueError("recipe_id cannot be empty.")
        if not self.name.strip():
            raise ValueError("name cannot be empty.")
        if self.default_servings <= 0:
            raise ValueError(f"default_servings must be > 0, got {self.default_servings}.")
        if self.estimated_cost < Decimal("0"):
            raise ValueError(f"estimated_cost cannot be negative, got {self.estimated_cost}.")

    @property
    def cost_per_serving(self) -> Decimal:
        """Calculate the cost for a single serving."""
        return self.estimated_cost / Decimal(self.default_servings)


@dataclass
class Meal:
    """An individual meal assigned to a specific schedule slot.
    
    Attributes:
        day: Day of the week ('Monday' through 'Sunday').
        meal_type: Slot type ('breakfast', 'lunch', or 'dinner').
        recipe_id: Reference to Recipe.recipe_id.
        servings: Number of servings to prepare for this specific meal.
    """

    day: str
    meal_type: str
    recipe_id: str
    servings: int

    def __post_init__(self) -> None:
        self.day = self.day.strip().capitalize()
        self.meal_type = self.meal_type.strip().lower()

        if self.day not in DAYS_OF_WEEK:
            raise ValueError(f"Invalid day: '{self.day}'. Must be in {DAYS_OF_WEEK}.")
        if self.meal_type not in MEAL_TYPES:
            raise ValueError(f"Invalid meal_type: '{self.meal_type}'. Must be in {MEAL_TYPES}.")
        if not self.recipe_id.strip():
            raise ValueError("recipe_id cannot be empty.")
        if self.servings <= 0:
            raise ValueError(f"servings must be positive, got {self.servings}.")

    @property
    def slot_key(self) -> str:
        """Return the unique slot identifier for this meal (e.g. 'Monday_breakfast')."""
        return make_slot_key(self.day, self.meal_type)

    def calculate_cost(self, recipe: Recipe) -> Decimal:
        """Calculate total cost of this meal based on assigned recipe and servings."""
        if recipe.recipe_id != self.recipe_id:
            raise ValueError(
                f"Recipe ID mismatch: Meal references '{self.recipe_id}', but recipe is '{recipe.recipe_id}'."
            )
        return recipe.cost_per_serving * Decimal(self.servings)


@dataclass
class UserPreferences:
    """User profile specifying dietary constraints, allergies, and weekly budget.
    
    Attributes:
        diet_tags: Allowed/required diet tags (e.g. ['vegetarian', 'dairy-free']).
        allergies: Ingredient names or tags that must be excluded (e.g. ['peanuts']).
        weekly_budget: Maximum allowed weekly budget in USD (Decimal).
        household_size: Number of people to plan meals for.
    """

    diet_tags: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    weekly_budget: Decimal = Decimal("150.00")
    household_size: int = 2

    def __post_init__(self) -> None:
        if self.weekly_budget <= Decimal("0"):
            raise ValueError(f"weekly_budget must be positive, got {self.weekly_budget}.")
        if self.household_size <= 0:
            raise ValueError(f"household_size must be positive, got {self.household_size}.")


@dataclass
class MealPlan:
    """The authoritative 7-day weekly meal plan.
    
    Contains up to 21 meal slots (7 days x 3 meals per day).
    
    Attributes:
        plan_id: Unique identifier for this meal plan.
        plan_version: Integer version counter incremented on every committed change.
        preferences: Associated UserPreferences.
        meals: Dictionary mapping slot_key (e.g. 'Monday_breakfast') to Meal instance.
    """

    plan_id: str
    plan_version: int = 1
    preferences: UserPreferences = field(default_factory=UserPreferences)
    meals: Dict[str, Meal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plan_id.strip():
            raise ValueError("plan_id cannot be empty.")
        if self.plan_version < 1:
            raise ValueError(f"plan_version must be >= 1, got {self.plan_version}.")

    @classmethod
    def all_expected_slot_keys(cls) -> List[str]:
        """Return all 21 standard slot keys in weekly chronological order."""
        return [
            make_slot_key(day, meal_type)
            for day in DAYS_OF_WEEK
            for meal_type in MEAL_TYPES
        ]

    def set_meal(self, meal: Meal) -> None:
        """Assign a meal to its designated slot."""
        self.meals[meal.slot_key] = meal

    def get_meal(self, day: str, meal_type: str) -> Optional[Meal]:
        """Retrieve the meal for a specific day and slot, or None if unfilled."""
        key = make_slot_key(day, meal_type)
        return self.meals.get(key)

    def is_complete(self) -> bool:
        """Check if all 21 slots have an assigned meal."""
        return len(self.meals) == 21 and all(
            k in self.meals for k in self.all_expected_slot_keys()
        )

    def calculate_total_cost(self, catalog: Dict[str, Recipe]) -> Decimal:
        """Calculate the total cost across all scheduled meals in this plan."""
        total = Decimal("0.00")
        for meal in self.meals.values():
            if meal.recipe_id not in catalog:
                raise KeyError(f"Recipe '{meal.recipe_id}' not found in catalog.")
            recipe = catalog[meal.recipe_id]
            total += meal.calculate_cost(recipe)
        return total


@dataclass(frozen=True)
class GroceryItem:
    """An aggregated item in the grocery basket.
    
    Attributes:
        ingredient_id: Canonical Ingredient ID.
        name: Canonical Ingredient display name.
        category: Ingredient category for grocery aisle organization.
        quantity: Aggregated required quantity (Decimal).
        unit: Canonical measurement unit.
    """

    ingredient_id: str
    name: str
    category: str
    quantity: Decimal
    unit: str


@dataclass(frozen=True)
class GroceryBasket:
    """A purely derived projection of ingredients required for an active MealPlan.
    
    CRITICAL ARCHITECTURAL INVARIANT:
        This object is immutable (frozen) and MUST NOT be directly modified by an LLM.
        It is computed exclusively by inspecting all active recipes in a MealPlan.
    
    Attributes:
        basket_version: Mirrors the plan_version from which it was computed.
        items: Dictionary mapping ingredient_id to aggregated GroceryItem.
    """

    basket_version: int
    items: Dict[str, GroceryItem] = field(default_factory=dict)

    @classmethod
    def from_meal_plan(
        cls,
        plan: MealPlan,
        catalog: Dict[str, Recipe],
        ingredients: Dict[str, Ingredient],
    ) -> "GroceryBasket":
        """Calculate a grocery basket deterministically from a meal plan and catalog.
        
        Args:
            plan: The active MealPlan containing scheduled meals.
            catalog: Dictionary mapping recipe_id to Recipe.
            ingredients: Dictionary mapping ingredient_id to canonical Ingredient.
            
        Returns:
            A new immutable GroceryBasket instance.
        """
        aggregated_quantities: Dict[str, Decimal] = {}
        units_map: Dict[str, str] = {}

        for meal in plan.meals.values():
            if meal.recipe_id not in catalog:
                raise KeyError(
                    f"Cannot recompute basket: recipe '{meal.recipe_id}' missing from catalog."
                )
            recipe = catalog[meal.recipe_id]

            # Scale factor: meal.servings / recipe.default_servings
            scale_factor = Decimal(meal.servings) / Decimal(recipe.default_servings)

            for req in recipe.requirements:
                if req.ingredient_id not in ingredients:
                    raise KeyError(
                        f"Cannot recompute basket: ingredient '{req.ingredient_id}' missing from catalog."
                    )

                scaled_amount = req.quantity * scale_factor

                if req.ingredient_id in aggregated_quantities:
                    # Check unit compatibility
                    existing_unit = units_map[req.ingredient_id]
                    if existing_unit != req.unit:
                        raise ValueError(
                            f"Unit conflict for ingredient '{req.ingredient_id}': "
                            f"'{existing_unit}' vs '{req.unit}'."
                        )
                    aggregated_quantities[req.ingredient_id] += scaled_amount
                else:
                    aggregated_quantities[req.ingredient_id] = scaled_amount
                    units_map[req.ingredient_id] = req.unit

        # Build final GroceryItem map
        basket_items: Dict[str, GroceryItem] = {}
        for ing_id, total_qty in aggregated_quantities.items():
            ing = ingredients[ing_id]
            basket_items[ing_id] = GroceryItem(
                ingredient_id=ing_id,
                name=ing.name,
                category=ing.category,
                quantity=total_qty,
                unit=units_map[ing_id],
            )

        return cls(basket_version=plan.plan_version, items=basket_items)


def load_catalog_from_json(
    filepath: Path | str,
) -> Tuple[Dict[str, Ingredient], Dict[str, Recipe]]:
    """Load and parse canonical ingredients and recipes from a catalog JSON file.
    
    Args:
        filepath: Path to the catalog JSON file.
        
    Returns:
        A tuple of (ingredients_dict, recipes_dict).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Parse Ingredients
    ingredients: Dict[str, Ingredient] = {}
    for ing_data in data.get("ingredients", []):
        raw_allergens = ing_data.get("allergens")
        allergens_val = (
            tuple(raw_allergens) if raw_allergens is not None else None
        )
        ing = Ingredient(
            ingredient_id=ing_data["ingredient_id"],
            name=ing_data["name"],
            category=ing_data["category"],
            standard_unit=ing_data["standard_unit"],
            allergens=allergens_val,
        )
        ingredients[ing.ingredient_id] = ing

    # 2. Parse Recipes
    recipes: Dict[str, Recipe] = {}
    for rec_data in data.get("recipes", []):
        reqs: List[IngredientRequirement] = []
        for req_data in rec_data.get("requirements", []):
            ing_id = req_data["ingredient_id"]
            if ing_id not in ingredients:
                raise ValueError(
                    f"Recipe '{rec_data['recipe_id']}' references unknown ingredient '{ing_id}'."
                )
            req = IngredientRequirement(
                ingredient_id=ing_id,
                quantity=Decimal(str(req_data["quantity"])),
                unit=req_data["unit"],
            )
            reqs.append(req)

        rec = Recipe(
            recipe_id=rec_data["recipe_id"],
            name=rec_data["name"],
            tags=list(rec_data.get("tags", [])),
            default_servings=int(rec_data["default_servings"]),
            estimated_cost=Decimal(str(rec_data["estimated_cost"])),
            requirements=reqs,
        )
        recipes[rec.recipe_id] = rec

    return ingredients, recipes
