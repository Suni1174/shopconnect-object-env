"""Condition C: Object-Centric Progressive Disclosure Context Builder.

This builder represents the proposed research paradigm where the application state
is treated as persistent, authoritative Python domain objects.

Instead of receiving monolithic JSON plan dumps (Condition A) or flat function calling
schemas with bulky state snapshots (Condition B), the agent initially receives only:
1. Compact domain class and method signatures.
2. Short method docstrings and parameter contracts.
3. User preferences and weekly constraints.
4. Plan version and target slot summary (if known).
5. Instructions for progressive, on-demand inspection of sub-objects.

Sub-objects (individual slots, recipes, recipe ingredients, canonical ingredients) are
retrieved selectively via safe, read-only inspection methods, and all disclosures
are tracked in an audit log for token and information disclosure measurement.
"""

import copy
from typing import Any, Dict, List, Optional

from shopconnect_env.context.base import (
    BaseContextBuilder,
    ConditionType,
    ContextRequest,
    ContextResult,
    deterministic_json,
    serialize_decimal,
    serialize_ingredient,
    serialize_ingredient_requirement,
    serialize_meal,
    serialize_user_preferences,
)
from shopconnect_env.domain.models import (
    DAYS_OF_WEEK,
    MEAL_TYPES,
    GroceryBasket,
    Ingredient,
    MealPlan,
    Recipe,
    make_slot_key,
)


# Compact domain class and interface specifications exposed to the agent
COMPACT_CLASS_SIGNATURES: str = """
# =============================================================================
# 1. AUTHORITATIVE DOMAIN OBJECTS (READ-ONLY ACCESSORS)
# =============================================================================

class MealPlan:
    \"\"\"Authoritative weekly meal plan containing 21 schedule slots across 7 days.

    NOTE: MealPlan does NOT provide direct mutation methods (e.g. replace_meal).
    State changes cannot be executed directly on domain objects; they must be
    proposed as allowlisted actions and executed via the Stage 3 ActionBroker.
    \"\"\"
    plan_id: str
    plan_version: int
    preferences: UserPreferences
    meals: Dict[str, Meal]

    def get_meal(self, day: str, meal_type: str) -> Optional[Meal]:
        \"\"\"Retrieve scheduled meal for a slot, or None if unfilled.\"\"\"

    def calculate_total_cost(self, catalog: Dict[str, Recipe]) -> Decimal:
        \"\"\"Calculate total ingredient cost across all scheduled meals.\"\"\"

    def is_complete(self) -> bool:
        \"\"\"Check if all 21 weekly slots have an assigned meal.\"\"\"

class Recipe:
    \"\"\"Culinary blueprint specifying standard portions, tags, and costs.\"\"\"
    recipe_id: str
    name: str
    tags: List[str]
    default_servings: int
    estimated_cost: Decimal
    cost_per_serving: Decimal
    requirements: List[IngredientRequirement]

class Ingredient:
    \"\"\"Canonical ingredient entity with units and monitored allergen tags.\"\"\"
    ingredient_id: str
    name: str
    category: str
    standard_unit: str
    allergens: Optional[Tuple[str, ...]]  # None = unknown; () = checked allergen-free

class UserPreferences:
    \"\"\"Household profile specifying dietary constraints, allergies, and budget.\"\"\"
    diet_tags: List[str]
    allergies: List[str]
    weekly_budget: Decimal
    household_size: int


# =============================================================================
# 2. STATE MUTATIONS (PROPOSED ACTIONS EXECUTED VIA ACTIONBROKER)
# =============================================================================

# All state modifications must be submitted as typed ActionProposal objects to
# the Stage 3 ActionBroker (ActionBroker.execute(proposal)).
# The broker validates schemas, stages changes on isolated candidate copies,
# verifies all policy invariants (budget, allergens, diet), recomputes the
# derived grocery basket, and commits atomically.

Supported Operations:

  Operation: "replace_meal"
    Arguments:
      day: str             # 'Monday'..'Sunday'
      meal_type: str       # 'breakfast', 'lunch', or 'dinner'
      recipe_id: str       # Canonical ID from catalog (e.g. 'rec_berry_oatmeal')
      servings: int | None # Target portions (defaults to recipe default_servings)
    Execution:
      Stages candidate via apply_replace_meal(), runs policy checks, and commits atomically.

  Operation: "change_servings"
    Arguments:
      day: str             # 'Monday'..'Sunday'
      meal_type: str       # 'breakfast', 'lunch', or 'dinner'
      servings: int        # Strictly positive integer (> 0)
    Execution:
      Stages candidate via apply_change_servings(), runs policy checks, and commits atomically.


# =============================================================================
# 3. ON-DEMAND PROGRESSIVE INSPECTION API (CONDITION C)
# =============================================================================

# Safe, read-only methods for querying application sub-objects without monolithic dumps:

inspect_slot(day: str, meal_type: str) -> Dict[str, Any]
  Query the assigned recipe and portion size for a specific schedule slot.

inspect_recipe(recipe_id: str) -> Dict[str, Any]
  Query recipe metadata, tags, standard servings, and estimated costs.

inspect_recipe_ingredients(recipe_id: str) -> Dict[str, Any]
  Query required ingredients, exact quantities, and measurement units for a recipe.

inspect_ingredient(ingredient_id: str) -> Dict[str, Any]
  Query canonical ingredient classification, standard unit, and allergen tags.

inspect_plan() -> Dict[str, Any]
  Query active plan metadata, plan_version, and total assigned meal count.

inspect_preferences() -> Dict[str, Any]
  Query household dietary constraints, allergy restrictions, and weekly budget.

inspect(target_type: str, target_id: str | None = None, day: str | None = None, meal_type: str | None = None) -> Dict[str, Any]
  Unified inspection query dispatcher matching InspectObjectArgs schema.
""".strip()


class ObjectContextBuilder(BaseContextBuilder):
    """Builder for Condition C (Object-Centric Progressive Disclosure).

    Provides compact class/method signatures and manages read-only, on-demand
    selective disclosure with full disclosure auditing.
    """

    SYSTEM_PROMPT: str = (
        "You are an AI meal planning assistant operating under Condition C (Object-Centric Progressive Disclosure).\n"
        "Authoritative application state is maintained in persistent Python domain objects.\n"
        "You are provided with compact class signatures, user preferences, and a target slot summary.\n\n"
        "Guidelines:\n"
        "1. Do NOT expect monolithic state dumps or entire recipe databases upfront.\n"
        "2. If you need details on a specific slot, recipe, or ingredient, use the read-only inspection methods:\n"
        "   - inspect_slot(day, meal_type): Query meal assigned to a specific schedule slot.\n"
        "   - inspect_recipe(recipe_id): Query recipe metadata, tags, servings, and costs.\n"
        "   - inspect_recipe_ingredients(recipe_id): Query ingredient requirements for a recipe.\n"
        "   - inspect_ingredient(ingredient_id): Query canonical ingredient details and allergen classifications.\n"
        "   - inspect_plan(): Query active plan ID, version, and meal counts.\n"
        "   - inspect_preferences(): Query dietary restrictions, household size, and budget.\n"
        "3. State mutations are NOT executed as direct methods on MealPlan.\n"
        "   Propose typed ActionProposals executed atomically through the Stage 3 ActionBroker."
    )

    def __init__(
        self,
        plan: Optional[MealPlan] = None,
        catalog: Optional[Dict[str, Recipe]] = None,
        ingredients: Optional[Dict[str, Ingredient]] = None,
        basket: Optional[GroceryBasket] = None,
    ) -> None:
        """Initialize the object context builder with optional pre-bound state."""
        self._plan = copy.deepcopy(plan) if plan is not None else None
        self._catalog = copy.deepcopy(catalog) if catalog is not None else None
        self._ingredients = copy.deepcopy(ingredients) if ingredients is not None else None
        self._basket = copy.deepcopy(basket) if basket is not None else None
        self._target_day: Optional[str] = None
        self._target_meal_type: Optional[str] = None
        self._disclosures: List[Dict[str, Any]] = []

    def bind_request(self, request: ContextRequest) -> None:
        """Bind state from a ContextRequest safely without reference aliasing."""
        self._plan = copy.deepcopy(request.plan)
        self._catalog = copy.deepcopy(request.catalog)
        self._ingredients = copy.deepcopy(request.ingredients)
        self._basket = copy.deepcopy(request.basket) if request.basket is not None else None
        self._target_day = request.target_day
        self._target_meal_type = request.target_meal_type

    def build(self, request: ContextRequest) -> ContextResult:
        """Construct the compact initial context for Condition C."""
        self.bind_request(request)

        # 1. Target slot summary (if target coordinates are supplied)
        target_slot_summary: Optional[Dict[str, Any]] = None
        if self._target_day and self._target_meal_type and self._plan:
            norm_day = self._target_day.strip().capitalize()
            norm_type = self._target_meal_type.strip().lower()
            slot_key = make_slot_key(norm_day, norm_type)
            if slot_key in self._plan.meals:
                meal = self._plan.meals[slot_key]
                recipe = self._catalog.get(meal.recipe_id) if self._catalog else None
                target_slot_summary = {
                    "slot_key": slot_key,
                    "day": norm_day,
                    "meal_type": norm_type,
                    "status": "assigned",
                    "recipe_id": meal.recipe_id,
                    "recipe_name": recipe.name if recipe else meal.recipe_id,
                    "servings": meal.servings,
                    "estimated_cost": (
                        serialize_decimal(meal.calculate_cost(recipe))
                        if recipe
                        else "0.00"
                    ),
                }
            else:
                target_slot_summary = {
                    "slot_key": slot_key,
                    "day": norm_day,
                    "meal_type": norm_type,
                    "status": "unfilled",
                    "recipe_id": None,
                    "recipe_name": None,
                    "servings": 0,
                    "estimated_cost": "0.00",
                }

        # 2. Compact initial context payload
        initial_context: Dict[str, Any] = {
            "plan_id": request.plan.plan_id,
            "plan_version": request.plan.plan_version,
            "preferences": serialize_user_preferences(request.plan.preferences),
            "target_slot": target_slot_summary,
            "available_inspection_operations": {
                "inspect_slot": "inspect_slot(day, meal_type) -> View assigned meal details for a slot.",
                "inspect_recipe": "inspect_recipe(recipe_id) -> View recipe tags, servings, and costs.",
                "inspect_recipe_ingredients": "inspect_recipe_ingredients(recipe_id) -> View required ingredients.",
                "inspect_ingredient": "inspect_ingredient(ingredient_id) -> View canonical units and allergens.",
                "inspect_plan": "inspect_plan() -> View plan ID, version, and assigned meal count.",
                "inspect_preferences": "inspect_preferences() -> View household dietary constraints and budget.",
            },
        }

        serialized_context = deterministic_json(initial_context)

        # 3. Evaluation metadata
        metadata: Dict[str, Any] = {
            "condition": ConditionType.CONDITION_C.value,
            "condition_name": "Condition C: Object-Centric Progressive Disclosure",
            "initial_context_character_count": len(serialized_context),
            "initial_context_line_count": len(serialized_context.splitlines()),
            "disclosure_count": len(self._disclosures),
            "target_slot_specified": target_slot_summary is not None,
            "disclosures": copy.deepcopy(self._disclosures),
        }

        return ContextResult(
            condition="Condition C: Object-Centric Progressive Disclosure",
            condition_id=ConditionType.CONDITION_C.value,
            system_prompt=self.SYSTEM_PROMPT,
            task_prompt=request.user_request,
            serialized_context=serialized_context,
            signatures=COMPACT_CLASS_SIGNATURES,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    # Progressive Read-Only Inspection Methods
    # -------------------------------------------------------------------------

    def inspect_slot(self, day: str, meal_type: str) -> Dict[str, Any]:
        """Inspect the meal scheduled for a specific day and meal slot.

        Args:
            day: Day of week (e.g. 'Wednesday').
            meal_type: Slot type ('breakfast', 'lunch', 'dinner').

        Returns:
            Dictionary containing slot details or error information.
        """
        if self._plan is None or self._catalog is None:
            return {"status": "error", "error": "uninitialized", "message": "Builder has no bound state."}

        norm_day = day.strip().capitalize()
        norm_type = meal_type.strip().lower()

        if norm_day not in DAYS_OF_WEEK:
            return {
                "status": "error",
                "error": "invalid_day",
                "message": f"Invalid day: '{day}'. Must be one of {DAYS_OF_WEEK}.",
            }
        if norm_type not in MEAL_TYPES:
            return {
                "status": "error",
                "error": "invalid_meal_type",
                "message": f"Invalid meal_type: '{meal_type}'. Must be one of {MEAL_TYPES}.",
            }

        slot_key = make_slot_key(norm_day, norm_type)
        if slot_key in self._plan.meals:
            meal = self._plan.meals[slot_key]
            recipe = self._catalog.get(meal.recipe_id)
            result = {
                "status": "assigned",
                "slot_key": slot_key,
                "day": norm_day,
                "meal_type": norm_type,
                "recipe_id": meal.recipe_id,
                "recipe_name": recipe.name if recipe else meal.recipe_id,
                "servings": meal.servings,
                "estimated_cost": serialize_decimal(meal.calculate_cost(recipe)) if recipe else "0.00",
            }
        else:
            result = {
                "status": "unfilled",
                "slot_key": slot_key,
                "day": norm_day,
                "meal_type": norm_type,
                "recipe_id": None,
                "recipe_name": None,
                "servings": 0,
                "estimated_cost": "0.00",
            }

        self._record_disclosure(
            object_type="slot",
            object_id=slot_key,
            status=result["status"],
            summary=f"Inspected slot {slot_key} ({result['status']})",
            data=result,
        )
        return copy.deepcopy(result)

    def inspect_recipe(self, recipe_id: str) -> Dict[str, Any]:
        """Inspect high-level metadata, tags, and costs for a specific recipe.

        Args:
            recipe_id: Recipe identifier (e.g. 'rec_berry_oatmeal').

        Returns:
            Dictionary with recipe properties or not-found error.
        """
        if self._catalog is None:
            return {"status": "error", "error": "uninitialized", "message": "Builder has no bound catalog."}

        clean_id = recipe_id.strip()
        if clean_id not in self._catalog:
            result = {
                "status": "error",
                "error": "not_found",
                "message": f"Recipe '{clean_id}' not found in catalog.",
            }
            return result

        recipe = self._catalog[clean_id]
        result = {
            "status": "found",
            "recipe_id": recipe.recipe_id,
            "name": recipe.name,
            "tags": sorted(list(recipe.tags)),
            "default_servings": recipe.default_servings,
            "estimated_cost": serialize_decimal(recipe.estimated_cost),
            "cost_per_serving": serialize_decimal(recipe.cost_per_serving),
            "ingredient_count": len(recipe.requirements),
        }

        self._record_disclosure(
            object_type="recipe",
            object_id=clean_id,
            status="found",
            summary=f"Inspected recipe '{recipe.name}' ({clean_id})",
            data=result,
        )
        return copy.deepcopy(result)

    def inspect_recipe_ingredients(self, recipe_id: str) -> Dict[str, Any]:
        """Inspect the detailed ingredient requirements for a specific recipe.

        Args:
            recipe_id: Recipe identifier (e.g. 'rec_red_lentil_dahl').

        Returns:
            Dictionary containing list of required ingredients and quantities.
        """
        if self._catalog is None or self._ingredients is None:
            return {"status": "error", "error": "uninitialized", "message": "Builder has no bound catalog/ingredients."}

        clean_id = recipe_id.strip()
        if clean_id not in self._catalog:
            return {
                "status": "error",
                "error": "not_found",
                "message": f"Recipe '{clean_id}' not found in catalog.",
            }

        recipe = self._catalog[clean_id]
        sorted_reqs = sorted(recipe.requirements, key=lambda r: r.ingredient_id)
        reqs = [
            serialize_ingredient_requirement(
                req,
                self._ingredients.get(req.ingredient_id),
            )
            for req in sorted_reqs
        ]
        result = {
            "status": "found",
            "recipe_id": recipe.recipe_id,
            "recipe_name": recipe.name,
            "ingredient_requirements": reqs,
        }

        self._record_disclosure(
            object_type="recipe_ingredients",
            object_id=clean_id,
            status="found",
            summary=f"Inspected ingredients for '{recipe.name}' ({len(reqs)} items)",
            data=result,
        )
        return copy.deepcopy(result)

    def inspect_ingredient(self, ingredient_id: str) -> Dict[str, Any]:
        """Inspect canonical specifications and allergen tags for an ingredient.

        Args:
            ingredient_id: Ingredient identifier (e.g. 'ing_rolled_oats').

        Returns:
            Dictionary with canonical ingredient details or not-found error.
        """
        if self._ingredients is None:
            return {"status": "error", "error": "uninitialized", "message": "Builder has no bound ingredients."}

        clean_id = ingredient_id.strip()
        if clean_id not in self._ingredients:
            return {
                "status": "error",
                "error": "not_found",
                "message": f"Ingredient '{clean_id}' not found in catalog.",
            }

        ing = self._ingredients[clean_id]
        result = {
            "status": "found",
            **serialize_ingredient(ing),
        }

        self._record_disclosure(
            object_type="ingredient",
            object_id=clean_id,
            status="found",
            summary=f"Inspected ingredient '{ing.name}' ({clean_id})",
            data=result,
        )
        return copy.deepcopy(result)

    def inspect_plan(self) -> Dict[str, Any]:
        """Inspect high-level metadata for the active MealPlan.

        Returns:
            Dictionary with plan_id, plan_version, and total assigned meals count.
        """
        if self._plan is None:
            return {"status": "error", "error": "uninitialized", "message": "No bound plan."}
        res = {
            "status": "found",
            "plan_id": self._plan.plan_id,
            "plan_version": self._plan.plan_version,
            "total_assigned_meals": len(self._plan.meals),
        }
        self._record_disclosure("plan", self._plan.plan_id, "found", "Inspected plan summary", res)
        return copy.deepcopy(res)

    def inspect_preferences(self) -> Dict[str, Any]:
        """Inspect household user preferences and dietary constraints.

        Returns:
            Dictionary containing diet_tags, allergies, weekly_budget, and household_size.
        """
        if self._plan is None:
            return {"status": "error", "error": "uninitialized", "message": "No bound preferences."}
        res = {
            "status": "found",
            **serialize_user_preferences(self._plan.preferences),
        }
        self._record_disclosure("preferences", "user_preferences", "found", "Inspected user preferences", res)
        return copy.deepcopy(res)

    def inspect(
        self,
        target_type: str,
        target_id: Optional[str] = None,
        day: Optional[str] = None,
        meal_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Unified inspection dispatcher matching InspectObjectArgs query schema."""
        norm_type = target_type.strip().lower()

        if norm_type in {"slot", "meal"}:
            if day and meal_type:
                return self.inspect_slot(day, meal_type)
            if target_id and ("." in target_id or "_" in target_id):
                sep = "_" if "_" in target_id else "."
                parts = target_id.split(sep, 1)
                return self.inspect_slot(parts[0], parts[1])
            return {
                "status": "error",
                "error": "invalid_parameters",
                "message": "Slot inspection requires 'day' and 'meal_type' or target_id like 'Wednesday_dinner' or 'Wednesday.dinner'.",
            }

        if norm_type == "recipe":
            if not target_id:
                return {"status": "error", "error": "missing_target_id", "message": "Recipe inspection requires target_id."}
            return self.inspect_recipe(target_id)

        if norm_type in {"recipe_ingredients", "recipe_ingredient", "requirements"}:
            if not target_id:
                return {"status": "error", "error": "missing_target_id", "message": "Recipe ingredients inspection requires target_id."}
            return self.inspect_recipe_ingredients(target_id)

        if norm_type == "ingredient":
            if not target_id:
                return {"status": "error", "error": "missing_target_id", "message": "Ingredient inspection requires target_id."}
            return self.inspect_ingredient(target_id)

        if norm_type in {"plan", "meal_plan"}:
            return self.inspect_plan()

        if norm_type == "preferences":
            return self.inspect_preferences()

        return {
            "status": "error",
            "error": "unsupported_target_type",
            "message": f"Unsupported target_type: '{target_type}'. Valid targets: slot, recipe, recipe_ingredients, ingredient, plan, preferences.",
        }

    # -------------------------------------------------------------------------
    # Disclosure Tracking Audit Utilities
    # -------------------------------------------------------------------------

    def _record_disclosure(
        self,
        object_type: str,
        object_id: str,
        status: str,
        summary: str,
        data: Dict[str, Any],
    ) -> None:
        """Append an entry to the progressive disclosure audit log."""
        serialized = deterministic_json(data)
        self._disclosures.append({
            "disclosure_id": len(self._disclosures) + 1,
            "object_type": object_type,
            "object_id": object_id,
            "status": status,
            "summary": summary,
            "revealed_fields": sorted(list(data.keys())),
            "character_count": len(serialized),
        })

    def get_disclosure_log(self) -> List[Dict[str, Any]]:
        """Return a safe copy of all disclosures recorded by this builder."""
        return copy.deepcopy(self._disclosures)

    def get_disclosure_count(self) -> int:
        """Return the total number of successful inspection calls recorded."""
        return len(self._disclosures)

    def get_cumulative_revealed_characters(self) -> int:
        """Return the sum of character counts across all disclosed inspection payloads."""
        return sum(d.get("character_count", 0) for d in self._disclosures)

    def get_total_context_characters(self, initial_result: ContextResult) -> int:
        """Return combined character count (initial compact context + all selective disclosures).

        NOTE: Character count reflects payload string length and is NOT identical to model
        token counts. Actual token consumption must be measured with model tokenizers in Stage 6.
        """
        return initial_result.metadata.get("initial_context_character_count", 0) + self.get_cumulative_revealed_characters()

    def reset_disclosure_log(self) -> None:
        """Clear the disclosure audit log for a new task session."""
        self._disclosures.clear()
