"""Condition A: Full-State Regeneration Context Builder.

This builder represents the baseline monolithic interaction pattern where the entire
application state—including the complete 7-day schedule (all 21 slots), full recipe
definitions, ingredient requirements, canonical ingredient records, user preferences,
and current derived grocery basket—is serialized directly into the agent's context.

The agent in Condition A is expected to read the complete state and re-emit the entire
updated meal plan.
"""

from typing import Any, Dict

from shopconnect_env.context.base import (
    BaseContextBuilder,
    ConditionType,
    ContextRequest,
    ContextResult,
    deterministic_json,
    serialize_all_21_slots,
    serialize_decimal,
    serialize_grocery_basket,
    serialize_ingredient,
    serialize_recipe,
    serialize_user_preferences,
)


class FullStateContextBuilder(BaseContextBuilder):
    """Builder for Condition A (Full-State Regeneration).

    Generates a comprehensive, monolithic serialization of the entire meal planning
    domain state without selective disclosure or progressive filtering.
    """

    SYSTEM_PROMPT: str = (
        "You are an AI meal planning assistant operating under Condition A (Full-State Regeneration).\n"
        "You have authoritative visibility over the entire application state, including all 7 days "
        "(21 meal slots), complete recipe definitions, ingredient requirements, user preferences, "
        "and the aggregated grocery basket.\n\n"
        "Instructions:\n"
        "1. Analyze the user's natural language request.\n"
        "2. Review the full current schedule and constraint profile.\n"
        "3. You must regenerate and emit the COMPLETE updated meal plan for the entire week.\n"
        "4. Preserve all meals that do not require modification.\n"
        "5. Ensure the revised plan respects all dietary tags, allergen constraints, and budget limits."
    )

    def build(self, request: ContextRequest) -> ContextResult:
        """Construct the monolithic full-state context for Condition A."""
        basket = request.basket
        assert basket is not None, "Basket must be populated in ContextRequest."

        # 1. Serialize all 21 slots deterministically (Monday through Sunday, 3 meals/day)
        schedule_slots = serialize_all_21_slots(request.plan, request.catalog)

        # 2. Serialize full details for recipes currently assigned in the schedule
        assigned_recipe_ids = sorted({
            meal.recipe_id
            for meal in request.plan.meals.values()
            if meal.recipe_id in request.catalog
        })
        recipes_in_plan = {
            r_id: serialize_recipe(request.catalog[r_id], request.ingredients)
            for r_id in assigned_recipe_ids
        }

        # 3. Serialize all available recipes in the catalog
        available_recipes = [
            serialize_recipe(request.catalog[r_id], request.ingredients)
            for r_id in sorted(request.catalog.keys())
        ]

        # 4. Serialize canonical ingredient catalog
        ingredient_catalog = [
            serialize_ingredient(request.ingredients[ing_id])
            for ing_id in sorted(request.ingredients.keys())
        ]

        # 5. Assemble monolithic state payload
        full_state_payload: Dict[str, Any] = {
            "plan_id": request.plan.plan_id,
            "plan_version": request.plan.plan_version,
            "preferences": serialize_user_preferences(request.plan.preferences),
            "schedule": schedule_slots,
            "recipes_in_plan": recipes_in_plan,
            "available_recipes": available_recipes,
            "ingredient_catalog": ingredient_catalog,
            "grocery_basket": serialize_grocery_basket(basket),
        }

        serialized_context = deterministic_json(full_state_payload)

        # 6. Compute metadata for evaluation
        assigned_count = len(request.plan.meals)
        metadata: Dict[str, Any] = {
            "condition": ConditionType.CONDITION_A.value,
            "condition_name": "Condition A: Full-State Regeneration",
            "total_slots": len(schedule_slots),
            "assigned_slots": assigned_count,
            "unfilled_slots": len(schedule_slots) - assigned_count,
            "total_recipes_in_catalog": len(request.catalog),
            "total_ingredients_in_catalog": len(request.ingredients),
            "recipes_in_plan_count": len(recipes_in_plan),
            "basket_total_items": len(basket.items),
            "context_character_count": len(serialized_context),
            "context_line_count": len(serialized_context.splitlines()),
            "disclosure_count": 0,
        }

        return ContextResult(
            condition="Condition A: Full-State Regeneration",
            condition_id=ConditionType.CONDITION_A.value,
            system_prompt=self.SYSTEM_PROMPT,
            task_prompt=request.user_request,
            serialized_context=serialized_context,
            signatures=None,  # Condition A does not supply tool/method signatures
            metadata=metadata,
        )
