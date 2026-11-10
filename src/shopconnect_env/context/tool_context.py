"""Condition B: Conventional Flat Tool-Calling Context Builder.

This builder represents the standard industry paradigm (e.g. OpenAI / Anthropic function calling),
where an AI agent receives flat tool/function specifications alongside a serialized
application state snapshot.

The agent in Condition B interacts with the system by invoking one of the declared tools
(e.g., `replace_meal` or `change_servings`) rather than regenerating the complete document.
"""

from typing import Any, Dict, List

from shopconnect_env.context.base import (
    BaseContextBuilder,
    ConditionType,
    ContextRequest,
    ContextResult,
    deterministic_json,
    serialize_decimal,
    serialize_meal,
    serialize_user_preferences,
)
from shopconnect_env.domain.models import DAYS_OF_WEEK, MEAL_TYPES
from shopconnect_env.validation.schemas import ChangeServingsArgs, ReplaceMealArgs


def get_conventional_tool_definitions() -> List[Dict[str, Any]]:
    """Return flat function-calling specifications derived directly from Pydantic schemas.

    Uses Stage 3 Pydantic schemas as the single source of truth for argument constraints,
    descriptions, and required parameters.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "replace_meal",
                "description": (
                    "Replace the scheduled recipe in a specific day and meal slot "
                    "with an alternative recipe from the catalog."
                ),
                "parameters": ReplaceMealArgs.model_json_schema(),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "change_servings",
                "description": (
                    "Change the number of servings for an existing scheduled meal "
                    "in a specific day and meal slot."
                ),
                "parameters": ChangeServingsArgs.model_json_schema(),
            },
        },
    ]


class ToolContextBuilder(BaseContextBuilder):
    """Builder for Condition B (Conventional Flat Tool-Calling Agent).

    Supplies flat function definitions and a structured application state snapshot.
    Does not expose progressive object inspection or monolithic raw ingredient tables.
    """

    SYSTEM_PROMPT: str = (
        "You are an AI meal planning assistant operating under Condition B (Conventional Tool Agent).\n"
        "You have access to defined tools to perform operations on the weekly meal plan.\n\n"
        "Available Tools:\n"
        "1. replace_meal(day, meal_type, recipe_id, servings=None)\n"
        "2. change_servings(day, meal_type, servings)\n\n"
        "Instructions:\n"
        "1. Inspect the provided state snapshot (current schedule, available recipes, user preferences).\n"
        "2. Select the single appropriate tool to accomplish the user's request.\n"
        "3. Emit a structured tool call matching the tool schema.\n"
        "4. Do NOT attempt to rewrite the entire meal plan."
    )

    def build(self, request: ContextRequest) -> ContextResult:
        """Construct the flat tool-calling context for Condition B."""
        basket = request.basket
        assert basket is not None, "Basket must be populated in ContextRequest."

        # 1. Tool definitions derived from Stage 3 Pydantic schemas
        tools = get_conventional_tool_definitions()

        # 2. Schedule snapshot: currently assigned meals in deterministic calendar order
        def slot_sort_key(m):
            day_idx = DAYS_OF_WEEK.index(m.day) if m.day in DAYS_OF_WEEK else 99
            type_idx = MEAL_TYPES.index(m.meal_type) if m.meal_type in MEAL_TYPES else 99
            return (day_idx, type_idx)

        sorted_meals = sorted(request.plan.meals.values(), key=slot_sort_key)
        current_schedule = [
            serialize_meal(meal, request.catalog)
            for meal in sorted_meals
        ]

        # 3. Available recipes summary: high-level catalog for selection without raw ingredient dumps
        available_recipes = [
            {
                "recipe_id": r.recipe_id,
                "name": r.name,
                "tags": sorted(list(r.tags)),
                "default_servings": r.default_servings,
                "estimated_cost": serialize_decimal(r.estimated_cost),
                "cost_per_serving": serialize_decimal(r.cost_per_serving),
            }
            for r in sorted(request.catalog.values(), key=lambda x: x.recipe_id)
        ]

        # 4. State snapshot payload
        state_snapshot: Dict[str, Any] = {
            "plan_id": request.plan.plan_id,
            "plan_version": request.plan.plan_version,
            "preferences": serialize_user_preferences(request.plan.preferences),
            "current_schedule": current_schedule,
            "available_recipes": available_recipes,
            "basket_summary": {
                "basket_version": basket.basket_version,
                "total_items": len(basket.items),
            },
        }

        serialized_context = deterministic_json(state_snapshot)

        # 5. Metadata for comparative evaluation
        metadata: Dict[str, Any] = {
            "condition": ConditionType.CONDITION_B.value,
            "condition_name": "Condition B: Conventional Tool Agent",
            "tool_count": len(tools),
            "tool_names": [t["function"]["name"] for t in tools],
            "scheduled_meals_count": len(current_schedule),
            "available_recipes_count": len(available_recipes),
            "basket_total_items": len(basket.items),
            "context_character_count": len(serialized_context),
            "context_line_count": len(serialized_context.splitlines()),
            "disclosure_count": 0,
        }

        return ContextResult(
            condition="Condition B: Conventional Tool Agent",
            condition_id=ConditionType.CONDITION_B.value,
            system_prompt=self.SYSTEM_PROMPT,
            task_prompt=request.user_request,
            serialized_context=serialized_context,
            signatures=tools,
            metadata=metadata,
        )
