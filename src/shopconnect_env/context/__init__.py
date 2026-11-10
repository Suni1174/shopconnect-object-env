"""Context construction strategies for comparative evaluation.

Exports builders for:
- Condition A: FullStateContextBuilder (Monolithic full-plan regeneration baseline)
- Condition B: ToolContextBuilder (Conventional flat tool-calling baseline)
- Condition C: ObjectContextBuilder (Persistent domain objects with progressive disclosure)

Also exports shared ContextRequest and ContextResult data models and deterministic
serialization utilities.
"""

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
    serialize_ingredient_requirement,
    serialize_meal,
    serialize_recipe,
    serialize_unfilled_slot,
    serialize_user_preferences,
)
from shopconnect_env.context.full_state import FullStateContextBuilder
from shopconnect_env.context.object_context import ObjectContextBuilder
from shopconnect_env.context.tool_context import ToolContextBuilder

__all__ = [
    "BaseContextBuilder",
    "ConditionType",
    "ContextRequest",
    "ContextResult",
    "FullStateContextBuilder",
    "ToolContextBuilder",
    "ObjectContextBuilder",
    "deterministic_json",
    "serialize_decimal",
    "serialize_ingredient",
    "serialize_ingredient_requirement",
    "serialize_recipe",
    "serialize_meal",
    "serialize_unfilled_slot",
    "serialize_all_21_slots",
    "serialize_user_preferences",
    "serialize_grocery_basket",
]
