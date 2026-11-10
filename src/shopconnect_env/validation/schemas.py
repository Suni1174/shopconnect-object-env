"""Pydantic action schemas for the ShopConnect Object-Centric Intelligence Environment.

Enforces strict syntax and type boundaries between untrusted LLM action proposals
and the trusted Python domain runtime.
"""

from __future__ import annotations

from enum import Enum
import json
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shopconnect_env.domain.models import DAYS_OF_WEEK, MEAL_TYPES


class OperationType(str, Enum):
    """Allowlisted operation identifiers."""

    REPLACE_MEAL = "replace_meal"
    CHANGE_SERVINGS = "change_servings"
    INSPECT_OBJECT = "inspect_object"


class ActionStatus(str, Enum):
    """Transaction outcome status."""

    COMMITTED = "COMMITTED"
    REJECTED = "REJECTED"


class ReplaceMealArgs(BaseModel):
    """Arguments for the replace_meal operation."""

    model_config = ConfigDict(extra="forbid")

    day: str = Field(..., description="Day of the week (e.g. 'Wednesday')")
    meal_type: str = Field(..., description="Meal slot ('breakfast', 'lunch', or 'dinner')")
    recipe_id: str = Field(..., min_length=1, description="Unique recipe identifier from catalog")
    servings: Optional[int] = Field(default=None, description="Optional positive integer serving count")

    @field_validator("day")
    @classmethod
    def validate_day(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Day must be a non-empty string.")
        norm = v.strip().capitalize()
        if norm not in DAYS_OF_WEEK:
            raise ValueError(f"Invalid day: '{v}'. Must be one of {DAYS_OF_WEEK}.")
        return norm

    @field_validator("meal_type")
    @classmethod
    def validate_meal_type(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Meal type must be a non-empty string.")
        norm = v.strip().lower()
        if norm not in MEAL_TYPES:
            raise ValueError(f"Invalid meal_type: '{v}'. Must be one of {MEAL_TYPES}.")
        return norm

    @field_validator("recipe_id")
    @classmethod
    def validate_recipe_id(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("recipe_id cannot be empty.")
        return v.strip()

    @field_validator("servings", mode="before")
    @classmethod
    def validate_servings(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("servings cannot be a boolean value.")
        if isinstance(v, float):
            raise ValueError("servings must be an integer, not a float.")
        if not isinstance(v, int):
            raise ValueError(f"servings must be an integer, got {type(v).__name__}.")
        if v <= 0:
            raise ValueError(f"servings must be a positive integer (> 0), got {v}.")
        return v


class ChangeServingsArgs(BaseModel):
    """Arguments for the change_servings operation."""

    model_config = ConfigDict(extra="forbid")

    day: str = Field(..., description="Day of the week (e.g. 'Tuesday')")
    meal_type: str = Field(..., description="Meal slot ('breakfast', 'lunch', or 'dinner')")
    servings: int = Field(..., description="Positive integer serving count")

    @field_validator("day")
    @classmethod
    def validate_day(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Day must be a non-empty string.")
        norm = v.strip().capitalize()
        if norm not in DAYS_OF_WEEK:
            raise ValueError(f"Invalid day: '{v}'. Must be one of {DAYS_OF_WEEK}.")
        return norm

    @field_validator("meal_type")
    @classmethod
    def validate_meal_type(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Meal type must be a non-empty string.")
        norm = v.strip().lower()
        if norm not in MEAL_TYPES:
            raise ValueError(f"Invalid meal_type: '{v}'. Must be one of {MEAL_TYPES}.")
        return norm

    @field_validator("servings", mode="before")
    @classmethod
    def validate_servings(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("servings cannot be a boolean value.")
        if isinstance(v, float):
            raise ValueError("servings must be an integer, not a float.")
        if not isinstance(v, int):
            raise ValueError(f"servings must be an integer, got {type(v).__name__}.")
        if v <= 0:
            raise ValueError(f"servings must be a positive integer (> 0), got {v}.")
        return v


class InspectObjectArgs(BaseModel):
    """Arguments for the inspect_object query."""

    model_config = ConfigDict(extra="forbid")

    target_type: str = Field(..., min_length=1, description="Object category ('recipe', 'meal', 'plan', 'preferences', 'ingredient')")
    target_id: Optional[str] = Field(default=None, description="Optional entity identifier")

    @field_validator("target_type")
    @classmethod
    def validate_target_type(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("target_type must be a non-empty string.")
        norm = v.strip().lower()
        valid_targets = {"recipe", "meal", "plan", "preferences", "ingredient"}
        if norm not in valid_targets:
            raise ValueError(f"Invalid target_type: '{v}'. Must be one of {valid_targets}.")
        return norm


class ActionProposal(BaseModel):
    """Untrusted action proposal emitted by an agent."""

    model_config = ConfigDict(extra="forbid")

    operation: OperationType = Field(..., description="Allowlisted operation type")
    arguments: Dict[str, Any] = Field(..., description="Operation-specific arguments dictionary")
    reason: str = Field(..., min_length=1, description="Human-readable justification for the proposal")
    expected_plan_version: Optional[int] = Field(default=None, ge=1, description="Optional expected plan version for stale-state protection")

    @model_validator(mode="after")
    def validate_arguments_match_operation(self) -> "ActionProposal":
        """Verify that arguments match the operation-specific schema."""
        if self.operation == OperationType.REPLACE_MEAL:
            ReplaceMealArgs.model_validate(self.arguments)
        elif self.operation == OperationType.CHANGE_SERVINGS:
            ChangeServingsArgs.model_validate(self.arguments)
        elif self.operation == OperationType.INSPECT_OBJECT:
            InspectObjectArgs.model_validate(self.arguments)
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")
        return self

    def get_typed_arguments(self) -> Union[ReplaceMealArgs, ChangeServingsArgs, InspectObjectArgs]:
        """Return the validated, strongly-typed arguments model."""
        if self.operation == OperationType.REPLACE_MEAL:
            return ReplaceMealArgs.model_validate(self.arguments)
        elif self.operation == OperationType.CHANGE_SERVINGS:
            return ChangeServingsArgs.model_validate(self.arguments)
        elif self.operation == OperationType.INSPECT_OBJECT:
            return InspectObjectArgs.model_validate(self.arguments)
        raise ValueError(f"Unsupported operation: {self.operation}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionProposal":
        """Construct and validate an ActionProposal from a Python dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "ActionProposal":
        """Construct and validate an ActionProposal from a JSON string."""
        return cls.model_validate_json(json_str)


class ActionResult(BaseModel):
    """Authoritative result returned by the runtime after processing a proposal."""

    model_config = ConfigDict(extra="forbid")

    status: ActionStatus = Field(..., description="Final transaction status")
    operation: str = Field(..., description="Target operation name")
    plan_version_before: int = Field(..., ge=1, description="Plan version prior to execution")
    plan_version_after: int = Field(..., ge=1, description="Plan version following execution")
    message: str = Field(..., description="Informative status message")
    execution_duration_ms: float = Field(..., ge=0.0, description="Execution time in milliseconds")
    rejection_reason: Optional[str] = Field(default=None, description="Detailed failure or rejection reason")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Diagnostic or state delta metadata")
