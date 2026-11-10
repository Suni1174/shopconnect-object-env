"""Unit tests for Stage 3 Pydantic action schemas."""

import json
import pytest
from pydantic import ValidationError

from shopconnect_env.validation.schemas import (
    ActionProposal,
    ActionResult,
    ActionStatus,
    ChangeServingsArgs,
    InspectObjectArgs,
    OperationType,
    ReplaceMealArgs,
)


def test_valid_replace_meal_proposal():
    """Verify construction of a valid replace_meal proposal from dict."""
    data = {
        "operation": "replace_meal",
        "arguments": {
            "day": "Wednesday",
            "meal_type": "dinner",
            "recipe_id": "rec_pan_seared_salmon",
            "servings": 2,
        },
        "reason": "Swap Wednesday dinner to salmon",
        "expected_plan_version": 1,
    }
    proposal = ActionProposal.from_dict(data)
    assert proposal.operation == OperationType.REPLACE_MEAL
    assert proposal.expected_plan_version == 1
    assert proposal.reason == "Swap Wednesday dinner to salmon"

    args = proposal.get_typed_arguments()
    assert isinstance(args, ReplaceMealArgs)
    assert args.day == "Wednesday"
    assert args.meal_type == "dinner"
    assert args.recipe_id == "rec_pan_seared_salmon"
    assert args.servings == 2


def test_valid_change_servings_proposal():
    """Verify construction of a valid change_servings proposal from JSON string."""
    json_str = json.dumps({
        "operation": "change_servings",
        "arguments": {
            "day": "Tuesday",
            "meal_type": "lunch",
            "servings": 4,
        },
        "reason": "Family visiting, need 4 servings",
    })
    proposal = ActionProposal.from_json(json_str)
    assert proposal.operation == OperationType.CHANGE_SERVINGS
    assert proposal.expected_plan_version is None

    args = proposal.get_typed_arguments()
    assert isinstance(args, ChangeServingsArgs)
    assert args.day == "Tuesday"
    assert args.meal_type == "lunch"
    assert args.servings == 4


def test_malformed_json():
    """Verify that syntactically malformed JSON raises JSONDecodeError."""
    bad_json = '{"operation": "replace_meal", "arguments": ... bad json ...}'
    with pytest.raises(Exception):  # JSONDecodeError
        ActionProposal.from_json(bad_json)


def test_unknown_operation():
    """Verify that unknown operation strings are rejected."""
    data = {
        "operation": "delete_all_meals",
        "arguments": {},
        "reason": "Malicious or unknown operation",
    }
    with pytest.raises(ValidationError):
        ActionProposal.from_dict(data)


def test_missing_required_arguments():
    """Verify that missing required arguments fail schema validation."""
    data = {
        "operation": "replace_meal",
        "arguments": {
            "day": "Wednesday",
            # missing meal_type and recipe_id
        },
        "reason": "Incomplete payload",
    }
    with pytest.raises(ValidationError):
        ActionProposal.from_dict(data)


def test_unexpected_fields_forbidden():
    """Verify that extra unexpected fields trigger rejection under ConfigDict(extra='forbid')."""
    # Extra field at proposal level
    data_proposal = {
        "operation": "replace_meal",
        "arguments": {
            "day": "Wednesday",
            "meal_type": "dinner",
            "recipe_id": "rec_01",
        },
        "reason": "Valid reason",
        "extra_evil_field": "injected",
    }
    with pytest.raises(ValidationError):
        ActionProposal.from_dict(data_proposal)

    # Extra field at arguments level
    data_args = {
        "operation": "replace_meal",
        "arguments": {
            "day": "Wednesday",
            "meal_type": "dinner",
            "recipe_id": "rec_01",
            "extra_side_effect": 123,
        },
        "reason": "Valid reason",
    }
    with pytest.raises(ValidationError):
        ActionProposal.from_dict(data_args)


def test_invalid_day_and_meal_type():
    """Verify that invalid days and meal slots are rejected."""
    # Invalid day
    with pytest.raises(ValidationError):
        ReplaceMealArgs(day="Funday", meal_type="dinner", recipe_id="rec_01")

    # Invalid meal slot
    with pytest.raises(ValidationError):
        ReplaceMealArgs(day="Monday", meal_type="midnight_snack", recipe_id="rec_01")


def test_zero_and_negative_servings_rejected():
    """Verify that zero and negative servings are rejected."""
    with pytest.raises(ValidationError):
        ReplaceMealArgs(day="Monday", meal_type="dinner", recipe_id="rec_01", servings=0)

    with pytest.raises(ValidationError):
        ChangeServingsArgs(day="Monday", meal_type="dinner", servings=-2)


def test_float_and_boolean_servings_explicitly_rejected():
    """Verify that floats and booleans are not coerced into integer servings."""
    # Float servings
    with pytest.raises(ValidationError):
        ReplaceMealArgs(day="Monday", meal_type="dinner", recipe_id="rec_01", servings=2.5)  # type: ignore

    with pytest.raises(ValidationError):
        ChangeServingsArgs(day="Monday", meal_type="dinner", servings=3.0)  # type: ignore

    # Boolean servings
    with pytest.raises(ValidationError):
        ReplaceMealArgs(day="Monday", meal_type="dinner", recipe_id="rec_01", servings=True)  # type: ignore

    with pytest.raises(ValidationError):
        ChangeServingsArgs(day="Monday", meal_type="dinner", servings=False)  # type: ignore


def test_wrong_argument_schema_for_operation():
    """Verify that passing change_servings arguments to replace_meal is caught."""
    data = {
        "operation": "replace_meal",
        "arguments": {
            # change_servings arguments, missing recipe_id
            "day": "Tuesday",
            "meal_type": "lunch",
            "servings": 4,
        },
        "reason": "Mismatched argument structure",
    }
    with pytest.raises(ValidationError):
        ActionProposal.from_dict(data)


def test_stale_version_field_parsing():
    """Verify that expected_plan_version parses correctly and rejects negative versions."""
    proposal = ActionProposal.from_dict({
        "operation": "change_servings",
        "arguments": {"day": "Tuesday", "meal_type": "lunch", "servings": 2},
        "reason": "Test",
        "expected_plan_version": 3,
    })
    assert proposal.expected_plan_version == 3

    # Negative version should fail
    with pytest.raises(ValidationError):
        ActionProposal.from_dict({
            "operation": "change_servings",
            "arguments": {"day": "Tuesday", "meal_type": "lunch", "servings": 2},
            "reason": "Test",
            "expected_plan_version": 0,
        })


def test_inspect_object_args_validation():
    """Verify InspectObjectArgs validation for target types."""
    valid = InspectObjectArgs(target_type="recipe", target_id="rec_01")
    assert valid.target_type == "recipe"
    assert valid.target_id == "rec_01"

    # Invalid target type
    with pytest.raises(ValidationError):
        InspectObjectArgs(target_type="database_table")
