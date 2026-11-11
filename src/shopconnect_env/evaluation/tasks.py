"""Benchmark task fixtures, starting-plan fixtures, and strict validation engine.

Part of Stage 6 Phase 6.3 evaluation infrastructure:
- PlanFixture container and plan fixture validation.
- BenchmarkTask container and task fixture validation.
- Strict loaders ensuring zero silent repairs and full domain/policy compliance.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from shopconnect_env.domain.models import (
    DAYS_OF_WEEK,
    MEAL_TYPES,
    GroceryBasket,
    Ingredient,
    Meal,
    MealPlan,
    Recipe,
    UserPreferences,
    make_slot_key,
)
from shopconnect_env.operations.basket_calculator import calculate_grocery_basket
from shopconnect_env.validation.policy_engine import PolicyEngine


class FixtureValidationError(ValueError):
    """Raised when plan fixture data violates domain, schema, or policy rules."""
    pass


class TaskValidationError(ValueError):
    """Raised when benchmark task data is invalid, malformed, or references unknown entities."""
    pass


class ExpectedOutcome(str, Enum):
    """Allowlisted expected transaction outcome categories."""
    COMMIT = "COMMIT"
    REJECT = "REJECT"
    CLARIFY = "CLARIFY"


@dataclass(frozen=True)
class PlanFixture:
    """A deterministic starting-plan fixture for comparative evaluation.

    Attributes:
        fixture_id: Unique fixture identifier (e.g. 'fixture_standard').
        name: Human-readable scenario title.
        description: Description of the starting scenario and constraints.
        plan: The authoritative MealPlan domain model.
        basket: The verified authoritative GroceryBasket.
        raw_data: Immutable snapshot of the raw fixture dictionary.
    """
    fixture_id: str
    name: str
    description: str
    plan: MealPlan
    basket: GroceryBasket
    raw_data: Dict[str, Any]


@dataclass(frozen=True)
class BenchmarkTask:
    """A frozen evaluation scenario for comparing Conditions A, B, and C.

    Attributes:
        task_id: Unique stable task identifier (e.g. 'task_01_commit_replace_wed_dinner').
        fixture_id: Reference to the starting plan fixture.
        category: Task categorization (e.g. 'meal_replacement', 'budget_overflow').
        user_request: The natural language prompt delivered to the agent.
        expected_outcome: COMMIT, REJECT, or CLARIFY.
        expected_operation: Expected operation type ('replace_meal', 'change_servings', or None).
        expected_target: Optional dict {'day': str, 'meal_type': str} for target slot.
        expected_day: Day of week for target slot (when applicable).
        expected_meal_type: Meal slot type (when applicable).
        expected_recipe_id: Recipe ID expected to be scheduled (when applicable).
        expected_servings: Servings count expected (when applicable).
        expected_rejection_category: Expected policy/validation failure category (when REJECT).
        clarification_detail: Explanation of what is ambiguous/missing (when CLARIFY).
    """
    task_id: str
    fixture_id: str
    category: str
    user_request: str
    expected_outcome: ExpectedOutcome
    expected_operation: Optional[str] = None
    expected_target: Optional[Dict[str, str]] = None
    expected_day: Optional[str] = None
    expected_meal_type: Optional[str] = None
    expected_recipe_id: Optional[str] = None
    expected_servings: Optional[int] = None
    expected_rejection_category: Optional[str] = None
    clarification_detail: Optional[str] = None


def load_plan_fixtures(
    fixtures_path: Union[str, Path],
    catalog: Dict[str, Recipe],
    ingredients: Dict[str, Ingredient],
    expected_count: Optional[int] = 4,
) -> Dict[str, PlanFixture]:
    """Load and strictly validate all starting plan fixtures from JSON.

    Validates:
    1. File exists and contains valid JSON.
    2. Exactly `expected_count` fixtures are present (if expected_count is set).
    3. Each fixture contains exactly 21 unique slot coordinates.
    4. All assigned recipe IDs exist in catalog.
    5. Starting dietary preferences are satisfied via PolicyEngine.
    6. Monitored allergens are satisfied via PolicyEngine.
    7. Starting weekly budget is satisfied via PolicyEngine.
    8. Grocery basket matches deterministic calculate_grocery_basket output.
    9. Does not silently repair invalid data; raises FixtureValidationError.
    """
    path = Path(fixtures_path)
    if not path.exists():
        raise FixtureValidationError(f"Plan fixtures file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as err:
        raise FixtureValidationError(f"Malformed JSON in plan fixtures file: {err}")

    if not isinstance(data, dict):
        raise FixtureValidationError(f"Expected dict at root of plan fixtures JSON, got {type(data).__name__}.")

    if expected_count is not None and len(data) != expected_count:
        raise FixtureValidationError(
            f"Expected exactly {expected_count} plan fixtures, but found {len(data)}: {list(data.keys())}."
        )

    engine = PolicyEngine(catalog=catalog, ingredients=ingredients)
    fixtures: Dict[str, PlanFixture] = {}

    for fixture_id, fix_dict in data.items():
        if not isinstance(fix_dict, dict):
            raise FixtureValidationError(f"Fixture '{fixture_id}' must be a dictionary.")

        if fix_dict.get("fixture_id") != fixture_id:
            raise FixtureValidationError(
                f"Fixture key '{fixture_id}' does not match inner fixture_id '{fix_dict.get('fixture_id')}'."
            )

        # Validate schedule slots
        schedule = fix_dict.get("schedule")
        if not isinstance(schedule, list):
            raise FixtureValidationError(f"Fixture '{fixture_id}' missing 'schedule' list.")

        if len(schedule) != 21:
            raise FixtureValidationError(
                f"Fixture '{fixture_id}' schedule must contain exactly 21 slots, found {len(schedule)}."
            )

        seen_coords: Set[tuple] = set()
        for slot in schedule:
            if not isinstance(slot, dict):
                raise FixtureValidationError(f"Fixture '{fixture_id}' slot must be a dict.")
            day = slot.get("day")
            slot_type = slot.get("meal_type")
            status = slot.get("status")

            if day not in DAYS_OF_WEEK:
                raise FixtureValidationError(f"Fixture '{fixture_id}' has invalid day '{day}'.")
            if slot_type not in MEAL_TYPES:
                raise FixtureValidationError(f"Fixture '{fixture_id}' has invalid meal_type '{slot_type}'.")
            if (day, slot_type) in seen_coords:
                raise FixtureValidationError(f"Fixture '{fixture_id}' has duplicate slot ({day}, {slot_type}).")
            seen_coords.add((day, slot_type))

            if status == "assigned":
                rec_id = slot.get("recipe_id")
                serv = slot.get("servings")
                if not rec_id or rec_id not in catalog:
                    raise FixtureValidationError(
                        f"Fixture '{fixture_id}' assigned slot ({day}, {slot_type}) references invalid recipe_id '{rec_id}'."
                    )
                if not isinstance(serv, int) or serv <= 0:
                    raise FixtureValidationError(
                        f"Fixture '{fixture_id}' assigned slot ({day}, {slot_type}) has invalid servings: {serv}."
                    )
            elif status == "unfilled":
                if slot.get("recipe_id") is not None:
                    raise FixtureValidationError(
                        f"Fixture '{fixture_id}' unfilled slot ({day}, {slot_type}) has non-null recipe_id."
                    )
            else:
                raise FixtureValidationError(
                    f"Fixture '{fixture_id}' slot ({day}, {slot_type}) has invalid status '{status}'."
                )

        # Preferences
        raw_prefs = fix_dict.get("preferences", {})
        try:
            prefs = UserPreferences(
                diet_tags=list(raw_prefs.get("diet_tags", [])),
                allergies=list(raw_prefs.get("allergies", [])),
                weekly_budget=Decimal(str(raw_prefs.get("weekly_budget", "150.00"))),
                household_size=int(raw_prefs.get("household_size", 2)),
            )
        except Exception as e:
            raise FixtureValidationError(f"Fixture '{fixture_id}' has invalid preferences: {e}")

        # Construct MealPlan
        plan = MealPlan(
            plan_id=fix_dict.get("plan_id", f"plan_{fixture_id}"),
            plan_version=int(fix_dict.get("plan_version", 1)),
            preferences=prefs,
        )
        for slot in schedule:
            if slot.get("status") == "assigned":
                plan.set_meal(Meal(
                    day=slot["day"],
                    meal_type=slot["meal_type"],
                    recipe_id=slot["recipe_id"],
                    servings=slot["servings"],
                ))

        # Recompute GroceryBasket deterministically
        calculated_basket = calculate_grocery_basket(plan, catalog, ingredients)

        # Verify against PolicyEngine
        diet_decision = engine.validate_dietary_restrictions(plan)
        if not diet_decision.allowed:
            raise FixtureValidationError(
                f"Fixture '{fixture_id}' fails dietary policy: {diet_decision.reason}"
            )

        allergen_decision = engine.validate_allergens(plan)
        if not allergen_decision.allowed:
            raise FixtureValidationError(
                f"Fixture '{fixture_id}' fails allergen policy: {allergen_decision.reason}"
            )

        budget_decision = engine.validate_budget(plan)
        if not budget_decision.allowed:
            raise FixtureValidationError(
                f"Fixture '{fixture_id}' fails budget policy: {budget_decision.reason}"
            )

        fixtures[fixture_id] = PlanFixture(
            fixture_id=fixture_id,
            name=fix_dict.get("name", fixture_id),
            description=fix_dict.get("description", ""),
            plan=plan,
            basket=calculated_basket,
            raw_data=fix_dict,
        )

    return fixtures


def load_benchmark_tasks(
    tasks_path: Union[str, Path],
    fixtures: Dict[str, PlanFixture],
    catalog: Dict[str, Recipe],
    expected_count: Optional[int] = 25,
) -> List[BenchmarkTask]:
    """Load and strictly validate all benchmark tasks from JSONL.

    Validates:
    1. File exists and each line is valid JSON.
    2. Exactly `expected_count` tasks are present (if expected_count is set).
    3. Task IDs are unique and non-empty.
    4. Each task references an existing fixture in `fixtures`.
    5. Expected outcome is one of COMMIT, REJECT, CLARIFY.
    6. COMMIT tasks specify valid coordinates, valid catalog recipe_id, and positive servings.
    7. REJECT tasks specify a non-empty expected_rejection_category.
    8. CLARIFY tasks specify a non-empty clarification_detail.
    9. Does not silently repair invalid data; raises TaskValidationError.
    """
    path = Path(tasks_path)
    if not path.exists():
        raise TaskValidationError(f"Tasks file not found: {path}")

    tasks: List[BenchmarkTask] = []
    seen_ids: Set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                task_dict = json.loads(line_str)
            except json.JSONDecodeError as err:
                raise TaskValidationError(f"Line {line_num} in {path.name} is not valid JSON: {err}")

            if not isinstance(task_dict, dict):
                raise TaskValidationError(f"Line {line_num} in {path.name} is not a JSON object.")

            task_id = task_dict.get("task_id")
            if not task_id or not isinstance(task_id, str) or not task_id.strip():
                raise TaskValidationError(f"Line {line_num} missing valid 'task_id'.")

            if task_id in seen_ids:
                raise TaskValidationError(f"Duplicate task_id '{task_id}' on line {line_num}.")
            seen_ids.add(task_id)

            fixture_id = task_dict.get("fixture_id")
            if not fixture_id or fixture_id not in fixtures:
                raise TaskValidationError(
                    f"Task '{task_id}' references unknown fixture_id '{fixture_id}'."
                )

            user_req = task_dict.get("user_request")
            if not user_req or not isinstance(user_req, str) or not user_req.strip():
                raise TaskValidationError(f"Task '{task_id}' missing 'user_request'.")

            raw_outcome = task_dict.get("expected_outcome")
            try:
                outcome = ExpectedOutcome(raw_outcome)
            except ValueError:
                raise TaskValidationError(
                    f"Task '{task_id}' has invalid expected_outcome '{raw_outcome}'. "
                    f"Must be one of {[e.value for e in ExpectedOutcome]}."
                )

            target = task_dict.get("expected_target")
            exp_op = task_dict.get("expected_operation")
            exp_day = task_dict.get("expected_day")
            exp_slot = task_dict.get("expected_meal_type")
            exp_rec = task_dict.get("expected_recipe_id")
            exp_serv = task_dict.get("expected_servings")
            exp_rej = task_dict.get("expected_rejection_category")
            clar_detail = task_dict.get("clarification_detail")

            # Target consistency validation: if target is provided, enforce exact equality with exp_day/exp_slot
            if target is not None:
                if not isinstance(target, dict):
                    raise TaskValidationError(f"Task '{task_id}' has malformed expected_target: must be dict or null.")
                target_day = target.get("day")
                target_slot = target.get("meal_type")
                if target_day is not None and target_day not in DAYS_OF_WEEK:
                    raise TaskValidationError(f"Task '{task_id}' expected_target has invalid day '{target_day}'.")
                if target_slot is not None and target_slot not in MEAL_TYPES:
                    raise TaskValidationError(f"Task '{task_id}' expected_target has invalid meal_type '{target_slot}'.")
                if exp_day is not None and target_day != exp_day:
                    raise TaskValidationError(
                        f"Task '{task_id}' target consistency error: expected_target['day'] '{target_day}' "
                        f"does not match expected_day '{exp_day}'."
                    )
                if exp_slot is not None and target_slot != exp_slot:
                    raise TaskValidationError(
                        f"Task '{task_id}' target consistency error: expected_target['meal_type'] '{target_slot}' "
                        f"does not match expected_meal_type '{exp_slot}'."
                    )

            if outcome == ExpectedOutcome.COMMIT:
                if exp_op not in ("replace_meal", "change_servings"):
                    raise TaskValidationError(
                        f"COMMIT task '{task_id}' requires expected_operation in ('replace_meal', 'change_servings'), got '{exp_op}'."
                    )
                if not exp_day or exp_day not in DAYS_OF_WEEK:
                    raise TaskValidationError(
                        f"COMMIT task '{task_id}' requires valid expected_day in {DAYS_OF_WEEK}, got '{exp_day}'."
                    )
                if not exp_slot or exp_slot not in MEAL_TYPES:
                    raise TaskValidationError(
                        f"COMMIT task '{task_id}' requires valid expected_meal_type in {MEAL_TYPES}, got '{exp_slot}'."
                    )
                if target is None:
                    raise TaskValidationError(
                        f"COMMIT task '{task_id}' requires non-null expected_target."
                    )
                if target.get("day") != exp_day or target.get("meal_type") != exp_slot:
                    raise TaskValidationError(
                        f"COMMIT task '{task_id}' expected_target {target} must match ({exp_day}, {exp_slot})."
                    )
                if not exp_rec or exp_rec not in catalog:
                    raise TaskValidationError(
                        f"COMMIT task '{task_id}' requires valid catalog expected_recipe_id, got '{exp_rec}'."
                    )
                if not isinstance(exp_serv, int) or exp_serv <= 0 or exp_serv > 20:
                    raise TaskValidationError(
                        f"COMMIT task '{task_id}' requires positive expected_servings (1-20), got {exp_serv}."
                    )
                if exp_rej is not None:
                    raise TaskValidationError(
                        f"COMMIT task '{task_id}' must not define expected_rejection_category."
                    )
                if clar_detail is not None:
                    raise TaskValidationError(
                        f"COMMIT task '{task_id}' must not define clarification_detail."
                    )

                # Differentiate replace_meal vs change_servings against starting fixture state
                fixture = fixtures[fixture_id]
                slot_key = make_slot_key(exp_day, exp_slot)
                if exp_op == "change_servings":
                    if slot_key not in fixture.plan.meals:
                        raise TaskValidationError(
                            f"change_servings task '{task_id}' targets unassigned slot '{slot_key}' in fixture '{fixture_id}'."
                        )
                    assigned_meal = fixture.plan.meals[slot_key]
                    if assigned_meal.recipe_id != exp_rec:
                        raise TaskValidationError(
                            f"change_servings task '{task_id}' expected_recipe_id '{exp_rec}' does not match "
                            f"assigned recipe '{assigned_meal.recipe_id}' in slot '{slot_key}'."
                        )
                    if assigned_meal.servings == exp_serv:
                        raise TaskValidationError(
                            f"change_servings task '{task_id}' expected_servings {exp_serv} is identical to "
                            f"current servings in fixture '{fixture_id}' (would be a no-op)."
                        )
                elif exp_op == "replace_meal":
                    if slot_key in fixture.plan.meals:
                        assigned_meal = fixture.plan.meals[slot_key]
                        if assigned_meal.recipe_id == exp_rec and assigned_meal.servings == exp_serv:
                            raise TaskValidationError(
                                f"replace_meal task '{task_id}' replaces slot '{slot_key}' with identical recipe "
                                f"'{exp_rec}' and servings {exp_serv} (would be a no-op)."
                            )

            elif outcome == ExpectedOutcome.REJECT:
                if not exp_rej or not isinstance(exp_rej, str) or not exp_rej.strip():
                    raise TaskValidationError(
                        f"REJECT task '{task_id}' requires non-empty expected_rejection_category."
                    )
                if clar_detail is not None:
                    raise TaskValidationError(
                        f"REJECT task '{task_id}' must not define clarification_detail."
                    )

            elif outcome == ExpectedOutcome.CLARIFY:
                if not clar_detail or not isinstance(clar_detail, str) or not clar_detail.strip():
                    raise TaskValidationError(
                        f"CLARIFY task '{task_id}' requires non-empty clarification_detail."
                    )
                if exp_rej is not None:
                    raise TaskValidationError(
                        f"CLARIFY task '{task_id}' must not define expected_rejection_category."
                    )

            tasks.append(BenchmarkTask(
                task_id=task_id,
                fixture_id=fixture_id,
                category=task_dict.get("category", "unspecified"),
                user_request=user_req,
                expected_outcome=outcome,
                expected_operation=exp_op,
                expected_target=target,
                expected_day=exp_day,
                expected_meal_type=exp_slot,
                expected_recipe_id=exp_rec,
                expected_servings=exp_serv,
                expected_rejection_category=exp_rej,
                clarification_detail=clar_detail,
            ))

    if expected_count is not None and len(tasks) != expected_count:
        raise TaskValidationError(
            f"Expected exactly {expected_count} tasks in {path.name}, but found {len(tasks)}."
        )

    return tasks
