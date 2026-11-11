"""Comprehensive tests for Stage 6 Phase 6.3 Benchmark Tasks and Plan Fixtures.

Covers:
1. All four fixtures load successfully.
2. Each fixture contains 21 unique slots.
3. All fixture recipes exist in the catalog.
4. Starting dietary and allergen policies are satisfied.
5. Basket calculations and budgets are correct.
6. Pescatarian fixture includes the intended salmon recipe.
7. Tight-budget fixture has the specified $20 budget and $14.50 committed cost.
8. All 25 tasks load successfully.
9. Task IDs are unique.
10. All fixture references resolve.
11. Every ExpectedOutcome has the correct required fields.
12. Invalid fixture/task records are rejected.
13. JSONL is deterministic and contains exactly 25 tasks.
"""

from decimal import Decimal
import json
from pathlib import Path
import pytest

from shopconnect_env.domain.models import (
    DAYS_OF_WEEK,
    MEAL_TYPES,
    load_catalog_from_json,
    make_slot_key,
)
from shopconnect_env.evaluation import (
    BenchmarkTask,
    ExpectedOutcome,
    FixtureValidationError,
    PlanFixture,
    TaskValidationError,
    load_benchmark_tasks,
    load_plan_fixtures,
)
from shopconnect_env.operations.basket_calculator import calculate_grocery_basket
from shopconnect_env.validation.policy_engine import PolicyEngine


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def catalog_data():
    """Load canonical ingredients and recipe catalog from seed JSON."""
    catalog_path = Path(__file__).parent.parent / "data" / "catalog_seed.json"
    return load_catalog_from_json(catalog_path)


@pytest.fixture
def plan_fixtures(catalog_data):
    """Load and validate all starting plan fixtures."""
    ingredients, recipes = catalog_data
    fixtures_path = Path(__file__).parent.parent / "data" / "plan_fixtures.json"
    return load_plan_fixtures(fixtures_path, recipes, ingredients, expected_count=4)


@pytest.fixture
def benchmark_tasks(plan_fixtures, catalog_data):
    """Load and validate all 25 benchmark tasks."""
    _, recipes = catalog_data
    tasks_path = Path(__file__).parent.parent / "data" / "tasks.jsonl"
    return load_benchmark_tasks(tasks_path, plan_fixtures, recipes, expected_count=25)


# =============================================================================
# PART 1: PLAN FIXTURES TESTS
# =============================================================================

def test_all_four_fixtures_load_successfully(plan_fixtures):
    """1. All four starting plan fixtures load and validate cleanly."""
    assert len(plan_fixtures) == 4
    expected_ids = {
        "fixture_standard",
        "fixture_vegetarian_profile",
        "fixture_pescatarian_profile",
        "fixture_tight_budget",
    }
    assert set(plan_fixtures.keys()) == expected_ids
    for fid, fixture in plan_fixtures.items():
        assert isinstance(fixture, PlanFixture)
        assert fixture.fixture_id == fid
        assert fixture.plan is not None
        assert fixture.basket is not None


def test_each_fixture_contains_21_unique_slots(plan_fixtures):
    """2. Each fixture schedule strictly contains all 21 standard day/slot combinations."""
    expected_keys = {
        make_slot_key(d, m)
        for d in DAYS_OF_WEEK
        for m in MEAL_TYPES
    }
    assert len(expected_keys) == 21

    for fid, fixture in plan_fixtures.items():
        schedule = fixture.raw_data["schedule"]
        assert len(schedule) == 21, f"Fixture '{fid}' has {len(schedule)} slots, expected 21."

        slot_keys = {make_slot_key(s["day"], s["meal_type"]) for s in schedule}
        assert slot_keys == expected_keys, f"Fixture '{fid}' slot coordinates do not match 21 weekly slots."


def test_all_fixture_recipes_exist_in_catalog(plan_fixtures, catalog_data):
    """3. All scheduled meals reference valid recipe IDs from the catalog."""
    _, recipes = catalog_data
    for fid, fixture in plan_fixtures.items():
        for meal in fixture.plan.meals.values():
            assert meal.recipe_id in recipes, (
                f"Fixture '{fid}' references missing recipe '{meal.recipe_id}'."
            )


def test_starting_dietary_and_allergen_policies_satisfied(plan_fixtures, catalog_data):
    """4. Starting dietary preferences and allergen restrictions are strictly satisfied."""
    ingredients, recipes = catalog_data
    engine = PolicyEngine(catalog=recipes, ingredients=ingredients)

    for fid, fixture in plan_fixtures.items():
        diet_decision = engine.validate_dietary_restrictions(fixture.plan)
        assert diet_decision.allowed is True, (
            f"Fixture '{fid}' failed dietary check: {diet_decision.reason}"
        )

        allergen_decision = engine.validate_allergens(fixture.plan)
        assert allergen_decision.allowed is True, (
            f"Fixture '{fid}' failed allergen check: {allergen_decision.reason}"
        )


def test_basket_calculations_and_budgets_correct(plan_fixtures, catalog_data):
    """5. Basket items match calculate_grocery_basket and weekly budgets are satisfied."""
    ingredients, recipes = catalog_data
    engine = PolicyEngine(catalog=recipes, ingredients=ingredients)

    for fid, fixture in plan_fixtures.items():
        # Verify deterministic basket recomputation
        calculated = calculate_grocery_basket(fixture.plan, recipes, ingredients)
        assert fixture.basket.items == calculated.items, (
            f"Fixture '{fid}' basket items do not match recomputed grocery basket."
        )

        # Verify budget compliance
        budget_decision = engine.validate_budget(fixture.plan)
        assert budget_decision.allowed is True, (
            f"Fixture '{fid}' failed budget check: {budget_decision.reason}"
        )


def test_pescatarian_fixture_includes_salmon_recipe(plan_fixtures, catalog_data):
    """6. Pescatarian fixture schedules rec_pan_seared_salmon satisfying pescatarian requirement."""
    _, recipes = catalog_data
    fixture = plan_fixtures["fixture_pescatarian_profile"]

    # Verify pan-seared salmon is scheduled
    salmon_meals = [m for m in fixture.plan.meals.values() if m.recipe_id == "rec_pan_seared_salmon"]
    assert len(salmon_meals) == 1
    assert salmon_meals[0].day == "Friday"
    assert salmon_meals[0].meal_type == "dinner"

    # Verify recipe has pescatarian tag in catalog
    salmon_recipe = recipes["rec_pan_seared_salmon"]
    assert "pescatarian" in salmon_recipe.tags


def test_tight_budget_fixture_specifications(plan_fixtures, catalog_data):
    """7. Tight-budget fixture has budget $20.00, salmon with 2 servings, and committed cost $14.50."""
    _, recipes = catalog_data
    fixture = plan_fixtures["fixture_tight_budget"]

    # Budget is $20.00
    assert fixture.plan.preferences.weekly_budget == Decimal("20.00")

    # Salmon meal has 2 servings
    assert "Friday_dinner" in fixture.plan.meals
    salmon_meal = fixture.plan.meals["Friday_dinner"]
    assert salmon_meal.recipe_id == "rec_pan_seared_salmon"
    assert salmon_meal.servings == 2

    # Committed cost is exactly $14.50, remaining budget is $5.50
    total_cost = fixture.plan.calculate_total_cost(recipes)
    assert total_cost == Decimal("14.50")
    remaining_budget = fixture.plan.preferences.weekly_budget - total_cost
    assert remaining_budget == Decimal("5.50")


# =============================================================================
# PART 2: BENCHMARK TASKS TESTS
# =============================================================================

def test_all_25_tasks_load_successfully(benchmark_tasks):
    """8. All 25 benchmark tasks load into typed BenchmarkTask containers."""
    assert len(benchmark_tasks) == 25
    for task in benchmark_tasks:
        assert isinstance(task, BenchmarkTask)
        assert task.task_id.startswith("task_")
        assert len(task.user_request.strip()) > 0


def test_task_ids_are_unique(benchmark_tasks):
    """9. Task IDs are strictly unique across all 25 benchmark records."""
    task_ids = [t.task_id for t in benchmark_tasks]
    assert len(task_ids) == 25
    assert len(set(task_ids)) == 25


def test_all_fixture_references_resolve(benchmark_tasks, plan_fixtures):
    """10. Every task references an existing, valid plan fixture ID."""
    for task in benchmark_tasks:
        assert task.fixture_id in plan_fixtures, (
            f"Task '{task.task_id}' references unknown fixture '{task.fixture_id}'."
        )


def test_every_expected_outcome_has_correct_required_fields(benchmark_tasks, catalog_data):
    """11. Every ExpectedOutcome (COMMIT, REJECT, CLARIFY) adheres to strict field constraints."""
    _, recipes = catalog_data

    commit_count = 0
    reject_count = 0
    clarify_count = 0

    for task in benchmark_tasks:
        if task.expected_outcome == ExpectedOutcome.COMMIT:
            commit_count += 1
            assert task.expected_operation in ("replace_meal", "change_servings")
            assert task.expected_day in DAYS_OF_WEEK
            assert task.expected_meal_type in MEAL_TYPES
            assert task.expected_target == {"day": task.expected_day, "meal_type": task.expected_meal_type}
            assert task.expected_recipe_id in recipes
            assert isinstance(task.expected_servings, int) and 1 <= task.expected_servings <= 20
            assert task.expected_rejection_category is None
            assert task.clarification_detail is None

        elif task.expected_outcome == ExpectedOutcome.REJECT:
            reject_count += 1
            assert task.expected_rejection_category is not None
            assert len(task.expected_rejection_category.strip()) > 0
            assert task.clarification_detail is None

        elif task.expected_outcome == ExpectedOutcome.CLARIFY:
            clarify_count += 1
            assert task.clarification_detail is not None
            assert len(task.clarification_detail.strip()) > 0
            assert task.expected_rejection_category is None

    # Expected distribution
    assert commit_count == 13
    assert reject_count == 8
    assert clarify_count == 4


def test_invalid_fixture_and_task_records_rejected(catalog_data, plan_fixtures, tmp_path):
    """12. Strict validators reject malformed or policy-violating fixtures and tasks."""
    ingredients, recipes = catalog_data

    # A. Malformed fixture: only 20 slots
    bad_fixture_data = {
        "fixture_bad": {
            "fixture_id": "fixture_bad",
            "name": "Bad Fixture",
            "description": "Missing 1 slot",
            "plan_id": "plan_bad",
            "preferences": {"diet_tags": [], "allergies": [], "weekly_budget": "150.00", "household_size": 2},
            "schedule": plan_fixtures["fixture_standard"].raw_data["schedule"][:-1],  # 20 slots
        }
    }
    bad_fix_file = tmp_path / "bad_fixtures.json"
    with open(bad_fix_file, "w") as f:
        json.dump(bad_fixture_data, f)

    with pytest.raises(FixtureValidationError, match="must contain exactly 21 slots"):
        load_plan_fixtures(bad_fix_file, recipes, ingredients, expected_count=1)

    # B. Malformed fixture: dietary violation (meat scheduled in vegetarian plan)
    veg_bad_schedule = [dict(s) for s in plan_fixtures["fixture_vegetarian_profile"].raw_data["schedule"]]
    for s in veg_bad_schedule:
        if s["day"] == "Wednesday" and s["meal_type"] == "dinner":
            s["recipe_id"] = "rec_baked_chicken_rice"  # Non-vegetarian
    bad_veg_data = {
        "fixture_veg_bad": {
            "fixture_id": "fixture_veg_bad",
            "name": "Bad Veg Fixture",
            "description": "Chicken in veg plan",
            "plan_id": "plan_veg_bad",
            "preferences": {"diet_tags": ["vegetarian"], "allergies": [], "weekly_budget": "150.00", "household_size": 2},
            "schedule": veg_bad_schedule,
        }
    }
    bad_veg_file = tmp_path / "bad_veg_fixtures.json"
    with open(bad_veg_file, "w") as f:
        json.dump(bad_veg_data, f)

    with pytest.raises(FixtureValidationError, match="fails dietary policy"):
        load_plan_fixtures(bad_veg_file, recipes, ingredients, expected_count=1)

    # C. Malformed task: references unknown fixture ID
    bad_task_file = tmp_path / "bad_tasks.jsonl"
    with open(bad_task_file, "w") as f:
        f.write(json.dumps({
            "task_id": "bad_task_01",
            "fixture_id": "nonexistent_fixture_999",
            "user_request": "Replace dinner",
            "expected_outcome": "COMMIT",
            "expected_day": "Monday",
            "expected_meal_type": "dinner",
            "expected_recipe_id": "rec_berry_oatmeal",
            "expected_servings": 2,
        }) + "\n")

    with pytest.raises(TaskValidationError, match="unknown fixture_id"):
        load_benchmark_tasks(bad_task_file, plan_fixtures, recipes, expected_count=1)

    # D. Malformed task: invalid outcome category
    with open(bad_task_file, "w") as f:
        f.write(json.dumps({
            "task_id": "bad_task_02",
            "fixture_id": "fixture_standard",
            "user_request": "Do something",
            "expected_outcome": "MAYBE",
        }) + "\n")

    with pytest.raises(TaskValidationError, match="invalid expected_outcome"):
        load_benchmark_tasks(bad_task_file, plan_fixtures, recipes, expected_count=1)


def test_jsonl_deterministic_and_task_count():
    """13. tasks.jsonl contains exactly 25 non-empty, parseable JSON lines."""
    tasks_path = Path(__file__).parent.parent / "data" / "tasks.jsonl"
    assert tasks_path.exists()

    with open(tasks_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 25, f"Expected exactly 25 task lines, found {len(lines)}."

    for idx, line in enumerate(lines, 1):
        parsed = json.loads(line)
        assert isinstance(parsed, dict), f"Line {idx} is not a valid JSON dictionary."
        assert "task_id" in parsed
        assert "expected_outcome" in parsed


def test_commit_tasks_distinguish_replace_meal_and_change_servings(benchmark_tasks, plan_fixtures, catalog_data):
    """14. Unambiguously distinguish replace_meal from change_servings with valid ground truth."""
    _, recipes = catalog_data

    replace_tasks = [t for t in benchmark_tasks if t.expected_outcome == ExpectedOutcome.COMMIT and t.expected_operation == "replace_meal"]
    change_tasks = [t for t in benchmark_tasks if t.expected_outcome == ExpectedOutcome.COMMIT and t.expected_operation == "change_servings"]

    assert len(replace_tasks) == 9
    assert len(change_tasks) == 4
    assert len(replace_tasks) + len(change_tasks) == 13

    # Validate replace_meal tasks
    for task in replace_tasks:
        assert task.expected_recipe_id in recipes
        assert isinstance(task.expected_servings, int) and 1 <= task.expected_servings <= 20
        # If slot was assigned in fixture, new recipe or servings must differ (not a no-op)
        fixture = plan_fixtures[task.fixture_id]
        slot_key = make_slot_key(task.expected_day, task.expected_meal_type)
        if slot_key in fixture.plan.meals:
            assigned = fixture.plan.meals[slot_key]
            assert not (assigned.recipe_id == task.expected_recipe_id and assigned.servings == task.expected_servings), (
                f"Task '{task.task_id}' is a no-op replacement against fixture '{task.fixture_id}'."
            )

    # Validate change_servings tasks
    for task in change_tasks:
        assert task.expected_recipe_id in recipes
        assert isinstance(task.expected_servings, int) and 1 <= task.expected_servings <= 20
        fixture = plan_fixtures[task.fixture_id]
        slot_key = make_slot_key(task.expected_day, task.expected_meal_type)
        assert slot_key in fixture.plan.meals, (
            f"change_servings task '{task.task_id}' targets unassigned slot in fixture '{task.fixture_id}'."
        )
        assigned = fixture.plan.meals[slot_key]
        assert assigned.recipe_id == task.expected_recipe_id
        assert assigned.servings != task.expected_servings, (
            f"change_servings task '{task.task_id}' has identical servings (no-op) against fixture '{task.fixture_id}'."
        )


def test_target_consistency_and_contradiction_rejection(plan_fixtures, catalog_data, tmp_path):
    """15. Enforce target consistency and reject contradictory task records."""
    _, recipes = catalog_data

    # A. Contradictory expected_target['day'] != expected_day
    bad_task_file = tmp_path / "target_contradiction_day.jsonl"
    with open(bad_task_file, "w") as f:
        f.write(json.dumps({
            "task_id": "bad_target_01",
            "fixture_id": "fixture_standard",
            "user_request": "Replace Monday breakfast",
            "expected_outcome": "COMMIT",
            "expected_operation": "replace_meal",
            "expected_target": {"day": "Tuesday", "meal_type": "breakfast"},
            "expected_day": "Monday",
            "expected_meal_type": "breakfast",
            "expected_recipe_id": "rec_berry_oatmeal",
            "expected_servings": 2,
        }) + "\n")

    with pytest.raises(TaskValidationError, match="target consistency error"):
        load_benchmark_tasks(bad_task_file, plan_fixtures, recipes, expected_count=1)

    # B. Contradictory expected_target['meal_type'] != expected_meal_type
    bad_task_file = tmp_path / "target_contradiction_slot.jsonl"
    with open(bad_task_file, "w") as f:
        f.write(json.dumps({
            "task_id": "bad_target_02",
            "fixture_id": "fixture_standard",
            "user_request": "Replace Monday breakfast",
            "expected_outcome": "COMMIT",
            "expected_operation": "replace_meal",
            "expected_target": {"day": "Monday", "meal_type": "dinner"},
            "expected_day": "Monday",
            "expected_meal_type": "breakfast",
            "expected_recipe_id": "rec_berry_oatmeal",
            "expected_servings": 2,
        }) + "\n")

    with pytest.raises(TaskValidationError, match="target consistency error"):
        load_benchmark_tasks(bad_task_file, plan_fixtures, recipes, expected_count=1)

    # C. COMMIT task missing expected_target
    bad_task_file = tmp_path / "missing_commit_target.jsonl"
    with open(bad_task_file, "w") as f:
        f.write(json.dumps({
            "task_id": "bad_target_03",
            "fixture_id": "fixture_standard",
            "user_request": "Replace Monday breakfast",
            "expected_outcome": "COMMIT",
            "expected_operation": "replace_meal",
            "expected_target": None,
            "expected_day": "Monday",
            "expected_meal_type": "breakfast",
            "expected_recipe_id": "rec_spinach_omelet",
            "expected_servings": 2,
        }) + "\n")

    with pytest.raises(TaskValidationError, match="requires non-null expected_target"):
        load_benchmark_tasks(bad_task_file, plan_fixtures, recipes, expected_count=1)

    # D. COMMIT task with invalid expected_operation
    bad_task_file = tmp_path / "invalid_op.jsonl"
    with open(bad_task_file, "w") as f:
        f.write(json.dumps({
            "task_id": "bad_target_04",
            "fixture_id": "fixture_standard",
            "user_request": "Do custom thing",
            "expected_outcome": "COMMIT",
            "expected_operation": "custom_operation",
            "expected_target": {"day": "Monday", "meal_type": "breakfast"},
            "expected_day": "Monday",
            "expected_meal_type": "breakfast",
            "expected_recipe_id": "rec_spinach_omelet",
            "expected_servings": 2,
        }) + "\n")

    with pytest.raises(TaskValidationError, match="requires expected_operation in"):
        load_benchmark_tasks(bad_task_file, plan_fixtures, recipes, expected_count=1)


def test_loader_determinism_and_idempotence(plan_fixtures, benchmark_tasks, catalog_data):
    """16. Fixture and task loaders are strictly deterministic and idempotent."""
    ingredients, recipes = catalog_data
    fixtures_path = Path(__file__).parent.parent / "data" / "plan_fixtures.json"
    tasks_path = Path(__file__).parent.parent / "data" / "tasks.jsonl"

    fixtures_2 = load_plan_fixtures(fixtures_path, recipes, ingredients, expected_count=4)
    tasks_2 = load_benchmark_tasks(tasks_path, fixtures_2, recipes, expected_count=25)

    assert len(fixtures_2) == len(plan_fixtures)
    for fid, f1 in plan_fixtures.items():
        f2 = fixtures_2[fid]
        assert f1.fixture_id == f2.fixture_id
        assert f1.name == f2.name
        assert f1.description == f2.description
        assert f1.plan.plan_id == f2.plan.plan_id
        assert f1.plan.plan_version == f2.plan.plan_version
        assert f1.plan.preferences == f2.plan.preferences
        assert f1.plan.meals == f2.plan.meals
        assert f1.basket.items == f2.basket.items

    assert len(tasks_2) == len(benchmark_tasks)
    for t1, t2 in zip(benchmark_tasks, tasks_2):
        assert t1.task_id == t2.task_id
        assert t1.fixture_id == t2.fixture_id
        assert t1.category == t2.category
        assert t1.user_request == t2.user_request
        assert t1.expected_outcome == t2.expected_outcome
        assert t1.expected_operation == t2.expected_operation
        assert t1.expected_target == t2.expected_target
        assert t1.expected_day == t2.expected_day
        assert t1.expected_meal_type == t2.expected_meal_type
        assert t1.expected_recipe_id == t2.expected_recipe_id
        assert t1.expected_servings == t2.expected_servings
        assert t1.expected_rejection_category == t2.expected_rejection_category
        assert t1.clarification_detail == t2.clarification_detail
