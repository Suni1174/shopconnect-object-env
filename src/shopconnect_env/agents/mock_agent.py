"""Deterministic offline mock agent for testing and verification.

Implements BaseAgent to provide deterministic task-to-proposal mapping across
Conditions A, B, and C without external LLM dependencies, network calls, or
state mutation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from shopconnect_env.agents.base import AgentResponse, BaseAgent, ObjectInspector
from shopconnect_env.context.base import ContextResult
from shopconnect_env.domain.models import DAYS_OF_WEEK, MEAL_TYPES
from shopconnect_env.validation.schemas import (
    ActionProposal,
    OperationType,
)


def _extract_plan_version(serialized_context: str) -> Optional[int]:
    """Strictly extract and validate plan_version from serialized context.

    Returns the integer plan_version if present and >= 1.
    Rejects missing, boolean, float, string, negative, or zero values.
    """
    try:
        parsed = json.loads(serialized_context)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict) or "plan_version" not in parsed:
        return None
    v = parsed["plan_version"]
    # In Python, isinstance(True, int) is True; explicitly reject booleans
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        return None
    return v


class MockAgent(BaseAgent):
    """Deterministic offline mock agent for integration testing.

    Features:
    - Exact command grammar for meal replacement and serving adjustments.
    - Canned scenario dictionary for explicit benchmark tasks.
    - Explicit semantic test fixtures (nonexistent recipe, allergen conflict,
      budget overflow, stale version).
    - Strict plan-version extraction and validation.
    - Deterministic tag + meal-slot recipe selection for Conditions A & B.
    - Progressive inspection integration for Condition C.

    Metadata Invariant:
    All metadata mappings produced across all MockAgent execution paths
    strictly contain immutable scalar values (str, int, float, bool, None).
    No mutable nested containers (dicts, lists, sets) are emitted in metadata.
    This guarantees zero unintended state leakages or shared references across
    evaluations.
    """

    # Grammar patterns for deterministic command parsing
    _RE_REPLACE_EXACT = re.compile(
        r"^replace\s+(?P<day>[A-Za-z]+)\s+(?P<slot>breakfast|lunch|dinner)\s+with\s+(?P<recipe_id>rec_[A-Za-z0-9_]+)(?:\s+servings\s+(?P<servings>\d+))?$",
        re.IGNORECASE,
    )
    _RE_CHANGE_SERVINGS = re.compile(
        r"^(?:change|set|scale)\s+servings\s+(?:for\s+)?(?P<day>[A-Za-z]+)\s+(?P<slot>breakfast|lunch|dinner)\s+to\s+(?P<servings>\d+)$",
        re.IGNORECASE,
    )
    _RE_REPLACE_TAG = re.compile(
        r"^replace\s+(?P<day>[A-Za-z]+)\s+(?P<slot>breakfast|lunch|dinner)\s+with\s+(?P<tag>[A-Za-z0-9_-]+)\s+recipe$",
        re.IGNORECASE,
    )

    def __init__(self, canned_tasks: Optional[Dict[str, ActionProposal]] = None) -> None:
        """Initialize MockAgent with optional canned task mappings."""
        self.canned_tasks: Dict[str, ActionProposal] = dict(canned_tasks) if canned_tasks else {}

    def propose_action(
        self,
        context: ContextResult,
        inspector: Optional[ObjectInspector] = None,
    ) -> AgentResponse:
        """Formulate an ActionProposal given a standardized ContextResult."""
        task = context.task_prompt.strip()

        # ---------------------------------------------------------------------
        # 1. Strict plan_version validation
        # ---------------------------------------------------------------------
        plan_version = _extract_plan_version(context.serialized_context)
        if plan_version is None:
            return AgentResponse(
                proposal=None,
                rationale="Invalid or missing plan_version in context.",
                raw_output="",
                metadata={
                    "status": "invalid_plan_version",
                    "condition_id": context.condition_id,
                },
            )

        # ---------------------------------------------------------------------
        # 2. Canned task matching
        # ---------------------------------------------------------------------
        if task in self.canned_tasks:
            base_prop = self.canned_tasks[task]
            # All ordinary canned/custom proposals dynamically receive plan_version from context.
            # Only TEST_STALE_VERSION (or tasks ending with stale_version) may use a mismatched version.
            if task == "TEST_STALE_VERSION" or task.endswith("stale_version"):
                exp_version = (
                    base_prop.expected_plan_version
                    if base_prop.expected_plan_version is not None
                    else plan_version + 999
                )
            else:
                exp_version = plan_version

            proposal = ActionProposal(
                operation=base_prop.operation,
                arguments=dict(base_prop.arguments),
                reason=base_prop.reason,
                expected_plan_version=exp_version,
            )
            return AgentResponse(
                proposal=proposal,
                rationale=f"Canned task match: {proposal.reason}",
                raw_output=proposal.model_dump_json(),
                inspection_calls_made=0,
                metadata={
                    "match_type": "canned_task",
                    "condition_id": context.condition_id,
                    "plan_version": exp_version,
                },
            )

        # ---------------------------------------------------------------------
        # 3. Explicit semantic test fixtures
        # ---------------------------------------------------------------------
        # Fixture 1: Nonexistent recipe test
        if task == "TEST_NONEXISTENT_RECIPE":
            prop = ActionProposal(
                operation=OperationType.REPLACE_MEAL,
                arguments={
                    "day": "Wednesday",
                    "meal_type": "dinner",
                    "recipe_id": "rec_nonexistent",
                },
                reason="Test rejection of nonexistent recipe ID.",
                expected_plan_version=plan_version,
            )
            return AgentResponse(
                proposal=prop,
                rationale="Proposed nonexistent recipe to test broker catalog rejection.",
                raw_output=prop.model_dump_json(),
                metadata={"test_fixture": "TEST_NONEXISTENT_RECIPE"},
            )

        # Fixture 2: Allergen conflict test (real catalog recipe: rec_peanut_butter_banana_toast)
        if task == "TEST_ALLERGEN_CONFLICT":
            prop = ActionProposal(
                operation=OperationType.REPLACE_MEAL,
                arguments={
                    "day": "Wednesday",
                    "meal_type": "breakfast",
                    "recipe_id": "rec_peanut_butter_banana_toast",
                },
                reason="Test rejection of allergen-conflicting recipe.",
                expected_plan_version=plan_version,
            )
            return AgentResponse(
                proposal=prop,
                rationale="Proposed peanut recipe to test PolicyEngine allergen rejection.",
                raw_output=prop.model_dump_json(),
                metadata={"test_fixture": "TEST_ALLERGEN_CONFLICT"},
            )

        # Fixture 3: Budget overflow test (real catalog recipe: rec_pan_seared_salmon costing $14.50)
        if task == "TEST_BUDGET_OVERFLOW":
            prop = ActionProposal(
                operation=OperationType.REPLACE_MEAL,
                arguments={
                    "day": "Wednesday",
                    "meal_type": "dinner",
                    "recipe_id": "rec_pan_seared_salmon",
                },
                reason="Test rejection of budget-overflow recipe.",
                expected_plan_version=plan_version,
            )
            return AgentResponse(
                proposal=prop,
                rationale="Proposed expensive recipe to test PolicyEngine budget rejection.",
                raw_output=prop.model_dump_json(),
                metadata={"test_fixture": "TEST_BUDGET_OVERFLOW"},
            )

        # Fixture 4: Stale plan version test
        if task == "TEST_STALE_VERSION" or task.endswith("stale_version"):
            prop = ActionProposal(
                operation=OperationType.REPLACE_MEAL,
                arguments={
                    "day": "Wednesday",
                    "meal_type": "dinner",
                    "recipe_id": "rec_red_lentil_dahl",
                },
                reason="Test rejection of stale plan version.",
                expected_plan_version=plan_version + 999,
            )
            return AgentResponse(
                proposal=prop,
                rationale="Proposed action with stale expected_plan_version to test broker interception.",
                raw_output=prop.model_dump_json(),
                metadata={"test_fixture": "TEST_STALE_VERSION"},
            )

        # ---------------------------------------------------------------------
        # 4. Command Grammar: Change Servings
        # ---------------------------------------------------------------------
        m_change = self._RE_CHANGE_SERVINGS.match(task)
        if m_change:
            day_str = m_change.group("day").strip().capitalize()
            slot_str = m_change.group("slot").strip().lower()
            servings_val = int(m_change.group("servings"))

            if day_str not in DAYS_OF_WEEK or slot_str not in MEAL_TYPES:
                return AgentResponse(
                    proposal=None,
                    rationale=f"Invalid schedule coordinates: {day_str} {slot_str}.",
                    metadata={"status": "invalid_coordinates"},
                )

            prop = ActionProposal(
                operation=OperationType.CHANGE_SERVINGS,
                arguments={
                    "day": day_str,
                    "meal_type": slot_str,
                    "servings": servings_val,
                },
                reason=f"Scale servings for {day_str} {slot_str} to {servings_val}.",
                expected_plan_version=plan_version,
            )
            return AgentResponse(
                proposal=prop,
                rationale=prop.reason,
                raw_output=prop.model_dump_json(),
                inspection_calls_made=0,
                metadata={"grammar": "change_servings", "plan_version": plan_version},
            )

        # ---------------------------------------------------------------------
        # 5. Command Grammar: Replace Meal by exact ID
        # ---------------------------------------------------------------------
        m_replace_exact = self._RE_REPLACE_EXACT.match(task)
        if m_replace_exact:
            day_str = m_replace_exact.group("day").strip().capitalize()
            slot_str = m_replace_exact.group("slot").strip().lower()
            rec_id = m_replace_exact.group("recipe_id").strip()
            servings_str = m_replace_exact.group("servings")
            servings_val = int(servings_str) if servings_str else None

            if day_str not in DAYS_OF_WEEK or slot_str not in MEAL_TYPES:
                return AgentResponse(
                    proposal=None,
                    rationale=f"Invalid schedule coordinates: {day_str} {slot_str}.",
                    metadata={"status": "invalid_coordinates"},
                )

            inspection_count = 0
            # Condition C: Use ObjectInspector to progressively inspect the exact recipe
            if context.condition_id == "C" and inspector is not None:
                inspector.inspect_recipe(rec_id)
                inspection_count = 1

            arguments: Dict[str, Any] = {
                "day": day_str,
                "meal_type": slot_str,
                "recipe_id": rec_id,
            }
            if servings_val is not None:
                arguments["servings"] = servings_val

            prop = ActionProposal(
                operation=OperationType.REPLACE_MEAL,
                arguments=arguments,
                reason=f"Replace {day_str} {slot_str} with {rec_id}.",
                expected_plan_version=plan_version,
            )
            return AgentResponse(
                proposal=prop,
                rationale=prop.reason,
                raw_output=prop.model_dump_json(),
                inspection_calls_made=inspection_count,
                metadata={
                    "grammar": "replace_exact",
                    "condition_id": context.condition_id,
                    "plan_version": plan_version,
                },
            )

        # ---------------------------------------------------------------------
        # 6. Command Grammar: Replace Meal by Tag + Slot category (A & B only)
        # ---------------------------------------------------------------------
        m_replace_tag = self._RE_REPLACE_TAG.match(task)
        if m_replace_tag:
            day_str = m_replace_tag.group("day").strip().capitalize()
            slot_str = m_replace_tag.group("slot").strip().lower()
            tag_str = m_replace_tag.group("tag").strip().lower()

            if day_str not in DAYS_OF_WEEK or slot_str not in MEAL_TYPES:
                return AgentResponse(
                    proposal=None,
                    rationale=f"Invalid schedule coordinates: {day_str} {slot_str}.",
                    metadata={"status": "invalid_coordinates"},
                )

            # Condition C check: Initial context does not contain available_recipes,
            # and ObjectInspector does not provide catalog enumeration/search.
            if context.condition_id == "C":
                return AgentResponse(
                    proposal=None,
                    rationale=(
                        "Condition C requires exact recipe_id or canned fixture for "
                        "progressive inspection (catalog search is not exposed)."
                    ),
                    metadata={
                        "status": "unsupported_in_condition_c",
                        "condition_id": "C",
                    },
                )

            # In Conditions A and B: Parse available_recipes
            try:
                parsed = json.loads(context.serialized_context)
                recipes = parsed.get("available_recipes", [])
            except (json.JSONDecodeError, TypeError):
                recipes = []

            # Filter by BOTH the requested dietary tag AND the meal slot category
            matching: List[Dict[str, Any]] = [
                r for r in recipes
                if tag_str in [t.lower() for t in r.get("tags", [])]
                and slot_str in [t.lower() for t in r.get("tags", [])]
            ]

            if not matching:
                return AgentResponse(
                    proposal=None,
                    rationale=f"No recipes found matching tag '{tag_str}' and meal slot '{slot_str}'.",
                    metadata={"status": "no_matching_recipes"},
                )

            # Deterministic tie-breaking: sort alphabetically by recipe_id
            matching.sort(key=lambda r: r.get("recipe_id", ""))
            chosen_recipe = matching[0]
            chosen_id = chosen_recipe["recipe_id"]

            prop = ActionProposal(
                operation=OperationType.REPLACE_MEAL,
                arguments={
                    "day": day_str,
                    "meal_type": slot_str,
                    "recipe_id": chosen_id,
                },
                reason=f"Replace {day_str} {slot_str} with {tag_str} recipe {chosen_id}.",
                expected_plan_version=plan_version,
            )
            return AgentResponse(
                proposal=prop,
                rationale=prop.reason,
                raw_output=prop.model_dump_json(),
                inspection_calls_made=0,
                metadata={
                    "grammar": "replace_tag",
                    "chosen_recipe_id": chosen_id,
                    "condition_id": context.condition_id,
                    "plan_version": plan_version,
                },
            )

        # ---------------------------------------------------------------------
        # 7. Unsupported / Unrecognized Task
        # ---------------------------------------------------------------------
        return AgentResponse(
            proposal=None,
            rationale=f"Unsupported task: unrecognized command format '{task}'.",
            raw_output="",
            inspection_calls_made=0,
            metadata={"status": "unsupported_task", "task": task},
        )
