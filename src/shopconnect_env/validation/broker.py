"""Action Broker for the ShopConnect Object-Centric Intelligence Environment.

Coordinates transaction staging, policy validation, atomic commits, and rollbacks.
Ensures the authoritative application state is updated only if all preconditions,
mutations, derived calculations, and postconditions succeed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import time
from typing import Any, Dict, Optional, Union

from pydantic import ValidationError

from shopconnect_env.domain.models import (
    GroceryBasket,
    Ingredient,
    MealPlan,
    Recipe,
    make_slot_key,
)
from shopconnect_env.operations.basket_calculator import calculate_grocery_basket
from shopconnect_env.operations.mutations import (
    apply_change_servings,
    apply_replace_meal,
)
from shopconnect_env.validation.policy_engine import PolicyDecision, PolicyEngine
from shopconnect_env.validation.schemas import (
    ActionProposal,
    ActionResult,
    ActionStatus,
    ChangeServingsArgs,
    OperationType,
    ReplaceMealArgs,
)


@dataclass(frozen=True)
class RuntimeState:
    """Authoritative snapshot of the live application state."""

    plan: MealPlan
    basket: GroceryBasket


class ActionBroker:
    """Transactional coordinator managing state transitions and candidate staging.

    Principles:
        - The LLM proposes actions; the ActionBroker validates and executes them.
        - The live state is private and immutable from the outside.
        - Changes are staged on an isolated candidate copy.
        - Commits are atomic: MealPlan and GroceryBasket advance together.
        - Failed operations result in immediate rollback with zero state mutation.
    """

    def __init__(
        self,
        initial_plan: MealPlan,
        catalog: Dict[str, Recipe],
        ingredients: Dict[str, Ingredient],
        policy_engine: Optional[PolicyEngine] = None,
    ) -> None:
        # Deep-copy catalog and ingredients to protect against external caller mutations
        self.catalog = copy.deepcopy(catalog)
        self.ingredients = copy.deepcopy(ingredients)
        self.policy_engine = policy_engine or PolicyEngine(
            catalog=self.catalog,
            ingredients=self.ingredients,
        )

        # Recalculate initial baseline basket to ensure perfect sync
        initial_basket = calculate_grocery_basket(
            initial_plan,
            catalog=self.catalog,
            ingredients=self.ingredients,
        )

        # Authoritative internal state (private)
        self._state = RuntimeState(
            plan=copy.deepcopy(initial_plan),
            basket=initial_basket,
        )

    def get_plan(self) -> MealPlan:
        """Return a safe, isolated deep copy of the authoritative MealPlan.

        Callers cannot mutate the broker's authoritative internal state.
        """
        return copy.deepcopy(self._state.plan)

    def get_basket(self) -> GroceryBasket:
        """Return a safe snapshot of the authoritative GroceryBasket."""
        return copy.deepcopy(self._state.basket)

    def get_plan_version(self) -> int:
        """Return the current authoritative plan version."""
        return self._state.plan.plan_version

    def get_state_snapshot(self) -> RuntimeState:
        """Return an isolated copy of the full runtime state."""
        return RuntimeState(
            plan=copy.deepcopy(self._state.plan),
            basket=copy.deepcopy(self._state.basket),
        )

    def execute(
        self,
        proposal_input: Union[ActionProposal, Dict[str, Any], str],
    ) -> ActionResult:
        """Process an action proposal through staging, validation, and atomic commit.

        Args:
            proposal_input: ActionProposal instance, raw Python dictionary, or JSON string.

        Returns:
            An ActionResult with status COMMITTED or REJECTED.
        """
        start_time = time.perf_counter()
        version_before = self._state.plan.plan_version

        # -------------------------------------------------------------
        # STEP 1: Parse and validate input into a typed ActionProposal
        # -------------------------------------------------------------
        try:
            if isinstance(proposal_input, str):
                proposal = ActionProposal.from_json(proposal_input)
            elif isinstance(proposal_input, dict):
                proposal = ActionProposal.from_dict(proposal_input)
            elif isinstance(proposal_input, ActionProposal):
                proposal = proposal_input
            else:
                return self._reject(
                    operation="unknown",
                    version_before=version_before,
                    message=f"Unsupported proposal input type: {type(proposal_input).__name__}.",
                    rejection_reason="Invalid proposal input format.",
                    start_time=start_time,
                )
        except (ValidationError, json.JSONDecodeError, ValueError) as err:
            return self._reject(
                operation="unknown",
                version_before=version_before,
                message=f"Schema validation error: {err}",
                rejection_reason=str(err),
                start_time=start_time,
            )

        operation_name = proposal.operation.value

        # -------------------------------------------------------------
        # STEP 2: Verify operation allowlist for state mutations
        # -------------------------------------------------------------
        if proposal.operation == OperationType.INSPECT_OBJECT:
            return self._reject(
                operation=operation_name,
                version_before=version_before,
                message="inspect_object is a read-only query and does not perform state mutations.",
                rejection_reason="Mutation broker rejected read-only action.",
                start_time=start_time,
            )

        # -------------------------------------------------------------
        # STEP 3: Check expected_plan_version (Stale-State Protection)
        # -------------------------------------------------------------
        if (
            proposal.expected_plan_version is not None
            and proposal.expected_plan_version != version_before
        ):
            return self._reject(
                operation=operation_name,
                version_before=version_before,
                message=(
                    f"Stale plan version detected: proposal expected version {proposal.expected_plan_version}, "
                    f"but current authoritative plan version is {version_before}."
                ),
                rejection_reason="Stale plan version.",
                start_time=start_time,
                details={
                    "expected_version": proposal.expected_plan_version,
                    "current_version": version_before,
                },
            )

        # -------------------------------------------------------------
        # STEP 4: Candidate Staging via Deterministic Operations
        # -------------------------------------------------------------
        target_slot_key: Optional[str] = None
        try:
            if proposal.operation == OperationType.REPLACE_MEAL:
                args: ReplaceMealArgs = ReplaceMealArgs.model_validate(proposal.arguments)
                target_slot_key = make_slot_key(args.day, args.meal_type)
                candidate_plan = apply_replace_meal(
                    plan=self._state.plan,
                    day=args.day,
                    meal_type=args.meal_type,
                    recipe_id=args.recipe_id,
                    catalog=self.catalog,
                    servings=args.servings,
                )
            elif proposal.operation == OperationType.CHANGE_SERVINGS:
                args: ChangeServingsArgs = ChangeServingsArgs.model_validate(proposal.arguments)
                target_slot_key = make_slot_key(args.day, args.meal_type)
                candidate_plan = apply_change_servings(
                    plan=self._state.plan,
                    day=args.day,
                    meal_type=args.meal_type,
                    servings=args.servings,
                )
            else:
                return self._reject(
                    operation=operation_name,
                    version_before=version_before,
                    message=f"Unsupported operation: '{proposal.operation}'.",
                    rejection_reason="Unsupported operation.",
                    start_time=start_time,
                )
        except (KeyError, ValueError) as err:
            return self._reject(
                operation=operation_name,
                version_before=version_before,
                message=f"Deterministic operation failed: {err}",
                rejection_reason=str(err),
                start_time=start_time,
            )

        # -------------------------------------------------------------
        # STEP 5: Policy Engine Pre-Commit & Invariant Validation
        # -------------------------------------------------------------
        decision: PolicyDecision = self.policy_engine.validate_candidate(
            original_plan=self._state.plan,
            candidate_plan=candidate_plan,
            target_slot_key=target_slot_key,
        )

        if not decision.allowed:
            return self._reject(
                operation=operation_name,
                version_before=version_before,
                message=f"Policy rejection ({decision.rule_name}): {decision.reason}",
                rejection_reason=decision.reason,
                start_time=start_time,
                details=decision.details,
            )

        # -------------------------------------------------------------
        # STEP 6: Recalculate Derived GroceryBasket from Candidate
        # -------------------------------------------------------------
        try:
            candidate_basket = calculate_grocery_basket(
                plan=candidate_plan,
                catalog=self.catalog,
                ingredients=self.ingredients,
            )
        except Exception as err:
            return self._reject(
                operation=operation_name,
                version_before=version_before,
                message=f"Derived grocery basket calculation failed: {err}",
                rejection_reason=str(err),
                start_time=start_time,
            )

        # -------------------------------------------------------------
        # STEP 7: Verify Postconditions & Invariants
        # -------------------------------------------------------------
        if candidate_plan.plan_version != version_before + 1:
            return self._reject(
                operation=operation_name,
                version_before=version_before,
                message=(
                    f"Postcondition invariant broken: candidate plan version ({candidate_plan.plan_version}) "
                    f"must be exactly {version_before + 1}."
                ),
                rejection_reason="Plan version increment postcondition failed.",
                start_time=start_time,
            )

        if candidate_basket.basket_version != candidate_plan.plan_version:
            return self._reject(
                operation=operation_name,
                version_before=version_before,
                message=(
                    f"Postcondition invariant broken: basket version ({candidate_basket.basket_version}) "
                    f"does not match candidate plan version ({candidate_plan.plan_version})."
                ),
                rejection_reason="Basket version mismatch postcondition.",
                start_time=start_time,
            )

        # -------------------------------------------------------------
        # STEP 8: Atomic Commit
        # -------------------------------------------------------------
        # State swap occurs in a single assignment after all checks pass
        self._state = RuntimeState(
            plan=candidate_plan,
            basket=candidate_basket,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return ActionResult(
            status=ActionStatus.COMMITTED,
            operation=operation_name,
            plan_version_before=version_before,
            plan_version_after=self._state.plan.plan_version,
            message=f"Successfully executed and committed operation '{operation_name}'.",
            execution_duration_ms=round(duration_ms, 3),
            rejection_reason=None,
            details={
                "target_slot_key": target_slot_key,
                "reason": proposal.reason,
                "basket_item_count": len(self._state.basket.items),
            },
        )

    def _reject(
        self,
        operation: str,
        version_before: int,
        message: str,
        rejection_reason: str,
        start_time: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        """Create a rejected ActionResult with precise timing, preserving state."""
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return ActionResult(
            status=ActionStatus.REJECTED,
            operation=operation,
            plan_version_before=version_before,
            plan_version_after=version_before,  # Never increments on rejection
            message=message,
            rejection_reason=rejection_reason,
            execution_duration_ms=round(duration_ms, 3),
            details=copy.deepcopy(details) if details is not None else None,
        )
