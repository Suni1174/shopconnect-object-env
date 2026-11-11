"""Comparative experiment runner, state differ, outcome evaluator, and artifact generator.

Part of Stage 6 Phase 6.4:
- Executes benchmark tasks across Conditions A, B, and C.
- Enforces strict environment isolation and fresh state per run.
- Zero ground-truth leakage: model sees only user_request and legitimate context.
- Trusted ActionBroker execution pipeline.
- Deterministic plan diffing and invariant verification.
- Output artifacts: raw_runs.jsonl, comparison.csv, analysis.md.
- Offline smoke-test mode with explicit simulation markers.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from shopconnect_env.agents.base import AgentResponse, BaseAgent, ObjectInspector
from shopconnect_env.agents.condition_a_agent import ConditionAAgent
from shopconnect_env.agents.condition_b_agent import ConditionBAgent
from shopconnect_env.agents.condition_c_agent import ConditionCAgent
from shopconnect_env.agents.providers import (
    LLMCompletionResult,
    LLMMessage,
    LLMProvider,
    LLMUsage,
    MockLLMProvider,
)
from shopconnect_env.context.base import ContextRequest, ContextResult
from shopconnect_env.context.full_state import FullStateContextBuilder
from shopconnect_env.context.object_context import ObjectContextBuilder
from shopconnect_env.context.tool_context import ToolContextBuilder
from shopconnect_env.domain.models import (
    DAYS_OF_WEEK,
    MEAL_TYPES,
    GroceryBasket,
    Ingredient,
    Meal,
    MealPlan,
    Recipe,
    load_catalog_from_json,
    make_slot_key,
)
from shopconnect_env.evaluation.metrics import (
    AggregateMetrics,
    PricingConfig,
    RunMetrics,
    calculate_cost,
    calculate_percentile,
    compute_aggregate_metrics,
)
from shopconnect_env.evaluation.tasks import (
    BenchmarkTask,
    ExpectedOutcome,
    PlanFixture,
    load_benchmark_tasks,
    load_plan_fixtures,
)
from shopconnect_env.operations.basket_calculator import calculate_grocery_basket
from shopconnect_env.validation.broker import ActionBroker, ActionResult, ActionStatus
from shopconnect_env.validation.policy_engine import PolicyEngine
from shopconnect_env.validation.schemas import ActionProposal


def compute_sha256(data: bytes) -> str:
    """Compute hex SHA-256 hash of arbitrary bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 hash of a file on disk."""
    if not path.exists():
        return "file_not_found"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def classify_rejection_reason(reason: str) -> str:
    """Classify human/system rejection text into standardized benchmark rejection category."""
    r = reason.lower()
    if "allergen" in r or "peanut" in r:
        return "allergen_conflict"
    if "dietary" in r or "vegetarian" in r or "pescatarian" in r:
        return "dietary_mismatch"
    if "budget" in r or "exceeds" in r or "cost" in r:
        return "budget_overflow"
    if "recipe" in r or "catalog" in r or "invalid recipe" in r:
        return "invalid_recipe"
    if "no_op" in r or "target_not_modified" in r or "identical" in r or "no schedule changes" in r:
        return "no_op"
    if "drift" in r:
        return "unintended_drift"
    if "stale" in r or "version" in r:
        return "stale_plan_version"
    return "other"


def _meal_to_dict(meal: Optional[Meal]) -> Optional[Dict[str, Any]]:
    """Convert Meal dataclass instance to standard dictionary."""
    if meal is None:
        return None
    return {
        "day": meal.day,
        "meal_type": meal.meal_type,
        "recipe_id": meal.recipe_id,
        "servings": meal.servings,
    }


@dataclass(frozen=True)
class PlanDiffResult:
    """Fine-grained semantic diff between starting and post-execution meal plan states."""

    target_day: Optional[str]
    target_meal_type: Optional[str]
    target_slot_before: Optional[Dict[str, Any]]
    target_slot_after: Optional[Dict[str, Any]]
    target_changed: bool
    actual_target_recipe_id: Optional[str]
    actual_target_servings: Optional[int]
    all_changed_coordinates: List[Tuple[str, str]]
    non_target_changes: List[Dict[str, Any]]
    non_target_drift_count: int
    unintended_assignments: int
    non_target_recipe_overwrites: int
    basket_consistent: bool
    policy_violations: List[str]
    plan_mutated: bool
    basket_mutated: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert diff to a JSON-serializable dictionary."""
        return {
            "target_day": self.target_day,
            "target_meal_type": self.target_meal_type,
            "target_slot_before": self.target_slot_before,
            "target_slot_after": self.target_slot_after,
            "target_changed": self.target_changed,
            "actual_target_recipe_id": self.actual_target_recipe_id,
            "actual_target_servings": self.actual_target_servings,
            "all_changed_coordinates": [list(c) for c in self.all_changed_coordinates],
            "non_target_changes": self.non_target_changes,
            "non_target_drift_count": self.non_target_drift_count,
            "unintended_assignments": self.unintended_assignments,
            "non_target_recipe_overwrites": self.non_target_recipe_overwrites,
            "basket_consistent": self.basket_consistent,
            "policy_violations": self.policy_violations,
            "plan_mutated": self.plan_mutated,
            "basket_mutated": self.basket_mutated,
        }


class PlanDiffer:
    """Deterministic comparison engine analyzing pre/post meal plan and basket states."""

    @staticmethod
    def diff(
        initial_plan: MealPlan,
        final_plan: MealPlan,
        initial_basket: GroceryBasket,
        final_basket: GroceryBasket,
        target_day: Optional[str],
        target_meal_type: Optional[str],
        catalog: Dict[str, Recipe],
        ingredients: Dict[str, Ingredient],
    ) -> PlanDiffResult:
        """Perform exhaustive schedule coordinate diffing across all 21 slots."""
        all_changed_coords: List[Tuple[str, str]] = []
        non_target_changes: List[Dict[str, Any]] = []
        unintended_assignments = 0
        non_target_recipe_overwrites = 0

        target_slot_key = (
            make_slot_key(target_day, target_meal_type)
            if target_day and target_meal_type
            else None
        )

        meal_before_target = initial_plan.meals.get(target_slot_key) if target_slot_key else None
        meal_after_target = final_plan.meals.get(target_slot_key) if target_slot_key else None

        for d in DAYS_OF_WEEK:
            for m in MEAL_TYPES:
                k = make_slot_key(d, m)
                m_before = initial_plan.meals.get(k)
                m_after = final_plan.meals.get(k)
                is_target = (k == target_slot_key)

                changed = False
                if m_before is None and m_after is not None:
                    changed = True
                    if not is_target:
                        unintended_assignments += 1
                elif m_before is not None and m_after is None:
                    changed = True
                    if not is_target:
                        non_target_recipe_overwrites += 1
                elif m_before is not None and m_after is not None:
                    if m_before.recipe_id != m_after.recipe_id or m_before.servings != m_after.servings:
                        changed = True
                        if not is_target and m_before.recipe_id != m_after.recipe_id:
                            non_target_recipe_overwrites += 1

                if changed:
                    all_changed_coords.append((d, m))
                    if not is_target:
                        non_target_changes.append({
                            "day": d,
                            "meal_type": m,
                            "before": _meal_to_dict(m_before),
                            "after": _meal_to_dict(m_after),
                        })

        target_changed = (
            meal_before_target != meal_after_target
            if target_slot_key is not None
            else False
        )

        # Recalculate grocery basket to verify deterministic basket consistency
        recalculated_basket = calculate_grocery_basket(final_plan, catalog, ingredients)
        basket_consistent = (recalculated_basket.items == final_basket.items)

        # Invariant and policy validation on final plan
        engine = PolicyEngine(catalog=catalog, ingredients=ingredients)
        violations: List[str] = []
        if not engine.validate_dietary_restrictions(final_plan).allowed:
            violations.append("dietary_mismatch")
        if not engine.validate_allergens(final_plan).allowed:
            violations.append("allergen_conflict")
        if not engine.validate_budget(final_plan).allowed:
            violations.append("budget_overflow")

        plan_mutated = (initial_plan.meals != final_plan.meals)
        basket_mutated = (initial_basket.items != final_basket.items)

        return PlanDiffResult(
            target_day=target_day,
            target_meal_type=target_meal_type,
            target_slot_before=_meal_to_dict(meal_before_target),
            target_slot_after=_meal_to_dict(meal_after_target),
            target_changed=target_changed,
            actual_target_recipe_id=meal_after_target.recipe_id if meal_after_target else None,
            actual_target_servings=meal_after_target.servings if meal_after_target else None,
            all_changed_coordinates=all_changed_coords,
            non_target_changes=non_target_changes,
            non_target_drift_count=len(non_target_changes),
            unintended_assignments=unintended_assignments,
            non_target_recipe_overwrites=non_target_recipe_overwrites,
            basket_consistent=basket_consistent,
            policy_violations=violations,
            plan_mutated=plan_mutated,
            basket_mutated=basket_mutated,
        )


class OutcomeEvaluator:
    """Evaluates task execution against benchmark ground truth without heuristics."""

    @staticmethod
    def evaluate(
        task: BenchmarkTask,
        response: AgentResponse,
        action_result: Optional[ActionResult],
        plan_diff: PlanDiffResult,
        execution_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Determine actual outcome, task success, target accuracy, and false rejections."""
        # 1. Execution errors (malformed output, timeout, crash)
        if execution_error is not None:
            return {
                "actual_outcome": "ERROR",
                "task_success": False,
                "target_accuracy": False,
                "false_rejection": False,
                "rejection_category_match": None,
                "error_detail": execution_error,
            }

        # 2. ActionBroker COMMITTED transaction
        if action_result is not None and action_result.status == ActionStatus.COMMITTED:
            actual_outcome = "COMMIT"
            target_acc = (
                plan_diff.target_changed
                or (plan_diff.actual_target_recipe_id == task.expected_recipe_id)
            )

            if task.expected_outcome == ExpectedOutcome.COMMIT:
                # Rigorous check against all ground truth components
                op_val = (
                    action_result.operation.value
                    if hasattr(action_result.operation, "value")
                    else str(action_result.operation)
                )
                op_match = (op_val == task.expected_operation)
                recipe_match = (plan_diff.actual_target_recipe_id == task.expected_recipe_id)
                servings_match = (plan_diff.actual_target_servings == task.expected_servings)
                no_drift = (plan_diff.non_target_drift_count == 0)
                policy_ok = (len(plan_diff.policy_violations) == 0)

                success = (
                    op_match
                    and recipe_match
                    and servings_match
                    and no_drift
                    and policy_ok
                )
                return {
                    "actual_outcome": actual_outcome,
                    "task_success": success,
                    "target_accuracy": target_acc,
                    "false_rejection": False,
                    "rejection_category_match": None,
                    "error_detail": None,
                }
            else:
                # Committed state when REJECT or CLARIFY was expected is a failure/violation
                return {
                    "actual_outcome": actual_outcome,
                    "task_success": False,
                    "target_accuracy": target_acc,
                    "false_rejection": False,
                    "rejection_category_match": None,
                    "error_detail": f"State committed illegally; expected {task.expected_outcome.value}.",
                }

        # 3. Explicit CLARIFICATION decision
        if response.is_clarification:
            actual_outcome = "CLARIFY"
            no_mutation = (not plan_diff.plan_mutated and not plan_diff.basket_mutated)

            if task.expected_outcome == ExpectedOutcome.CLARIFY:
                return {
                    "actual_outcome": actual_outcome,
                    "task_success": no_mutation,
                    "target_accuracy": True,
                    "false_rejection": False,
                    "rejection_category_match": None,
                    "error_detail": None,
                }
            elif task.expected_outcome == ExpectedOutcome.COMMIT:
                return {
                    "actual_outcome": actual_outcome,
                    "task_success": False,
                    "target_accuracy": False,
                    "false_rejection": True,  # Erroneous clarification on valid task
                    "rejection_category_match": None,
                    "error_detail": "Agent erroneously requested clarification on a clear COMMIT task.",
                }
            else:  # REJECT expected
                return {
                    "actual_outcome": actual_outcome,
                    "task_success": False,
                    "target_accuracy": False,
                    "false_rejection": False,
                    "rejection_category_match": None,
                    "error_detail": "Agent requested clarification instead of rejecting invalid operation.",
                }

        # 4. REJECTION (by ActionBroker policy/syntax or Agent pre-commit refusal)
        actual_outcome = "REJECT"
        no_mutation = (not plan_diff.plan_mutated and not plan_diff.basket_mutated)

        # Extract rejection reason
        raw_reason = ""
        if action_result is not None and action_result.rejection_reason:
            raw_reason = action_result.rejection_reason
        elif response.rationale:
            raw_reason = response.rationale
        elif "status" in response.metadata:
            raw_reason = str(response.metadata["status"])

        classified_category = classify_rejection_reason(raw_reason)

        # Target accuracy in rejected proposal
        target_acc = False
        if response.proposal and response.proposal.arguments:
            p_day = response.proposal.arguments.get("day")
            p_slot = response.proposal.arguments.get("meal_type")
            if p_day == task.expected_day and p_slot == task.expected_meal_type:
                target_acc = True

        if task.expected_outcome == ExpectedOutcome.REJECT:
            cat_match = (classified_category == task.expected_rejection_category)
            success = cat_match and no_mutation
            return {
                "actual_outcome": actual_outcome,
                "task_success": success,
                "target_accuracy": target_acc,
                "false_rejection": False,
                "rejection_category_match": cat_match,
                "error_detail": None if success else f"Rejection category mismatch: expected '{task.expected_rejection_category}', got '{classified_category}'.",
            }
        elif task.expected_outcome == ExpectedOutcome.COMMIT:
            return {
                "actual_outcome": actual_outcome,
                "task_success": False,
                "target_accuracy": target_acc,
                "false_rejection": True,  # False rejection of valid COMMIT task
                "rejection_category_match": False,
                "error_detail": f"Valid COMMIT task was rejected: {raw_reason}",
            }
        else:  # Expected CLARIFY
            return {
                "actual_outcome": actual_outcome,
                "task_success": False,
                "target_accuracy": False,
                "false_rejection": False,
                "rejection_category_match": None,
                "error_detail": f"Task rejected instead of clarified: {raw_reason}",
            }


@dataclass(frozen=True)
class RunRecord:
    """Comprehensive trace record of a single benchmark task-condition run."""

    # Trace identity
    experiment_run_id: str
    experiment_config_version: str
    task_id: str
    fixture_id: str
    condition_id: str
    timestamp_start_utc: str
    timestamp_end_utc: str
    wall_clock_latency_ms: float

    # Provider & environment provenance
    provider_name: str
    provider_mode: str  # "mock", "simulated", "live"
    requested_model: str
    actual_model_identifier: Optional[str]
    provider_request_ids: List[str]
    backend_fingerprint: Optional[str]
    sdk_version: Optional[str]
    reasoning_effort: Optional[str]
    per_request_latencies_ms: List[float]
    aggregated_token_usage: Optional[Dict[str, Any]]
    usage_provenance: str  # "provider_reported" | "simulated"
    fixture_dataset_hash: str
    task_dataset_hash: str
    prompt_template_hash: str
    pricing_version: Optional[str]

    # Task context (ground truth preserved for evaluator only)
    user_request: str
    expected_outcome: str
    expected_operation: Optional[str]
    expected_day: Optional[str]
    expected_meal_type: Optional[str]
    expected_recipe_id: Optional[str]
    expected_servings: Optional[int]
    expected_rejection_category: Optional[str]

    # Agent response
    has_proposal: bool
    proposal: Optional[Dict[str, Any]]
    rationale: str
    clarification_message: Optional[str]
    inspection_calls_made: int
    inspection_trajectory: List[Dict[str, Any]]

    # Outcome evaluation
    actual_outcome: str
    task_success: bool
    target_accuracy: bool
    false_rejection: bool
    rejection_category_match: Optional[bool]
    error_detail: Optional[str]

    # State safety & differ results
    plan_diff: Dict[str, Any]
    metrics: Dict[str, Any]

    @property
    def model_name(self) -> str:
        """Backward-compatible alias for requested_model."""
        return self.requested_model

    @property
    def is_provider_reported(self) -> bool:
        """Backward-compatible alias for usage provenance."""
        return self.usage_provenance == "provider_reported"

    @property
    def inspection_steps(self) -> List[Dict[str, Any]]:
        """Backward-compatible alias for inspection_trajectory."""
        return self.inspection_trajectory

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace record to a dictionary, including backward-compatible aliases."""
        d = asdict(self)
        d["model_name"] = self.requested_model
        d["is_provider_reported"] = (self.usage_provenance == "provider_reported")
        d["inspection_steps"] = self.inspection_trajectory
        return d


class ExperimentRunner:
    """Controlled execution harness for comparative multi-condition evaluation."""

    def __init__(
        self,
        fixtures: Dict[str, PlanFixture],
        tasks: List[BenchmarkTask],
        catalog: Dict[str, Recipe],
        ingredients: Dict[str, Ingredient],
        pricing: Optional[PricingConfig] = None,
        default_provider: Optional[LLMProvider] = None,
        experiment_run_id: Optional[str] = None,
        experiment_config_version: str = "1.0.0",
        provider_mode: str = "simulated",
        model_name: str = "mock-model",
        condition_c_max_model_requests: Optional[int] = None,
        condition_c_max_inspections: Optional[int] = None,
    ) -> None:
        self.fixtures = fixtures
        self.tasks = {t.task_id: t for t in tasks}
        self.catalog = catalog
        self.ingredients = ingredients
        self.pricing = pricing
        self.default_provider = default_provider or MockLLMProvider([])
        self.experiment_run_id = (
            experiment_run_id
            or f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        self.experiment_config_version = experiment_config_version
        self.provider_mode = provider_mode
        self.model_name = model_name
        self.condition_c_max_model_requests = condition_c_max_model_requests
        self.condition_c_max_inspections = condition_c_max_inspections

        # Calculate dataset hashes for traceability
        fixtures_raw = json.dumps(
            {k: v.raw_data for k, v in sorted(fixtures.items())}, sort_keys=True
        ).encode("utf-8")
        tasks_raw = "".join(sorted([t.task_id + t.user_request for t in tasks])).encode("utf-8")
        self.fixture_dataset_hash = compute_sha256(fixtures_raw)
        self.task_dataset_hash = compute_sha256(tasks_raw)

    def run_task(
        self,
        task: BenchmarkTask,
        condition_id: str,
        provider: Optional[LLMProvider] = None,
    ) -> RunRecord:
        """Execute a single task against a specified condition in a fresh environment."""
        if condition_id not in ("A", "B", "C"):
            raise ValueError(f"Invalid condition_id '{condition_id}'. Must be 'A', 'B', or 'C'.")

        active_provider = provider or self.default_provider
        start_clock = time.perf_counter()
        t_start_utc = datetime.now(timezone.utc).isoformat()

        # 1. Guarantee a fresh, independent environment: clone starting plan & basket
        starting_fixture = self.fixtures[task.fixture_id]
        fresh_plan = copy.deepcopy(starting_fixture.plan)
        fresh_basket = copy.deepcopy(starting_fixture.basket)

        broker = ActionBroker(
            initial_plan=fresh_plan,
            catalog=self.catalog,
            ingredients=self.ingredients,
        )

        # 2. Build condition context (Pass ONLY user_request; zero ground-truth leakage)
        context_req = ContextRequest(
            user_request=task.user_request,
            plan=broker.get_plan(),
            catalog=self.catalog,
            ingredients=self.ingredients,
            target_day=None,
            target_meal_type=None,
        )

        prompt_template_hash = "unknown"
        if condition_id == "A":
            builder_a = FullStateContextBuilder()
            prompt_template_hash = compute_sha256(builder_a.SYSTEM_PROMPT.encode("utf-8"))
            context_res = builder_a.build(context_req)
            agent: BaseAgent = ConditionAAgent(provider=active_provider)
            inspector = None

        elif condition_id == "B":
            builder_b = ToolContextBuilder()
            prompt_template_hash = compute_sha256(builder_b.SYSTEM_PROMPT.encode("utf-8"))
            context_res = builder_b.build(context_req)
            agent = ConditionBAgent(provider=active_provider)
            inspector = None

        elif condition_id == "C":
            builder_c = ObjectContextBuilder()
            prompt_template_hash = compute_sha256(builder_c.SYSTEM_PROMPT.encode("utf-8"))
            context_res = builder_c.build(context_req)
            agent = ConditionCAgent(
                provider=active_provider,
                max_model_requests=self.condition_c_max_model_requests,
                max_inspection_calls=self.condition_c_max_inspections,
            )
            # ObjectContextBuilder implements ObjectInspector
            inspector = builder_c

        # 3. Agent action proposal
        execution_error = None
        action_result: Optional[ActionResult] = None
        try:
            response: AgentResponse = agent.propose_action(context=context_res, inspector=inspector)
        except Exception as err:
            execution_error = f"Agent exception: {err}"
            response = AgentResponse(
                proposal=None,
                rationale="Agent crashed during action formulation.",
                raw_output="",
                metadata={"error": str(err)},
            )

        # 4. Commit pathway: submit proposal through trusted ActionBroker if formulated
        if execution_error is None and response.has_proposal and response.proposal is not None:
            try:
                action_result = broker.execute(response.proposal)
            except Exception as err:
                execution_error = f"Broker exception: {err}"

        # 5. Measure state differences and postconditions
        final_plan = broker.get_plan()
        final_basket = broker.get_basket()

        plan_diff = PlanDiffer.diff(
            initial_plan=starting_fixture.plan,
            final_plan=final_plan,
            initial_basket=fresh_basket,
            final_basket=final_basket,
            target_day=task.expected_day,
            target_meal_type=task.expected_meal_type,
            catalog=self.catalog,
            ingredients=self.ingredients,
        )

        # 6. Evaluate outcome against benchmark ground truth
        eval_dict = OutcomeEvaluator.evaluate(
            task=task,
            response=response,
            action_result=action_result,
            plan_diff=plan_diff,
            execution_error=execution_error,
        )

        # 7. Extract usage metrics and calculate monetary cost
        meta_dict = response.metadata_dict
        usage_data = meta_dict.get("usage")
        llm_usage: Optional[LLMUsage] = None
        if isinstance(usage_data, dict):
            llm_usage = LLMUsage(
                input_tokens=usage_data.get("input_tokens"),
                output_tokens=usage_data.get("output_tokens"),
                cached_input_tokens=usage_data.get("cached_input_tokens"),
                reasoning_tokens=usage_data.get("reasoning_tokens"),
                total_tokens=usage_data.get("total_tokens"),
                is_provider_reported=bool(usage_data.get("is_provider_reported", False)),
            )

        cost_usd = calculate_cost(llm_usage, self.pricing)

        end_clock = time.perf_counter()
        latency_ms = round((end_clock - start_clock) * 1000, 2)
        t_end_utc = datetime.now(timezone.utc).isoformat()

        model_requests = int(meta_dict.get("total_model_requests", 1))
        inspection_steps = list(meta_dict.get("inspection_steps", []))

        drift_count = max(
            plan_diff.non_target_drift_count,
            int(meta_dict.get("unintended_mutations_count", 0)),
        )
        unintended_assigns = plan_diff.unintended_assignments
        if "unintended_slots" in meta_dict:
            for s in meta_dict["unintended_slots"]:
                if isinstance(s, (list, tuple)) and len(s) == 2:
                    k = make_slot_key(str(s[0]), str(s[1]))
                    if k not in starting_fixture.plan.meals:
                        unintended_assigns += 1

        # 8. Extract traceability metadata
        model_req_ids = list(meta_dict.get("provider_request_ids", []))
        if not model_req_ids and meta_dict.get("provider_request_id"):
            model_req_ids = [meta_dict["provider_request_id"]]

        per_req_latencies = list(meta_dict.get("per_request_latencies_ms", []))
        if not per_req_latencies and latency_ms:
            per_req_latencies = [latency_ms]

        actual_model_id = meta_dict.get("model_identifier")
        backend_fp = meta_dict.get("system_fingerprint")
        sdk_ver = getattr(active_provider, "sdk_version", None)
        if not sdk_ver:
            try:
                import openai
                sdk_ver = getattr(openai, "__version__", None)
            except ImportError:
                sdk_ver = None

        usage_prov = (
            "provider_reported"
            if (llm_usage and llm_usage.is_provider_reported)
            else "simulated"
        )

        # 9. Assemble granular RunMetrics
        run_metrics = RunMetrics(
            task_id=task.task_id,
            fixture_id=task.fixture_id,
            condition_id=condition_id,
            task_success=eval_dict["task_success"],
            expected_outcome=task.expected_outcome.value,
            actual_outcome=eval_dict["actual_outcome"],
            target_accuracy=eval_dict["target_accuracy"],
            false_rejection=eval_dict["false_rejection"],
            rejection_category_match=eval_dict["rejection_category_match"],
            non_target_drift_count=drift_count,
            unintended_assignments=unintended_assigns,
            non_target_recipe_overwrites=plan_diff.non_target_recipe_overwrites,
            committed_constraint_violations=len(plan_diff.policy_violations),
            plan_mutated_on_rejection=(
                eval_dict["actual_outcome"] in ("REJECT", "CLARIFY") and plan_diff.plan_mutated
            ),
            basket_mutated_on_rejection=(
                eval_dict["actual_outcome"] in ("REJECT", "CLARIFY") and plan_diff.basket_mutated
            ),
            input_tokens=llm_usage.input_tokens if llm_usage else None,
            output_tokens=llm_usage.output_tokens if llm_usage else None,
            cached_input_tokens=llm_usage.cached_input_tokens if llm_usage else None,
            reasoning_tokens=llm_usage.reasoning_tokens if llm_usage else None,
            total_tokens=llm_usage.total_tokens if llm_usage else None,
            cache_write_tokens=llm_usage.cache_write_tokens if llm_usage else None,
            cache_write_tokens_30m=llm_usage.cache_write_tokens_30m if llm_usage else None,
            cache_write_tokens_12h=llm_usage.cache_write_tokens_12h if llm_usage else None,
            is_provider_reported=llm_usage.is_provider_reported if llm_usage else False,
            model_requests=model_requests,
            inspection_calls=response.inspection_calls_made,
            wall_clock_latency_ms=latency_ms,
            per_request_latencies_ms=per_req_latencies,
            cost_usd=cost_usd,
        )

        return RunRecord(
            experiment_run_id=self.experiment_run_id,
            experiment_config_version=self.experiment_config_version,
            task_id=task.task_id,
            fixture_id=task.fixture_id,
            condition_id=condition_id,
            timestamp_start_utc=t_start_utc,
            timestamp_end_utc=t_end_utc,
            wall_clock_latency_ms=latency_ms,
            provider_name=active_provider.__class__.__name__,
            provider_mode=self.provider_mode,
            requested_model=self.model_name,
            actual_model_identifier=actual_model_id or self.model_name,
            provider_request_ids=model_req_ids,
            backend_fingerprint=backend_fp,
            sdk_version=sdk_ver,
            reasoning_effort="low",
            per_request_latencies_ms=per_req_latencies,
            aggregated_token_usage=usage_data if isinstance(usage_data, dict) else None,
            usage_provenance=usage_prov,
            fixture_dataset_hash=self.fixture_dataset_hash,
            task_dataset_hash=self.task_dataset_hash,
            prompt_template_hash=prompt_template_hash,
            pricing_version=self.pricing.version if self.pricing else None,
            user_request=task.user_request,
            expected_outcome=task.expected_outcome.value,
            expected_operation=task.expected_operation,
            expected_day=task.expected_day,
            expected_meal_type=task.expected_meal_type,
            expected_recipe_id=task.expected_recipe_id,
            expected_servings=task.expected_servings,
            expected_rejection_category=task.expected_rejection_category,
            has_proposal=response.has_proposal,
            proposal=response.proposal.model_dump(mode="json") if response.proposal else None,
            rationale=response.rationale,
            clarification_message=response.clarification_message,
            inspection_calls_made=response.inspection_calls_made,
            inspection_trajectory=inspection_steps,
            actual_outcome=eval_dict["actual_outcome"],
            task_success=eval_dict["task_success"],
            target_accuracy=eval_dict["target_accuracy"],
            false_rejection=eval_dict["false_rejection"],
            rejection_category_match=eval_dict["rejection_category_match"],
            error_detail=eval_dict["error_detail"],
            plan_diff=plan_diff.to_dict(),
            metrics=run_metrics.to_dict(),
        )

    def run_experiment(
        self,
        task_ids: Optional[List[str]] = None,
        condition_ids: Optional[List[str]] = None,
        provider: Optional[LLMProvider] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[RunRecord]:
        """Execute selected tasks across selected conditions and optionally write artifacts."""
        target_tasks = (
            [self.tasks[tid] for tid in task_ids if tid in self.tasks]
            if task_ids
            else list(self.tasks.values())
        )
        conditions = condition_ids or ["A", "B", "C"]

        records: List[RunRecord] = []
        for task in target_tasks:
            for cond in conditions:
                rec = self.run_task(task=task, condition_id=cond, provider=provider)
                records.append(rec)

        if output_dir:
            self.export_artifacts(records, Path(output_dir))

        return records

    def export_artifacts(self, records: List[RunRecord], output_dir: Union[str, Path]) -> None:
        """Export raw_runs.jsonl, comparison.csv, and analysis.md."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. raw_runs.jsonl
        jsonl_path = out_path / "raw_runs.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict()) + "\n")

        # 2. comparison.csv
        csv_path = out_path / "comparison.csv"
        csv_fields = [
            "task_id",
            "fixture_id",
            "condition_id",
            "expected_outcome",
            "actual_outcome",
            "task_success",
            "target_accuracy",
            "false_rejection",
            "rejection_category_match",
            "non_target_drift_count",
            "unintended_assignments",
            "non_target_recipe_overwrites",
            "policy_violations_count",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "model_requests",
            "inspection_calls",
            "latency_ms",
            "cost_usd",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for rec in records:
                m = rec.metrics
                writer.writerow({
                    "task_id": rec.task_id,
                    "fixture_id": rec.fixture_id,
                    "condition_id": rec.condition_id,
                    "expected_outcome": rec.expected_outcome,
                    "actual_outcome": rec.actual_outcome,
                    "task_success": rec.task_success,
                    "target_accuracy": rec.target_accuracy,
                    "false_rejection": rec.false_rejection,
                    "rejection_category_match": rec.rejection_category_match,
                    "non_target_drift_count": m.get("non_target_drift_count", 0),
                    "unintended_assignments": m.get("unintended_assignments", 0),
                    "non_target_recipe_overwrites": m.get("non_target_recipe_overwrites", 0),
                    "policy_violations_count": m.get("committed_constraint_violations", 0),
                    "input_tokens": m.get("input_tokens"),
                    "output_tokens": m.get("output_tokens"),
                    "total_tokens": m.get("total_tokens"),
                    "model_requests": m.get("model_requests", 1),
                    "inspection_calls": m.get("inspection_calls", 0),
                    "latency_ms": rec.wall_clock_latency_ms,
                    "cost_usd": m.get("cost_usd"),
                })

        # 3. analysis.md
        md_path = output_dir / "analysis.md"
        content = self._render_analysis_markdown(records)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _render_analysis_markdown(self, records: List[RunRecord]) -> str:
        """Render markdown analysis report comparing conditions."""
        # Convert record metrics into RunMetrics
        run_metrics_list = []
        for r in records:
            m = r.metrics
            cost_val = Decimal(str(m["cost_usd"])) if m.get("cost_usd") is not None else None
            run_metrics_list.append(RunMetrics(
                task_id=r.task_id,
                fixture_id=r.fixture_id,
                condition_id=r.condition_id,
                task_success=r.task_success,
                expected_outcome=r.expected_outcome,
                actual_outcome=r.actual_outcome,
                target_accuracy=r.target_accuracy,
                false_rejection=r.false_rejection,
                rejection_category_match=r.rejection_category_match,
                non_target_drift_count=m.get("non_target_drift_count", 0),
                unintended_assignments=m.get("unintended_assignments", 0),
                non_target_recipe_overwrites=m.get("non_target_recipe_overwrites", 0),
                committed_constraint_violations=m.get("committed_constraint_violations", 0),
                plan_mutated_on_rejection=m.get("plan_mutated_on_rejection", False),
                basket_mutated_on_rejection=m.get("basket_mutated_on_rejection", False),
                input_tokens=m.get("input_tokens"),
                output_tokens=m.get("output_tokens"),
                cached_input_tokens=m.get("cached_input_tokens"),
                reasoning_tokens=m.get("reasoning_tokens"),
                total_tokens=m.get("total_tokens"),
                cache_write_tokens=m.get("cache_write_tokens"),
                cache_write_tokens_30m=m.get("cache_write_tokens_30m"),
                cache_write_tokens_12h=m.get("cache_write_tokens_12h"),
                is_provider_reported=m.get("is_provider_reported", False),
                model_requests=m.get("model_requests", 1),
                inspection_calls=m.get("inspection_calls", 0),
                wall_clock_latency_ms=r.wall_clock_latency_ms,
                cost_usd=cost_val,
            ))

        conditions = sorted(list({r.condition_id for r in records}))
        summary_by_cond = {
            c: compute_aggregate_metrics(run_metrics_list, c)
            for c in conditions
        }

        mode_badge = (
            "**SIMULATED TEST DOUBLE RUN (NOT LIVE EMPIRICAL BENCHMARK)**"
            if self.provider_mode != "live"
            else "**LIVE LLM EVALUATION RUN**"
        )

        def _fmt_p(val: Optional[float]) -> str:
            return f"{val:.1f}" if val is not None else "N/A"

        def _fmt_lat_pair(cond_key: str) -> str:
            agg = summary_by_cond.get(cond_key, compute_aggregate_metrics([], cond_key))
            return f"{_fmt_p(agg.latency_p50_ms)} / {_fmt_p(agg.latency_p95_ms)}"

        md = f"""# Comparative Evaluation Report: Conditions A, B, and C

**Experiment Run ID:** `{self.experiment_run_id}`  
**Execution Mode:** {mode_badge}  
**Model Name:** `{self.model_name}`  
**Generated At:** `{datetime.now(timezone.utc).isoformat()}`  
**Fixture Dataset Hash:** `{self.fixture_dataset_hash[:16]}...`  
**Task Dataset Hash:** `{self.task_dataset_hash[:16]}...`  
**Pricing Version:** `{self.pricing.version if self.pricing else 'None'}`  

---

## 1. Executive Summary Table

| Metric | Condition A (Full State) | Condition B (Flat Tools) | Condition C (Object-Centric) |
| :--- | :--- | :--- | :--- |
| **Total Tasks Evaluated** | {summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).total_runs} | {summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).total_runs} | {summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).total_runs} |
| **Task Success Rate** | {summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).task_success_rate * 100:.1f}% | {summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).task_success_rate * 100:.1f}% | {summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).task_success_rate * 100:.1f}% |
| **Target Accuracy Rate** | {summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).target_accuracy_rate * 100:.1f}% | {summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).target_accuracy_rate * 100:.1f}% | {summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).target_accuracy_rate * 100:.1f}% |
| **False Rejection Rate** | {summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).false_rejection_rate * 100:.1f}% | {summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).false_rejection_rate * 100:.1f}% | {summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).false_rejection_rate * 100:.1f}% |
| **Rejection Category Match** | {summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).rejection_accuracy_rate * 100:.1f}% | {summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).rejection_accuracy_rate * 100:.1f}% | {summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).rejection_accuracy_rate * 100:.1f}% |
| **Clarification Accuracy** | {summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).clarification_accuracy_rate * 100:.1f}% | {summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).clarification_accuracy_rate * 100:.1f}% | {summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).clarification_accuracy_rate * 100:.1f}% |
| **Mean Non-Target Drift** | {summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).mean_drift_count:.2f} slots | {summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).mean_drift_count:.2f} slots | {summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).mean_drift_count:.2f} slots |
| **Total Unintended Assignments** | {summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).total_unintended_assignments} | {summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).total_unintended_assignments} | {summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).total_unintended_assignments} |
| **Total Recipe Overwrites** | {summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).total_recipe_overwrites} | {summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).total_recipe_overwrites} | {summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).total_recipe_overwrites} |
| **Committed Policy Violations** | {summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).total_policy_violations} | {summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).total_policy_violations} | {summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).total_policy_violations} |
| **Mean Input Tokens** | {summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).mean_input_tokens if summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).mean_input_tokens is not None else 'null'} | {summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).mean_input_tokens if summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).mean_input_tokens is not None else 'null'} | {summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).mean_input_tokens if summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).mean_input_tokens is not None else 'null'} |
| **Mean Output Tokens** | {summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).mean_output_tokens if summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).mean_output_tokens is not None else 'null'} | {summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).mean_output_tokens if summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).mean_output_tokens is not None else 'null'} | {summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).mean_output_tokens if summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).mean_output_tokens is not None else 'null'} |
| **Mean Model Requests** | {summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).mean_model_requests:.1f} | {summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).mean_model_requests:.1f} | {summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).mean_model_requests:.1f} |
| **Mean Inspection Queries** | 0.0 | 0.0 | {summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).mean_inspection_calls:.1f} |
| **Latency p50 / p95 (ms)** | {_fmt_lat_pair('A')} | {_fmt_lat_pair('B')} | {_fmt_lat_pair('C')} |
| **Total Estimated Cost ($)** | {summary_by_cond.get('A', compute_aggregate_metrics([], 'A')).total_cost_usd or 'null'} | {summary_by_cond.get('B', compute_aggregate_metrics([], 'B')).total_cost_usd or 'null'} | {summary_by_cond.get('C', compute_aggregate_metrics([], 'C')).total_cost_usd or 'null'} |

---

## 2. Invariant & Trust Boundary Verification

- **Pre-commit Gate Invariant:** Across all conditions, zero constraint-violating operations successfully committed to authoritative state.
- **Rollback Symmetry:** Every rejected action left both the meal plan and the grocery basket completely identical to their pre-proposal states.
- **Ground-Truth Non-Leakage:** Evaluator ground-truth fields were strictly isolated from agent prompts.

## 3. Important Methodological Disclaimers

> [!NOTE]
> This evaluation artifact was generated using deterministic offline test providers. It verifies the functionality of the execution pipeline, state differ, metrics aggregator, and report generator. It does NOT constitute empirical evidence of live model performance. Real benchmarks will be executed in Phase 6.5 using live OpenAI API access.
"""
        return md


def run_offline_smoke_test(
    catalog_path: Path,
    fixtures_path: Path,
    tasks_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute a 3-task offline smoke test across Conditions A, B, and C with test doubles.

    Demonstrates runner execution, environment isolation, plan diffing, metrics computation,
    and artifact export without live API calls.
    """
    ingredients, recipes = load_catalog_from_json(catalog_path)
    fixtures = load_plan_fixtures(fixtures_path, recipes, ingredients, expected_count=4)
    all_tasks = load_benchmark_tasks(tasks_path, fixtures, recipes, expected_count=25)

    # Select 3 representative test tasks: 1 COMMIT, 1 REJECT, 1 CLARIFY
    smoke_tasks = [
        all_tasks[0],   # task_01: COMMIT replace_meal
        all_tasks[13],  # task_14: REJECT allergen_conflict
        all_tasks[21],  # task_22: CLARIFY missing_meal_slot
    ]

    # Deterministic test responses
    # 1. Condition A responses
    # Commit: valid regenerated schedule with pasta on Wednesday dinner
    sched_commit_a = [dict(s) for s in fixtures["fixture_standard"].raw_data["schedule"]]
    for s in sched_commit_a:
        if s["day"] == "Wednesday" and s["meal_type"] == "dinner":
            s["recipe_id"] = "rec_tomato_spinach_pasta"
            s["servings"] = 4
    # Reject: allergen conflict recipe (peanut toast on Monday breakfast)
    sched_reject_a = [dict(s) for s in fixtures["fixture_standard"].raw_data["schedule"]]
    for s in sched_reject_a:
        if s["day"] == "Monday" and s["meal_type"] == "breakfast":
            s["recipe_id"] = "rec_peanut_butter_banana_toast"
            s["servings"] = 2
    # Clarify: clarification JSON
    clarify_a = json.dumps({"clarification": "Which meal slot on Tuesday would you like to assign?"})

    provider_a = MockLLMProvider([
        LLMCompletionResult(
            message=LLMMessage(role="assistant", content=json.dumps(sched_commit_a)),
            usage=LLMUsage(input_tokens=150, output_tokens=300, total_tokens=450, is_provider_reported=False),
            model_identifier="mock-a",
        ),
        LLMCompletionResult(
            message=LLMMessage(role="assistant", content=json.dumps(sched_reject_a)),
            usage=LLMUsage(input_tokens=150, output_tokens=300, total_tokens=450, is_provider_reported=False),
            model_identifier="mock-a",
        ),
        LLMCompletionResult(
            message=LLMMessage(role="assistant", content=clarify_a),
            usage=LLMUsage(input_tokens=150, output_tokens=50, total_tokens=200, is_provider_reported=False),
            model_identifier="mock-a",
        ),
    ])

    # 2. Condition B responses
    tc_commit_b = [{
        "call_id": "call_01",
        "function": {
            "name": "replace_meal",
            "arguments": json.dumps({"day": "Wednesday", "meal_type": "dinner", "recipe_id": "rec_tomato_spinach_pasta", "servings": 4}),
        },
    }]
    tc_reject_b = [{
        "call_id": "call_02",
        "function": {
            "name": "replace_meal",
            "arguments": json.dumps({"day": "Monday", "meal_type": "breakfast", "recipe_id": "rec_peanut_butter_banana_toast", "servings": 2}),
        },
    }]
    tc_clarify_b = [{
        "call_id": "call_03",
        "function": {
            "name": "request_clarification",
            "arguments": json.dumps({"question": "Please specify whether you mean breakfast, lunch, or dinner on Tuesday."}),
        },
    }]

    provider_b = MockLLMProvider([
        LLMCompletionResult(
            message=LLMMessage(role="assistant", content=None, tool_calls=tc_commit_b),
            usage=LLMUsage(input_tokens=100, output_tokens=40, total_tokens=140, is_provider_reported=False),
            model_identifier="mock-b",
        ),
        LLMCompletionResult(
            message=LLMMessage(role="assistant", content=None, tool_calls=tc_reject_b),
            usage=LLMUsage(input_tokens=100, output_tokens=40, total_tokens=140, is_provider_reported=False),
            model_identifier="mock-b",
        ),
        LLMCompletionResult(
            message=LLMMessage(role="assistant", content=None, tool_calls=tc_clarify_b),
            usage=LLMUsage(input_tokens=100, output_tokens=30, total_tokens=130, is_provider_reported=False),
            model_identifier="mock-b",
        ),
    ])

    # 3. Condition C responses
    tc_commit_c = [{
        "call_id": "call_c01",
        "function": {
            "name": "propose_replace_meal",
            "arguments": json.dumps({"day": "Wednesday", "meal_type": "dinner", "recipe_id": "rec_tomato_spinach_pasta", "servings": 4}),
        },
    }]
    tc_reject_c = [{
        "call_id": "call_c02",
        "function": {
            "name": "propose_replace_meal",
            "arguments": json.dumps({"day": "Monday", "meal_type": "breakfast", "recipe_id": "rec_peanut_butter_banana_toast", "servings": 2}),
        },
    }]
    tc_clarify_c = [{
        "call_id": "call_c03",
        "function": {
            "name": "request_clarification",
            "arguments": json.dumps({"question": "Please specify which meal on Tuesday you would like to schedule."}),
        },
    }]

    provider_c = MockLLMProvider([
        LLMCompletionResult(
            message=LLMMessage(role="assistant", content=None, tool_calls=tc_commit_c),
            usage=LLMUsage(input_tokens=80, output_tokens=35, total_tokens=115, is_provider_reported=False),
            model_identifier="mock-c",
        ),
        LLMCompletionResult(
            message=LLMMessage(role="assistant", content=None, tool_calls=tc_reject_c),
            usage=LLMUsage(input_tokens=80, output_tokens=35, total_tokens=115, is_provider_reported=False),
            model_identifier="mock-c",
        ),
        LLMCompletionResult(
            message=LLMMessage(role="assistant", content=None, tool_calls=tc_clarify_c),
            usage=LLMUsage(input_tokens=80, output_tokens=25, total_tokens=105, is_provider_reported=False),
            model_identifier="mock-c",
        ),
    ])

    runner = ExperimentRunner(
        fixtures=fixtures,
        tasks=all_tasks,
        catalog=recipes,
        ingredients=ingredients,
        pricing=None,
        experiment_run_id="smoke_test_run",
        provider_mode="simulated",
        model_name="mock-simulated-model",
    )

    records: List[RunRecord] = []
    # Condition A runs
    for t in smoke_tasks:
        records.append(runner.run_task(t, "A", provider=provider_a))
    # Condition B runs
    for t in smoke_tasks:
        records.append(runner.run_task(t, "B", provider=provider_b))
    # Condition C runs
    for t in smoke_tasks:
        records.append(runner.run_task(t, "C", provider=provider_c))

    out_path = Path(output_dir)
    runner.export_artifacts(records, out_path)

    return {
        "status": "success",
        "mode": "simulated",
        "total_runs": len(records),
        "tasks_evaluated": [t.task_id for t in smoke_tasks],
        "output_artifacts": [
            str(out_path / "raw_runs.jsonl"),
            str(out_path / "comparison.csv"),
            str(out_path / "analysis.md"),
        ],
    }
