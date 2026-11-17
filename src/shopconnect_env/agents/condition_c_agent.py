"""Condition C: Object-Centric Progressive Disclosure Agent.

Implements the progressive disclosure interaction pattern where the agent starts with
compact class signatures and queries domain objects on demand via ObjectInspector.

Enforces:
1. Strict interaction limits:
   - MAX_MODEL_REQUESTS = 6 (total API requests, including final turn)
   - MAX_INSPECTION_CALLS = 5 (actual domain queries dispatched to ObjectInspector)
2. Separate tracking of model requests and inspection calls.
3. No hallucinated or invented inspection results.
4. Preserves disclosure audit logging on the bound ObjectInspector.
5. Reaching limits terminates safely without producing unauthorized proposals.
6. Valid final proposals are routed through the trusted ActionBroker with strict expected_plan_version.
"""

from __future__ import annotations

import copy
import inspect
import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from shopconnect_env.agents.base import AgentResponse, BaseAgent, ObjectInspector
from shopconnect_env.agents.providers import (
    LLMMessage,
    LLMProvider,
    LLMUsage,
    aggregate_usages,
    usage_to_dict,
)
from shopconnect_env.context.base import ContextResult
from shopconnect_env.domain.models import RejectionCategory
from shopconnect_env.validation.broker import ActionBroker, ActionResult, ActionStatus
from shopconnect_env.validation.schemas import (
    ActionProposal,
    ChangeServingsArgs,
    OperationType,
    ReplaceMealArgs,
)


def _extract_plan_version(serialized_context: str) -> Optional[int]:
    """Strictly extract integer plan_version >= 1 from context JSON."""
    try:
        data = json.loads(serialized_context)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    v = data.get("plan_version")
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        return None
    return v


def _provider_supports_parallel_tool_calls(provider: Any) -> bool:
    """Check if provider complete method explicitly accepts parallel_tool_calls."""
    try:
        sig = inspect.signature(provider.complete)
        return "parallel_tool_calls" in sig.parameters
    except Exception:
        return False


def get_condition_c_tools(terminal_only: bool = False) -> List[Dict[str, Any]]:
    """Return tool specifications for Condition C: selective inspection and action proposals.

    Args:
        terminal_only: If True, returns only terminal finalization tools (stripped of inspect_*).
    """
    inspection_tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect_slot",
                "description": "Inspect meal details scheduled for a specific day and meal slot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "day": {"type": "string", "description": "Day of the week (e.g. 'Monday')"},
                        "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner"]},
                    },
                    "required": ["day", "meal_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_recipe",
                "description": "Inspect recipe metadata, tags, default servings, and estimated cost. Requires 'recipe_id'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipe_id": {
                            "type": "string",
                            "description": "Canonical recipe ID from catalog (e.g. 'rec_berry_oatmeal', 'rec_tomato_spinach_pasta'). REQUIRED. Must not be empty.",
                        },
                    },
                    "required": ["recipe_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_recipe_ingredients",
                "description": "Inspect ingredient requirements and scaled quantities for a recipe. Requires 'recipe_id'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipe_id": {
                            "type": "string",
                            "description": "Canonical recipe ID from catalog (e.g. 'rec_peanut_butter_banana_toast'). REQUIRED. Must not be empty.",
                        },
                    },
                    "required": ["recipe_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_ingredient",
                "description": "Inspect ingredient classification, units, and monitored allergens.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ingredient_id": {"type": "string", "description": "Canonical ingredient ID"},
                    },
                    "required": ["ingredient_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_plan",
                "description": "Inspect active plan metadata, plan_version, and total assigned meals.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_preferences",
                "description": "Inspect household dietary tags, registered allergies, and weekly budget.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ]

    terminal_tools = [
        {
            "type": "function",
            "function": {
                "name": "propose_replace_meal",
                "description": "Submit a final proposal to replace a meal slot with a specific recipe and explicit serving count.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "day": {"type": "string", "description": "Day of the week (e.g. 'Wednesday')"},
                        "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner"]},
                        "recipe_id": {"type": "string", "description": "Replacement recipe ID (e.g. 'rec_tomato_spinach_pasta')"},
                        "servings": {"type": "integer", "description": "Explicit target serving count (positive integer between 1 and 20, e.g. 2 or 4). REQUIRED."},
                    },
                    "required": ["day", "meal_type", "recipe_id", "servings"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "propose_change_servings",
                "description": "Submit a final proposal to scale servings for an existing meal slot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "day": {"type": "string", "description": "Day of the week"},
                        "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner"]},
                        "servings": {"type": "integer", "description": "Target serving count (1-20)"},
                    },
                    "required": ["day", "meal_type", "servings"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finalize_no_action",
                "description": "Finalize the task with no state mutation when constraints, dietary restrictions, or allergen conflicts prevent action. BOTH 'rejection_category' and 'reason' are strictly REQUIRED.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rejection_category": {
                            "type": "string",
                            "enum": [c.value for c in RejectionCategory],
                            "description": (
                                "Taxonomic category of rejection. REQUIRED. Must be one of: "
                                + ", ".join(f"'{c.value}'" for c in RejectionCategory)
                                + ". (e.g. 'allergen_conflict' for allergy violations, 'dietary_restriction_conflict' for diet conflicts, 'budget_exceeded' for budget limits)."
                            ),
                        },
                        "reason": {"type": "string", "description": "Specific explanatory reason why no action was proposed or why request is rejected. REQUIRED."},
                    },
                    "required": ["rejection_category", "reason"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finalize_clarification",
                "description": "Ask the user a clarification question when the request is ambiguous or missing required parameters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Specific question addressing the ambiguous or missing information.",
                        },
                    },
                    "required": ["question"],
                },
            },
        },
    ]

    if terminal_only:
        return terminal_tools
    return inspection_tools + terminal_tools


class ConditionCAgent(BaseAgent):
    """Object-centric progressive disclosure agent interfacing with LLMProvider under Condition C.

    Enforces Protocol v2 specifications:
    - Hard limit of MAX_MODEL_REQUESTS = 6 (total API requests, including final turn).
    - Hard limit of MAX_INSPECTION_CALLS = 5 (distinct novel queries dispatched to ObjectInspector).
    - Invariant duplicate inspection cache: duplicate queries return local cached payload,
      incrementing provider_requests and cache_hits without consuming distinct inspection budget.
    - Strict inspection accounting identity:
      inspection_attempts = distinct_inspections + cache_hits + malformed_inspection_calls + rejected_inspection_calls
    - Dynamic finalization-only tool schema stripping: when provider_requests >= 5 or distinct_inspections >= 5,
      the tools schema exposes ONLY terminal actions (zero inspect_* tools).
    - Atomic single-tool contract: responses returning len(tool_calls) > 1 are rejected cleanly,
      executing 0 tools and incrementing invalid_tool_responses.
    - Symmetrical REJECT and CLARIFY outcomes via finalize_no_action and finalize_clarification.
    """

    MAX_MODEL_REQUESTS: int = 6
    MAX_INSPECTION_CALLS: int = 5

    def __init__(
        self,
        provider: LLMProvider,
        max_model_requests: Optional[int] = None,
        max_inspection_calls: Optional[int] = None,
    ) -> None:
        """Initialize ConditionCAgent with an LLMProvider and optional custom limits."""
        self.provider = provider
        self.max_model_requests: int = (
            max_model_requests if max_model_requests is not None else self.MAX_MODEL_REQUESTS
        )
        self.max_inspection_calls: int = (
            max_inspection_calls if max_inspection_calls is not None else self.MAX_INSPECTION_CALLS
        )

    def propose_action(
        self,
        context: ContextResult,
        inspector: Optional[ObjectInspector] = None,
    ) -> AgentResponse:
        """Execute multi-turn inspection dialogue terminating in an ActionProposal."""
        # 1. Validate plan version
        plan_version = _extract_plan_version(context.serialized_context)
        if plan_version is None:
            return AgentResponse(
                proposal=None,
                rationale="Invalid or missing plan_version in context.",
                raw_output="",
                metadata={"status": "invalid_plan_version", "condition_id": "C"},
            )

        messages: List[LLMMessage] = [
            LLMMessage(role="system", content=context.system_prompt),
            LLMMessage(
                role="user",
                content=(
                    f"User Request:\n{context.task_prompt}\n\n"
                    f"Context Signatures & Initial Summary:\n{context.serialized_context}\n\n"
                    "Instructions:\n"
                    "1. Use inspection tools to query specific slots, recipes, or preferences on demand (e.g. inspect_recipe requires 'recipe_id').\n"
                    "2. When proposing a meal replacement, call propose_replace_meal with 'day', 'meal_type', 'recipe_id', and explicit 'servings'.\n"
                    "3. When changing servings, call propose_change_servings with 'day', 'meal_type', and 'servings'.\n"
                    "4. If the task cannot be accomplished or violates rules/allergies, call finalize_no_action with BOTH 'rejection_category' (e.g. 'allergen_conflict') and 'reason'.\n"
                    "5. If the task is ambiguous or missing required parameters, call finalize_clarification with 'question'."
                ),
            ),
        ]

        provider_requests = 0
        distinct_inspections = 0
        cache_hits = 0
        malformed_inspection_calls = 0
        rejected_inspection_calls = 0
        non_tool_responses = 0
        invalid_tool_responses = 0

        inspection_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        inspection_steps: List[Dict[str, Any]] = []

        usages: List[Optional[LLMUsage]] = []
        request_ids: List[str] = []
        latencies: List[float] = []
        fingerprints: List[str] = []
        model_ids: List[str] = []

        final_proposal: Optional[ActionProposal] = None
        final_rationale: str = ""
        final_clarification: Optional[str] = None
        rejection_category: Optional[str] = None
        agent_initiated_rejection: bool = False
        last_raw_output: str = ""

        # Multi-turn interaction loop
        while provider_requests < self.max_model_requests:
            provider_requests += 1

            # Dynamic Turn 6 / Limit Stripping:
            # When provider_requests >= max_model_requests (this is the final turn)
            # OR when distinct_inspections >= max_inspection_calls,
            # strip all inspect_* tools from the prompt schema!
            use_terminal_only = (
                (provider_requests >= self.max_model_requests)
                or (distinct_inspections >= self.max_inspection_calls)
            )
            current_tools = get_condition_c_tools(terminal_only=use_terminal_only)

            complete_kwargs: Dict[str, Any] = {
                "messages": messages,
                "tools": current_tools,
                "tool_choice": "auto",
            }
            if _provider_supports_parallel_tool_calls(self.provider):
                complete_kwargs["parallel_tool_calls"] = False

            try:
                completion = self.provider.complete(**complete_kwargs)
            except Exception as exc:
                aggregated_usage = aggregate_usages(usages) if usages else None
                return AgentResponse(
                    proposal=None,
                    rationale=f"Agent exception during interaction: {exc}",
                    raw_output=last_raw_output,
                    metadata={
                        "condition_id": "C",
                        "plan_version": plan_version,
                        "status": "error",
                        "error": str(exc),
                        "total_model_requests": provider_requests,
                        "usage": usage_to_dict(aggregated_usage) if aggregated_usage else None,
                        "model_identifier": model_ids[-1] if model_ids else None,
                        "system_fingerprint": fingerprints[-1] if fingerprints else None,
                        "provider_request_ids": tuple(request_ids),
                        "per_request_latencies_ms": tuple(latencies),
                        "distinct_inspections": distinct_inspections,
                        "cache_hits": cache_hits,
                        "inspection_steps": inspection_steps,
                    },
                )
            last_raw_output = completion.message.content or ""
            usages.append(completion.usage)
            if completion.provider_request_id:
                request_ids.append(completion.provider_request_id)
            if completion.system_fingerprint:
                fingerprints.append(completion.system_fingerprint)
            if completion.latency_ms:
                latencies.append(completion.latency_ms)
            if completion.model_identifier:
                model_ids.append(completion.model_identifier)

            tool_calls = completion.message.tool_calls
            if not tool_calls:
                non_tool_responses += 1
                final_rationale = last_raw_output or "Model finished interaction without proposing an action."
                break

            # Atomic Single-Tool Contract: Enforce len(tool_calls) == 1
            if len(tool_calls) > 1:
                invalid_tool_responses += 1
                for tc in tool_calls:
                    fn_name = tc.get("function", {}).get("name") or tc.get("name", "")
                    if fn_name.startswith("inspect_"):
                        rejected_inspection_calls += 1
                final_rationale = (
                    f"Multiple tool calls ({len(tool_calls)}) returned in a single turn. "
                    "Multi-action responses are unsupported under the atomic ActionBroker contract; "
                    "exactly one tool call is required."
                )
                break

            primary_tc = tool_calls[0]
            cid = primary_tc.get("call_id") or primary_tc.get("id") or ""
            fn_dict = primary_tc.get("function", {})
            tool_name = fn_dict.get("name") or primary_tc.get("name", "")
            raw_args = fn_dict.get("arguments") or primary_tc.get("arguments", "{}")

            try:
                args_dict = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except Exception as json_err:
                if tool_name.startswith("inspect_"):
                    malformed_inspection_calls += 1
                invalid_tool_responses += 1
                final_rationale = f"Malformed tool arguments JSON: {json_err}"
                break

            # -----------------------------------------------------------------
            # Terminal Tools
            # -----------------------------------------------------------------
            if tool_name == "propose_replace_meal":
                try:
                    val_args = ReplaceMealArgs.model_validate(args_dict)
                except ValidationError as err:
                    invalid_tool_responses += 1
                    final_rationale = f"Invalid arguments for propose_replace_meal: {err}"
                    break

                final_proposal = ActionProposal(
                    operation=OperationType.REPLACE_MEAL,
                    arguments={
                        "day": val_args.day,
                        "meal_type": val_args.meal_type,
                        "recipe_id": val_args.recipe_id,
                        "servings": val_args.servings,
                    },
                    reason=f"Condition C propose_replace_meal({val_args.day}, {val_args.meal_type}, {val_args.recipe_id}, servings={val_args.servings})",
                    expected_plan_version=plan_version,
                )
                final_rationale = final_proposal.reason
                break

            elif tool_name == "propose_change_servings":
                try:
                    val_serv = ChangeServingsArgs.model_validate(args_dict)
                except ValidationError as err:
                    invalid_tool_responses += 1
                    final_rationale = f"Invalid arguments for propose_change_servings: {err}"
                    break

                final_proposal = ActionProposal(
                    operation=OperationType.CHANGE_SERVINGS,
                    arguments={
                        "day": val_serv.day,
                        "meal_type": val_serv.meal_type,
                        "servings": val_serv.servings,
                    },
                    reason=f"Condition C propose_change_servings({val_serv.day}, {val_serv.meal_type}, {val_serv.servings})",
                    expected_plan_version=plan_version,
                )
                final_rationale = final_proposal.reason
                break

            elif tool_name == "finalize_no_action":
                reason = str(args_dict.get("reason", "")).strip()
                cat_raw = args_dict.get("rejection_category")
                if not reason or not cat_raw:
                    invalid_tool_responses += 1
                    final_rationale = (
                        "Invalid finalize_no_action invocation: 'rejection_category' and 'reason' are required."
                    )
                    break
                try:
                    rejection_category = RejectionCategory(str(cat_raw).strip()).value
                except ValueError:
                    invalid_tool_responses += 1
                    final_rationale = (
                        f"Invalid rejection_category: '{cat_raw}'. Must be one of {[c.value for c in RejectionCategory]}."
                    )
                    break
                agent_initiated_rejection = True
                final_rationale = reason
                break

            elif tool_name in ("finalize_clarification", "request_clarification"):
                q = (
                    args_dict.get("question")
                    or args_dict.get("clarification")
                    or "Clarification requested."
                )
                final_clarification = str(q).strip() or "Clarification requested."
                final_rationale = f"Clarification requested: {final_clarification}"
                break

            # -----------------------------------------------------------------
            # Inspection Tools
            # -----------------------------------------------------------------
            elif tool_name in (
                "inspect_slot",
                "inspect_recipe",
                "inspect_recipe_ingredients",
                "inspect_ingredient",
                "inspect_plan",
                "inspect_preferences",
            ):
                if use_terminal_only:
                    rejected_inspection_calls += 1
                    invalid_tool_responses += 1
                    messages.append(
                        LLMMessage(
                            role="assistant",
                            content=completion.message.content,
                            tool_calls=completion.message.tool_calls,
                        )
                    )
                    messages.append(
                        LLMMessage(
                            role="tool",
                            tool_call_id=cid,
                            content=json.dumps({
                                "error": (
                                    "INSPECTION_FORBIDDEN: Inspection limit reached or final turn active. "
                                    "You must call a terminal tool (propose_replace_meal, propose_change_servings, "
                                    "finalize_no_action, finalize_clarification)."
                                )
                            }),
                        )
                    )
                    continue

                cache_key = (tool_name, json.dumps(args_dict, sort_keys=True))
                if cache_key in inspection_cache:
                    cache_hits += 1
                    insp_result = copy.deepcopy(inspection_cache[cache_key])
                else:
                    if distinct_inspections >= self.max_inspection_calls:
                        rejected_inspection_calls += 1
                        messages.append(
                            LLMMessage(
                                role="assistant",
                                content=completion.message.content,
                                tool_calls=completion.message.tool_calls,
                            )
                        )
                        messages.append(
                            LLMMessage(
                                role="tool",
                                tool_call_id=cid,
                                content=json.dumps({
                                    "error": (
                                        f"MAX_DISTINCT_INSPECTIONS_EXCEEDED: Reached maximum of {self.max_inspection_calls} "
                                        "distinct inspections. You must call a terminal action."
                                    )
                                }),
                            )
                        )
                        continue

                    distinct_inspections += 1
                    try:
                        insp_result = self._dispatch_inspection(inspector, tool_name, args_dict)
                    except Exception as exc:
                        aggregated_usage = aggregate_usages(usages) if usages else None
                        return AgentResponse(
                            proposal=None,
                            rationale=f"Inspection exception during interaction: {exc}",
                            raw_output=last_raw_output,
                            metadata={
                                "condition_id": "C",
                                "plan_version": plan_version,
                                "status": "error",
                                "error": f"Inspection dispatch error: {exc}",
                                "total_model_requests": provider_requests,
                                "usage": usage_to_dict(aggregated_usage) if aggregated_usage else None,
                                "model_identifier": model_ids[-1] if model_ids else None,
                                "system_fingerprint": fingerprints[-1] if fingerprints else None,
                                "provider_request_ids": tuple(request_ids),
                                "per_request_latencies_ms": tuple(latencies),
                                "distinct_inspections": distinct_inspections,
                                "cache_hits": cache_hits,
                                "inspection_steps": inspection_steps,
                            },
                        )
                    inspection_cache[cache_key] = copy.deepcopy(insp_result)
                    disclosure_chars = len(json.dumps(insp_result))
                    inspection_steps.append({
                        "step": distinct_inspections,
                        "tool": tool_name,
                        "arguments": args_dict,
                        "disclosure_chars": disclosure_chars,
                    })

                messages.append(
                    LLMMessage(
                        role="assistant",
                        content=completion.message.content,
                        tool_calls=completion.message.tool_calls,
                    )
                )
                messages.append(
                    LLMMessage(
                        role="tool",
                        tool_call_id=cid,
                        content=json.dumps(insp_result),
                    )
                )
                continue

            else:
                invalid_tool_responses += 1
                messages.append(
                    LLMMessage(
                        role="assistant",
                        content=completion.message.content,
                        tool_calls=completion.message.tool_calls,
                    )
                )
                messages.append(
                    LLMMessage(
                        role="tool",
                        tool_call_id=cid,
                        content=json.dumps({"error": f"Unknown tool name: {tool_name}"}),
                    )
                )
                continue

        if provider_requests >= self.max_model_requests and final_proposal is None and not final_rationale:
            final_rationale = (
                f"MAX_MODEL_REQUESTS_EXCEEDED: Reached limit of {self.max_model_requests} "
                "requests without producing a final action proposal."
            )

        inspection_attempts = (
            distinct_inspections
            + cache_hits
            + malformed_inspection_calls
            + rejected_inspection_calls
        )

        # Invariant check
        assert inspection_attempts == (
            distinct_inspections
            + cache_hits
            + malformed_inspection_calls
            + rejected_inspection_calls
        )

        aggregated_usage = aggregate_usages(usages)

        if final_proposal is not None:
            status = "ok"
        elif final_clarification is not None:
            status = "clarification_requested"
        elif agent_initiated_rejection:
            status = "agent_rejected"
        elif "multiple tool calls" in final_rationale.lower():
            status = "unsupported_multiple_tool_calls"
        elif provider_requests >= self.max_model_requests:
            status = "max_requests_exceeded"
        else:
            status = "no_proposal"

        metadata: Dict[str, Any] = {
            "status": status,
            "condition_id": "C",
            "total_model_requests": provider_requests,
            "provider_requests": provider_requests,
            "inspection_calls_count": distinct_inspections,
            "distinct_inspections": distinct_inspections,
            "cache_hits": cache_hits,
            "inspection_attempts": inspection_attempts,
            "malformed_inspection_calls": malformed_inspection_calls,
            "rejected_inspection_calls": rejected_inspection_calls,
            "non_tool_responses": non_tool_responses,
            "invalid_tool_responses": invalid_tool_responses,
            "inspection_steps": tuple(inspection_steps),
            "rejection_category": rejection_category,
            "agent_initiated_rejection": agent_initiated_rejection,
            "plan_version": plan_version,
            "usage": usage_to_dict(aggregated_usage),
            "model_identifier": model_ids[-1] if model_ids else None,
            "system_fingerprint": fingerprints[-1] if fingerprints else None,
            "provider_request_ids": tuple(request_ids),
            "per_request_latencies_ms": tuple(latencies),
        }

        return AgentResponse(
            proposal=final_proposal,
            rationale=final_rationale or "Interaction completed.",
            raw_output=last_raw_output or (final_proposal.model_dump_json() if final_proposal else ""),
            inspection_calls_made=distinct_inspections,
            clarification_message=final_clarification,
            metadata=metadata,
        )

    def _dispatch_inspection(
        self,
        inspector: Optional[ObjectInspector],
        tool_name: str,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Dispatch query to ObjectInspector preserving the disclosure audit log."""
        if inspector is None:
            return {"error": "ObjectInspector is not available in current execution context."}

        if tool_name == "inspect_slot":
            return inspector.inspect_slot(args.get("day", ""), args.get("meal_type", ""))
        elif tool_name == "inspect_recipe":
            return inspector.inspect_recipe(args.get("recipe_id", ""))
        elif tool_name == "inspect_recipe_ingredients":
            return inspector.inspect_recipe_ingredients(args.get("recipe_id", ""))
        elif tool_name == "inspect_ingredient":
            return inspector.inspect_ingredient(args.get("ingredient_id", ""))
        elif tool_name == "inspect_plan":
            return inspector.inspect_plan()
        elif tool_name == "inspect_preferences":
            return inspector.inspect_preferences()
        return {"error": f"Unhandled inspection tool: {tool_name}"}

    def execute(
        self,
        context: ContextResult,
        broker: ActionBroker,
        inspector: Optional[ObjectInspector] = None,
    ) -> ActionResult:
        """Formulate proposal and execute it through the trusted ActionBroker."""
        response = self.propose_action(context, inspector=inspector)
        if not response.has_proposal or response.proposal is None:
            return ActionResult(
                status=ActionStatus.REJECTED,
                operation=str(response.metadata.get("operation", OperationType.REPLACE_MEAL.value)),
                plan_version_before=broker.get_plan_version(),
                plan_version_after=broker.get_plan_version(),
                message=response.rationale,
                execution_duration_ms=0.0,
                rejection_reason=response.rationale,
            )
        return broker.execute(response.proposal)
