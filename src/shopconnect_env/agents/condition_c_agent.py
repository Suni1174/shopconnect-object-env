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

import json
from typing import Any, Dict, List, Optional

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


def get_condition_c_tools() -> List[Dict[str, Any]]:
    """Return tool specifications for Condition C: selective inspection and action proposals."""
    return [
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
                "description": "Inspect recipe metadata, tags, default servings, and estimated cost.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipe_id": {"type": "string", "description": "Canonical recipe ID (e.g. 'rec_berry_oatmeal')"},
                    },
                    "required": ["recipe_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_recipe_ingredients",
                "description": "Inspect ingredient requirements and scaled quantities for a recipe.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipe_id": {"type": "string", "description": "Canonical recipe ID"},
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
        {
            "type": "function",
            "function": {
                "name": "propose_replace_meal",
                "description": "Submit a final proposal to replace a meal slot with a specific recipe.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "day": {"type": "string", "description": "Day of the week (e.g. 'Wednesday')"},
                        "meal_type": {"type": "string", "enum": ["breakfast", "lunch", "dinner"]},
                        "recipe_id": {"type": "string", "description": "Replacement recipe ID"},
                        "servings": {"type": "integer", "description": "Optional servings count (1-20)"},
                    },
                    "required": ["day", "meal_type", "recipe_id"],
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
                "description": "Finalize the task with no state mutation (e.g. when constraints prevent action).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Reason why no action was proposed"},
                    },
                    "required": ["reason"],
                },
            },
        },
    ]


class ConditionCAgent(BaseAgent):
    """Object-centric progressive disclosure agent interfacing with LLMProvider under Condition C.

    Enforces:
    - Hard limit of MAX_MODEL_REQUESTS = 6.
    - Hard limit of MAX_INSPECTION_CALLS = 5.
    - Independent tracking of model requests and inspection queries.
    - Safe termination when limits are reached without valid proposals.
    """

    MAX_MODEL_REQUESTS: int = 6
    MAX_INSPECTION_CALLS: int = 5

    def __init__(self, provider: LLMProvider) -> None:
        """Initialize ConditionCAgent with an LLMProvider."""
        self.provider = provider

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

        tools = get_condition_c_tools()
        messages: List[LLMMessage] = [
            LLMMessage(role="system", content=context.system_prompt),
            LLMMessage(
                role="user",
                content=(
                    f"User Request:\n{context.task_prompt}\n\n"
                    f"Context Signatures & Initial Summary:\n{context.serialized_context}\n\n"
                    "Instructions:\n"
                    "1. Use inspection tools to query specific slots, recipes, or preferences on demand.\n"
                    "2. When you have resolved the task, call propose_replace_meal or propose_change_servings.\n"
                    "3. If the task cannot be accomplished or violates rules, call finalize_no_action."
                ),
            ),
        ]

        model_requests_count = 0
        inspection_calls_count = 0
        inspection_steps: List[Dict[str, Any]] = []
        usages: List[Optional[LLMUsage]] = []
        final_proposal: Optional[ActionProposal] = None
        final_rationale: str = ""
        last_raw_output: str = ""

        # Multi-turn interaction loop
        while model_requests_count < self.MAX_MODEL_REQUESTS:
            model_requests_count += 1
            completion = self.provider.complete(messages=messages, tools=tools, tool_choice="auto")
            last_raw_output = completion.message.content or ""
            usages.append(completion.usage)

            tool_calls = completion.message.tool_calls
            if not tool_calls:
                # Model stopped issuing tool calls and returned plain text
                final_rationale = last_raw_output or "Model finished interaction without proposing an action."
                break

            # 1. Append single assistant message representing this turn's response
            messages.append(
                LLMMessage(
                    role="assistant",
                    content=completion.message.content,
                    tool_calls=completion.message.tool_calls,
                )
            )

            # 2. Process tool calls in this turn
            executed_any_inspection = False
            for tc in tool_calls:
                cid = tc.get("call_id") or tc.get("id") or ""
                fn_dict = tc.get("function", {})
                tool_name = fn_dict.get("name") or tc.get("name", "")
                raw_args = fn_dict.get("arguments") or tc.get("arguments", "{}")

                try:
                    args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args_dict = {}

                # -------------------------------------------------------------
                # A. FINAL ACTION PROPOSALS
                # -------------------------------------------------------------
                if tool_name == "propose_replace_meal":
                    try:
                        val_args = ReplaceMealArgs.model_validate(args_dict)
                    except ValidationError as err:
                        final_rationale = f"Invalid arguments for propose_replace_meal: {err}"
                        final_proposal = None
                        messages.append(
                            LLMMessage(
                                role="tool",
                                tool_call_id=cid,
                                content=json.dumps({"error": final_rationale}),
                            )
                        )
                        break

                    final_proposal = ActionProposal(
                        operation=OperationType.REPLACE_MEAL,
                        arguments={
                            "day": val_args.day,
                            "meal_type": val_args.meal_type,
                            "recipe_id": val_args.recipe_id,
                        },
                        reason=f"Condition C propose_replace_meal({val_args.day}, {val_args.meal_type}, {val_args.recipe_id})",
                        expected_plan_version=plan_version,
                    )
                    final_rationale = final_proposal.reason
                    messages.append(
                        LLMMessage(
                            role="tool",
                            tool_call_id=cid,
                            content=json.dumps({"status": "proposal_staged", "operation": "replace_meal"}),
                        )
                    )
                    break

                elif tool_name == "propose_change_servings":
                    try:
                        val_serv = ChangeServingsArgs.model_validate(args_dict)
                    except ValidationError as err:
                        final_rationale = f"Invalid arguments for propose_change_servings: {err}"
                        final_proposal = None
                        messages.append(
                            LLMMessage(
                                role="tool",
                                tool_call_id=cid,
                                content=json.dumps({"error": final_rationale}),
                            )
                        )
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
                    messages.append(
                        LLMMessage(
                            role="tool",
                            tool_call_id=cid,
                            content=json.dumps({"status": "proposal_staged", "operation": "change_servings"}),
                        )
                    )
                    break

                elif tool_name == "finalize_no_action":
                    final_proposal = None
                    final_rationale = args_dict.get("reason", "Agent concluded no action should be taken.")
                    messages.append(
                        LLMMessage(
                            role="tool",
                            tool_call_id=cid,
                            content=json.dumps({"status": "finalized", "reason": final_rationale}),
                        )
                    )
                    break

                # -------------------------------------------------------------
                # B. ON-DEMAND DOMAIN INSPECTION QUERIES
                # -------------------------------------------------------------
                elif tool_name in (
                    "inspect_slot",
                    "inspect_recipe",
                    "inspect_recipe_ingredients",
                    "inspect_ingredient",
                    "inspect_plan",
                    "inspect_preferences",
                ):
                    # Check inspection call limit (MAX 5)
                    if inspection_calls_count >= self.MAX_INSPECTION_CALLS:
                        # Reached maximum allowed domain inspections
                        error_payload = {
                            "error": (
                                f"MAX_INSPECTION_CALLS_EXCEEDED: Reached maximum of {self.MAX_INSPECTION_CALLS} "
                                "inspection calls. You must now submit a proposal or call finalize_no_action."
                            )
                        }
                        messages.append(
                            LLMMessage(
                                role="tool",
                                tool_call_id=cid,
                                content=json.dumps(error_payload),
                            )
                        )
                        continue

                    # Execute query via ObjectInspector
                    executed_any_inspection = True
                    inspection_calls_count += 1
                    insp_result = self._dispatch_inspection(inspector, tool_name, args_dict)
                    disclosure_chars = len(json.dumps(insp_result))
                    inspection_steps.append({
                        "step": inspection_calls_count,
                        "tool": tool_name,
                        "arguments": args_dict,
                        "disclosure_chars": disclosure_chars,
                    })

                    messages.append(
                        LLMMessage(
                            role="tool",
                            tool_call_id=cid,
                            content=json.dumps(insp_result),
                        )
                    )

                else:
                    # Unknown tool name
                    messages.append(
                        LLMMessage(
                            role="tool",
                            tool_call_id=cid,
                            content=json.dumps({"error": f"Unknown tool name: {tool_name}"}),
                        )
                    )

            # If a proposal was reached or finalized, exit loop
            if final_proposal is not None or final_rationale:
                break

            # If no inspection was executed and limit reached, break
            if not executed_any_inspection and inspection_calls_count >= self.MAX_INSPECTION_CALLS:
                break

        # Check if loop terminated due to model request ceiling
        if model_requests_count >= self.MAX_MODEL_REQUESTS and final_proposal is None and not final_rationale:
            final_rationale = (
                f"MAX_MODEL_REQUESTS_EXCEEDED: Reached limit of {self.MAX_MODEL_REQUESTS} "
                "requests without producing a final action proposal."
            )

        aggregated_usage = aggregate_usages(usages)

        metadata: Dict[str, Any] = {
            "status": "ok" if final_proposal is not None else "no_proposal",
            "condition_id": "C",
            "total_model_requests": model_requests_count,
            "inspection_calls_count": inspection_calls_count,
            "inspection_steps": tuple(inspection_steps),
            "plan_version": plan_version,
            "usage": usage_to_dict(aggregated_usage),
        }

        return AgentResponse(
            proposal=final_proposal,
            rationale=final_rationale or "Interaction completed.",
            raw_output=last_raw_output or (final_proposal.model_dump_json() if final_proposal else ""),
            inspection_calls_made=inspection_calls_count,
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
