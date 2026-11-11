"""Condition B: Conventional Flat Tool-Calling Agent.

Implements the standard tool-calling baseline where the agent receives flat function
specifications alongside a structured state snapshot.

Enforces:
1. Supplies flat tool definitions (replace_meal, change_servings) derived from Stage 3 Pydantic schemas.
2. Parses function calls, preserving call IDs and argument payloads.
3. Supports multiple returned tool calls without dropping any.
4. Validates arguments using Stage 3 Pydantic schemas (ReplaceMealArgs, ChangeServingsArgs).
5. Routes proposals through the trusted ActionBroker with strict expected_plan_version.
6. Safely handles missing, malformed, or unsupported tool calls without state mutation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from shopconnect_env.agents.base import AgentResponse, BaseAgent, ObjectInspector
from shopconnect_env.agents.providers import LLMMessage, LLMProvider, usage_to_dict
from shopconnect_env.context.base import ContextResult
from shopconnect_env.context.tool_context import get_conventional_tool_definitions
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


class ConditionBAgent(BaseAgent):
    """Conventional tool agent interfacing with LLMProvider under Condition B."""

    def __init__(self, provider: LLMProvider) -> None:
        """Initialize ConditionBAgent with an LLMProvider."""
        self.provider = provider

    def propose_action(
        self,
        context: ContextResult,
        inspector: Optional[ObjectInspector] = None,
    ) -> AgentResponse:
        """Send state snapshot and tool schemas to provider, parse tool calls, and return AgentResponse."""
        # 1. Validate plan version
        plan_version = _extract_plan_version(context.serialized_context)
        if plan_version is None:
            return AgentResponse(
                proposal=None,
                rationale="Invalid or missing plan_version in context.",
                raw_output="",
                metadata={"status": "invalid_plan_version", "condition_id": "B"},
            )

        # 2. Prepare tools and conversation messages
        tools = get_conventional_tool_definitions()
        messages = [
            LLMMessage(role="system", content=context.system_prompt),
            LLMMessage(
                role="user",
                content=(
                    f"User Request:\n{context.task_prompt}\n\n"
                    f"Current Schedule and State Snapshot:\n{context.serialized_context}"
                ),
            ),
        ]

        completion = self.provider.complete(messages=messages, tools=tools, tool_choice="auto")
        raw_output = completion.message.content or ""

        base_meta: Dict[str, Any] = {
            "condition_id": "B",
            "plan_version": plan_version,
            "usage": usage_to_dict(completion.usage),
            "model_identifier": completion.model_identifier,
            "system_fingerprint": completion.system_fingerprint,
            "provider_request_ids": (completion.provider_request_id,) if completion.provider_request_id else (),
            "per_request_latencies_ms": (completion.latency_ms,) if completion.latency_ms else (),
        }

        # 3. Check for tool calls
        tool_calls = completion.message.tool_calls
        if not tool_calls:
            content = completion.message.content or ""
            is_clarify = bool(content.strip()) and (
                "clarif" in content.lower()
                or "question" in content.lower()
                or "which" in content.lower()
                or "unclear" in content.lower()
                or "ambiguous" in content.lower()
                or "specify" in content.lower()
            )
            return AgentResponse(
                proposal=None,
                rationale=content or "No tool call was emitted by the model.",
                raw_output=raw_output,
                clarification_message=content.strip() if is_clarify else None,
                metadata={
                    **base_meta,
                    "status": "clarification_requested" if is_clarify else "no_tool_call",
                },
            )

        # 4. Support multiple returned tool calls without silently dropping any
        all_calls_summary: List[Dict[str, Any]] = []
        for tc in tool_calls:
            cid = tc.get("call_id") or tc.get("id") or ""
            fn_dict = tc.get("function", {})
            name = fn_dict.get("name") or tc.get("name", "")
            args_str = fn_dict.get("arguments") or tc.get("arguments", "{}")
            all_calls_summary.append({
                "call_id": cid,
                "name": name,
                "arguments": args_str,
            })

        # Explicitly reject unsupported multi-action responses
        if len(tool_calls) > 1:
            return AgentResponse(
                proposal=None,
                rationale=(
                    f"Multiple tool calls ({len(tool_calls)}) returned in a single turn. "
                    "Multi-action responses are unsupported under the atomic ActionBroker contract; "
                    "exactly one tool call is required."
                ),
                raw_output=raw_output,
                metadata={
                    **base_meta,
                    "status": "unsupported_multiple_tool_calls",
                    "tool_calls_count": len(tool_calls),
                    "all_tool_calls": all_calls_summary,
                },
            )

        # Process the single tool call
        primary_call = all_calls_summary[0]
        call_id = primary_call["call_id"]
        tool_name = primary_call["name"]
        raw_args = primary_call["arguments"]

        # Parse JSON arguments
        if isinstance(raw_args, str):
            try:
                args_dict = json.loads(raw_args)
            except json.JSONDecodeError as err:
                return AgentResponse(
                    proposal=None,
                    rationale=f"Malformed tool call arguments JSON: {err}",
                    raw_output=raw_output,
                    metadata={
                        **base_meta,
                        "status": "malformed_tool_arguments",
                        "call_id": call_id,
                        "tool_calls_count": len(tool_calls),
                        "all_tool_calls": all_calls_summary,
                    },
                )
        elif isinstance(raw_args, dict):
            args_dict = raw_args
        else:
            args_dict = {}

        # 5. Validate arguments using Stage 3 Pydantic schemas
        if tool_name == "replace_meal":
            try:
                validated_args = ReplaceMealArgs.model_validate(args_dict)
            except ValidationError as val_err:
                return AgentResponse(
                    proposal=None,
                    rationale=f"Invalid arguments for replace_meal: {val_err}",
                    raw_output=raw_output,
                    metadata={
                        **base_meta,
                        "status": "invalid_arguments",
                        "error": str(val_err),
                        "tool_name": tool_name,
                        "call_id": call_id,
                        "tool_calls_count": len(tool_calls),
                        "all_tool_calls": all_calls_summary,
                    },
                )

            args_payload = {
                "day": validated_args.day,
                "meal_type": validated_args.meal_type,
                "recipe_id": validated_args.recipe_id,
            }
            if validated_args.servings is not None:
                args_payload["servings"] = validated_args.servings

            proposal = ActionProposal(
                operation=OperationType.REPLACE_MEAL,
                arguments=args_payload,
                reason=f"Tool call replace_meal({validated_args.day}, {validated_args.meal_type}, {validated_args.recipe_id})",
                expected_plan_version=plan_version,
            )

        elif tool_name == "change_servings":
            try:
                validated_serv = ChangeServingsArgs.model_validate(args_dict)
            except ValidationError as val_err:
                return AgentResponse(
                    proposal=None,
                    rationale=f"Invalid arguments for change_servings: {val_err}",
                    raw_output=raw_output,
                    metadata={
                        **base_meta,
                        "status": "invalid_arguments",
                        "error": str(val_err),
                        "tool_name": tool_name,
                        "call_id": call_id,
                        "tool_calls_count": len(tool_calls),
                        "all_tool_calls": all_calls_summary,
                    },
                )

            proposal = ActionProposal(
                operation=OperationType.CHANGE_SERVINGS,
                arguments={
                    "day": validated_serv.day,
                    "meal_type": validated_serv.meal_type,
                    "servings": validated_serv.servings,
                },
                reason=f"Tool call change_servings({validated_serv.day}, {validated_serv.meal_type}, {validated_serv.servings})",
                expected_plan_version=plan_version,
            )

        elif tool_name == "request_clarification":
            q = (
                args_dict.get("question")
                or args_dict.get("clarification")
                or "Clarification requested."
            )
            return AgentResponse(
                proposal=None,
                rationale=f"Clarification requested: {q}",
                raw_output=raw_output,
                clarification_message=q,
                metadata={
                    **base_meta,
                    "status": "clarification_requested",
                },
            )

        else:
            return AgentResponse(
                proposal=None,
                rationale=f"Unknown or unsupported tool name: '{tool_name}'.",
                raw_output=raw_output,
                metadata={
                    **base_meta,
                    "status": "unknown_tool",
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "tool_calls_count": len(tool_calls),
                    "all_tool_calls": all_calls_summary,
                },
            )

        return AgentResponse(
            proposal=proposal,
            rationale=proposal.reason,
            raw_output=raw_output or proposal.model_dump_json(),
            metadata={
                **base_meta,
                "status": "ok",
                "tool_name": tool_name,
                "call_id": call_id,
                "tool_calls_count": len(tool_calls),
                "all_tool_calls": all_calls_summary,
            },
        )

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
