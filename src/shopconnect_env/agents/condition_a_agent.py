"""Condition A: Full-State Regeneration Agent.

Implements the baseline interaction pattern where the agent receives the complete
serialized weekly schedule and regenerates all 21 slots.

Strict invariants enforced:
1. Validates all 21 unique coordinates (Monday through Sunday x 3 meals).
2. Enforces assigned (valid catalog recipe_id, 1 <= servings <= 20) and unfilled
   (recipe_id=None, servings=0) slot representations.
3. Detects all schedule differences and records raw non-target drift before rejection.
4. Strictly rejects candidates if any non-target slot is modified (Drift Rejection).
5. Deterministically translates valid target changes into ActionProposal:
   - Recipe ID changed: REPLACE_MEAL
   - Only servings changed: CHANGE_SERVINGS
   - No changes: REJECT (target_not_modified)
6. Submits proposals through the existing ActionBroker without bypassing validation.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from shopconnect_env.agents.base import AgentResponse, BaseAgent, ObjectInspector
from shopconnect_env.agents.providers import LLMMessage, LLMProvider, usage_to_dict
from shopconnect_env.context.base import ContextResult
from shopconnect_env.domain.models import DAYS_OF_WEEK, MEAL_TYPES, RejectionCategory, make_slot_key
from shopconnect_env.validation.broker import ActionBroker, ActionResult, ActionStatus
from shopconnect_env.validation.schemas import ActionProposal, OperationType


# =============================================================================
# CONDITION A RESPONSE ENVELOPES (PYDANTIC V2 DISCRIMINATED UNION)
# =============================================================================

class ScheduleSlotModel(BaseModel):
    """Pydantic model representing a single schedule slot in Condition A."""

    model_config = ConfigDict(extra="forbid")

    day: str
    meal_type: str
    status: Literal["assigned", "unfilled"]
    recipe_id: Optional[str] = None
    servings: int

    @field_validator("day")
    @classmethod
    def validate_day(cls, v: str) -> str:
        norm = v.strip().capitalize()
        if norm not in DAYS_OF_WEEK:
            raise ValueError(f"Invalid day: '{v}'. Must be one of {DAYS_OF_WEEK}.")
        return norm

    @field_validator("meal_type")
    @classmethod
    def validate_meal_type(cls, v: str) -> str:
        norm = v.strip().lower()
        if norm not in MEAL_TYPES:
            raise ValueError(f"Invalid meal_type: '{v}'. Must be one of {MEAL_TYPES}.")
        return norm

    @field_validator("servings")
    @classmethod
    def validate_servings(cls, v: int) -> int:
        if v < 0 or v > 20:
            raise ValueError(f"servings must be between 0 and 20, got {v}.")
        return v


class ConditionACommitEnvelope(BaseModel):
    """Envelope emitted when Condition A commits a 21-slot regenerated meal plan."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["COMMIT"]
    schedule: List[ScheduleSlotModel]

    @field_validator("schedule")
    @classmethod
    def validate_schedule_slots(cls, slots: List[ScheduleSlotModel]) -> List[ScheduleSlotModel]:
        if len(slots) != 21:
            raise ValueError(f"Invalid slot count: expected 21 slots, got {len(slots)}.")
        coords = set()
        for s in slots:
            coord = (s.day, s.meal_type)
            if coord in coords:
                raise ValueError(f"Duplicate slot coordinate detected: {coord[0]} {coord[1]}.")
            coords.add(coord)
            if s.status == "assigned":
                if not s.recipe_id or not s.recipe_id.strip():
                    raise ValueError(f"Assigned slot {coord} must specify a valid recipe_id.")
                if s.servings < 1 or s.servings > 20:
                    raise ValueError(f"Assigned slot {coord} servings must be 1-20, got {s.servings}.")
            elif s.status == "unfilled":
                if s.recipe_id is not None:
                    raise ValueError(f"Unfilled slot {coord} must have null recipe_id, got {s.recipe_id!r}.")
                if s.servings != 0:
                    raise ValueError(f"Unfilled slot {coord} must have servings=0, got {s.servings}.")
        for d in DAYS_OF_WEEK:
            for m in MEAL_TYPES:
                if (d, m) not in coords:
                    raise ValueError(f"Missing slot coordinate: {d} {m}.")
        return slots


class ConditionARejectEnvelope(BaseModel):
    """Envelope emitted when Condition A rejects an invalid or unsafe request."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["REJECT"]
    rejection_category: RejectionCategory
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("reason cannot be empty or whitespace.")
        return s


class ConditionAClarifyEnvelope(BaseModel):
    """Envelope emitted when Condition A requests user clarification."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["CLARIFY"]
    clarification_question: str = Field(min_length=1)

    @field_validator("clarification_question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("clarification_question cannot be empty or whitespace.")
        return s


ConditionAEnvelope = Annotated[
    Union[ConditionACommitEnvelope, ConditionARejectEnvelope, ConditionAClarifyEnvelope],
    Field(discriminator="decision"),
]

ConditionAEnvelopeAdapter: TypeAdapter[Any] = TypeAdapter(ConditionAEnvelope)


def _extract_plan_version(serialized_context: str) -> Optional[int]:
    """Strictly extract and validate integer plan_version >= 1 from context JSON."""
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


def _extract_original_schedule(serialized_context: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Extract canonical mapping of (day, meal_type) -> slot_dict from context JSON."""
    try:
        data = json.loads(serialized_context)
        schedule = data.get("schedule", [])
    except (json.JSONDecodeError, TypeError):
        return {}

    slot_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for slot in schedule:
        day = slot.get("day")
        meal_type = slot.get("meal_type")
        if day and meal_type:
            slot_map[(day, meal_type)] = slot
    return slot_map


def _extract_allowed_recipes(serialized_context: str) -> Optional[Set[str]]:
    """Extract all valid recipe IDs present in the context catalog."""
    try:
        data = json.loads(serialized_context)
        recipes = data.get("available_recipes", [])
        return {r["recipe_id"] for r in recipes if "recipe_id" in r}
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_target_coordinates(task_prompt: str, metadata: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Determine expected target (day, meal_type) from metadata or task prompt regex."""
    meta_day = metadata.get("target_day")
    meta_slot = metadata.get("target_meal_type")
    if meta_day and meta_slot:
        return (str(meta_day).capitalize(), str(meta_slot).lower())

    # Regex detection for days and meal types
    day_match = re.search(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", task_prompt, re.IGNORECASE)
    slot_match = re.search(r"\b(breakfast|lunch|dinner)\b", task_prompt, re.IGNORECASE)

    target_day = day_match.group(1).capitalize() if day_match else None
    target_slot = slot_match.group(1).lower() if slot_match else None
    return (target_day, target_slot)


def _parse_regenerated_json(raw_text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse JSON schedule from raw text, handling markdown fences if present."""
    text = raw_text.strip()
    # Strip markdown code blocks
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "schedule" in data and isinstance(data["schedule"], list):
            return data["schedule"]
        if "meals" in data and isinstance(data["meals"], list):
            return data["meals"]
    return None


def _validate_slot_representation(
    schedule: List[Dict[str, Any]],
    allowed_recipe_ids: Optional[Set[str]] = None,
) -> Tuple[Optional[Dict[Tuple[str, str], Dict[str, Any]]], Optional[str]]:
    """Validate that the schedule satisfies the 21-slot representation rules.

    Returns:
        (slot_map, None) on success, or (None, error_reason) on failure.
    """
    if len(schedule) != 21:
        return (None, f"Invalid slot count: expected 21 slots, got {len(schedule)}.")

    slot_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for slot in schedule:
        if not isinstance(slot, dict):
            return (None, "Invalid slot representation: slot must be a JSON object.")

        day = slot.get("day")
        meal_type = slot.get("meal_type")

        if not day or not isinstance(day, str) or day.capitalize() not in DAYS_OF_WEEK:
            return (None, f"Invalid or missing day in slot: {day!r}.")
        if not meal_type or not isinstance(meal_type, str) or meal_type.lower() not in MEAL_TYPES:
            return (None, f"Invalid or missing meal_type in slot: {meal_type!r}.")

        coord = (day.capitalize(), meal_type.lower())
        if coord in slot_map:
            return (None, f"Duplicate slot coordinate detected: {coord[0]} {coord[1]}.")

        status = slot.get("status")
        recipe_id = slot.get("recipe_id")
        servings = slot.get("servings")

        # Assigned slot rules
        if status == "assigned":
            if not recipe_id or not isinstance(recipe_id, str):
                return (None, f"Assigned slot {coord} must specify a valid recipe_id string.")
            if allowed_recipe_ids is not None and recipe_id not in allowed_recipe_ids:
                return (None, f"Recipe ID '{recipe_id}' does not exist in catalog.")
            if isinstance(servings, bool) or not isinstance(servings, int) or servings < 1 or servings > 20:
                return (None, f"Assigned slot {coord} must have integer servings between 1 and 20, got {servings!r}.")

        # Unfilled slot rules
        elif status == "unfilled":
            if recipe_id is not None:
                return (None, f"Unfilled slot {coord} must have null recipe_id, got {recipe_id!r}.")
            if servings != 0:
                return (None, f"Unfilled slot {coord} must have servings=0, got {servings!r}.")

        else:
            return (None, f"Invalid slot status for {coord}: expected 'assigned' or 'unfilled', got {status!r}.")

        slot_map[coord] = {
            "day": coord[0],
            "meal_type": coord[1],
            "status": status,
            "recipe_id": recipe_id,
            "servings": servings,
        }

    # Ensure all 21 canonical coordinates are populated
    for d in DAYS_OF_WEEK:
        for m in MEAL_TYPES:
            if (d, m) not in slot_map:
                return (None, f"Missing slot coordinate: {d} {m}.")

    return (slot_map, None)


class ConditionAAgent(BaseAgent):
    """Full-state regeneration agent interfacing with LLMProvider under Condition A.

    Enforces:
    - 21-slot complete regeneration parsing.
    - Assigned vs. unfilled slot validation.
    - Pre-sanitization raw drift recording.
    - Non-target drift rejection.
    - Deterministic translation to ActionProposal.
    - Safe execution via ActionBroker.
    """

    def __init__(self, provider: LLMProvider) -> None:
        """Initialize ConditionAAgent with a provider."""
        self.provider = provider

    def propose_action(
        self,
        context: ContextResult,
        inspector: Optional[ObjectInspector] = None,
    ) -> AgentResponse:
        """Execute full-state prompt, parse 21-slot schedule, detect drift, and formulate ActionProposal."""
        # 1. Validate plan version
        plan_version = _extract_plan_version(context.serialized_context)
        if plan_version is None:
            return AgentResponse(
                proposal=None,
                rationale="Invalid or missing plan_version in context.",
                raw_output="",
                metadata={"status": "invalid_plan_version", "condition_id": "A"},
            )

        orig_schedule = _extract_original_schedule(context.serialized_context)
        allowed_recipes = _extract_allowed_recipes(context.serialized_context)
        target_day, target_slot = _extract_target_coordinates(context.task_prompt, context.metadata)

        # 2. Invoke provider
        messages = [
            LLMMessage(role="system", content=context.system_prompt),
            LLMMessage(
                role="user",
                content=(
                    f"User Request:\n{context.task_prompt}\n\n"
                    f"Current Complete State:\n{context.serialized_context}\n\n"
                    "Emit a single JSON object conforming to the Condition A response schema:\n"
                    "- COMMIT: {\"decision\": \"COMMIT\", \"schedule\": [<21 slots>]}\n"
                    "- REJECT: {\"decision\": \"REJECT\", \"rejection_category\": \"<category>\", \"reason\": \"<explanation>\"}\n"
                    "- CLARIFY: {\"decision\": \"CLARIFY\", \"clarification_question\": \"<question>\"}\n"
                    "For COMMIT, every slot must have: day, meal_type, status ('assigned' or 'unfilled'), "
                    "recipe_id, and servings (1-20 for assigned, 0 for unfilled)."
                ),
            ),
        ]

        completion = self.provider.complete(messages=messages)
        raw_text = completion.message.content or ""

        base_meta: Dict[str, Any] = {
            "condition_id": "A",
            "plan_version": plan_version,
            "usage": usage_to_dict(completion.usage),
            "model_identifier": completion.model_identifier,
            "system_fingerprint": completion.system_fingerprint,
            "provider_request_ids": (completion.provider_request_id,) if completion.provider_request_id else (),
            "per_request_latencies_ms": (completion.latency_ms,) if completion.latency_ms else (),
        }

        # 3. Clean raw output and parse JSON
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            clean_lines = clean_text.splitlines()
            if clean_lines[0].startswith("```"):
                clean_lines = clean_lines[1:]
            if clean_lines and clean_lines[-1].startswith("```"):
                clean_lines = clean_lines[:-1]
            clean_text = "\n".join(clean_lines).strip()

        parsed_raw = None
        try:
            parsed_raw = json.loads(clean_text)
        except Exception:
            pass

        # 3a. Condition A Pydantic v2 Discriminated Envelope Handling
        raw_slots: Optional[List[Dict[str, Any]]] = None
        if isinstance(parsed_raw, dict) and "decision" in parsed_raw:
            try:
                envelope = ConditionAEnvelopeAdapter.validate_python(parsed_raw)
            except ValidationError as val_err:
                return AgentResponse(
                    proposal=None,
                    rationale=f"Envelope validation error: {val_err}",
                    raw_output=raw_text,
                    metadata={
                        **base_meta,
                        "status": "envelope_validation_error",
                        "error": str(val_err),
                    },
                )

            if envelope.decision == "REJECT":
                return AgentResponse(
                    proposal=None,
                    rationale=envelope.reason,
                    raw_output=raw_text,
                    metadata={
                        **base_meta,
                        "status": "agent_rejected",
                        "rejection_category": envelope.rejection_category.value,
                        "agent_initiated_rejection": True,
                    },
                )
            elif envelope.decision == "CLARIFY":
                return AgentResponse(
                    proposal=None,
                    rationale=f"Clarification requested: {envelope.clarification_question}",
                    raw_output=raw_text,
                    clarification_message=envelope.clarification_question,
                    metadata={
                        **base_meta,
                        "status": "clarification_requested",
                    },
                )
            elif envelope.decision == "COMMIT":
                raw_slots = [
                    {
                        "day": s.day,
                        "meal_type": s.meal_type,
                        "status": s.status,
                        "recipe_id": s.recipe_id,
                        "servings": s.servings,
                    }
                    for s in envelope.schedule
                ]

        # 3b. Legacy dictionary clarification handling
        if raw_slots is None and isinstance(parsed_raw, dict):
            clar_msg = (
                parsed_raw.get("clarification")
                or parsed_raw.get("clarification_request")
                or parsed_raw.get("question")
            )
            if clar_msg:
                return AgentResponse(
                    proposal=None,
                    rationale=f"Clarification requested: {clar_msg}",
                    raw_output=raw_text,
                    clarification_message=str(clar_msg),
                    metadata={
                        **base_meta,
                        "status": "clarification_requested",
                    },
                )

        # 3c. Free-text clarification detection fallback
        if (
            raw_slots is None
            and not raw_text.strip().startswith("[")
            and not raw_text.strip().startswith("{")
            and not raw_text.strip().startswith("```")
            and any(p in raw_text.lower() for p in ["clarif", "could you specify", "please specify", "unclear which meal", "which meal slot"])
        ):
            return AgentResponse(
                proposal=None,
                rationale=f"Clarification requested: {raw_text.strip()}",
                raw_output=raw_text,
                clarification_message=raw_text.strip(),
                metadata={
                    **base_meta,
                    "status": "clarification_requested",
                },
            )

        if raw_slots is None:
            raw_slots = _parse_regenerated_json(raw_text)

        if raw_slots is None:
            return AgentResponse(
                proposal=None,
                rationale="Failed to parse regenerated schedule: invalid JSON structure.",
                raw_output=raw_text,
                metadata={
                    **base_meta,
                    "status": "json_parse_error",
                },
            )

        # 4. Validate slot representations and uniqueness
        slot_map, validation_err = _validate_slot_representation(raw_slots, allowed_recipes)
        if validation_err or slot_map is None:
            return AgentResponse(
                proposal=None,
                rationale=validation_err or "Invalid slot representation.",
                raw_output=raw_text,
                metadata={
                    **base_meta,
                    "status": "slot_validation_error",
                    "error": validation_err,
                },
            )

        # 5. Semantic diffing & Raw drift measurement BEFORE any sanitization
        modified_slots: List[Dict[str, Any]] = []
        for coord, regen_slot in slot_map.items():
            orig_slot = orig_schedule.get(coord, {})
            orig_recipe = orig_slot.get("recipe_id")
            orig_servings = orig_slot.get("servings")
            orig_status = orig_slot.get("status")

            if (
                regen_slot["recipe_id"] != orig_recipe
                or regen_slot["servings"] != orig_servings
                or regen_slot["status"] != orig_status
            ):
                modified_slots.append({
                    "day": coord[0],
                    "meal_type": coord[1],
                    "original_recipe_id": orig_recipe,
                    "original_servings": orig_servings,
                    "new_recipe_id": regen_slot["recipe_id"],
                    "new_servings": regen_slot["servings"],
                    "new_status": regen_slot["status"],
                })

        # Separate target changes from non-target changes
        target_coord = (target_day, target_slot) if (target_day and target_slot) else None
        non_target_changes = [
            s for s in modified_slots
            if (s["day"], s["meal_type"]) != target_coord
        ]

        raw_drift_count = len(non_target_changes)
        raw_drift_slots = [(s["day"], s["meal_type"]) for s in non_target_changes]

        # 6. Drift rejection: if non-target changes exist, reject before broker execution
        if raw_drift_count > 0:
            return AgentResponse(
                proposal=None,
                rationale=(
                    f"Unintended state drift detected: {raw_drift_count} non-target slot(s) modified: {raw_drift_slots}."
                ),
                raw_output=raw_text,
                metadata={
                    **base_meta,
                    "status": "unintended_drift",
                    "unintended_mutations_count": raw_drift_count,
                    "unintended_slots": raw_drift_slots,
                    "total_modifications": len(modified_slots),
                },
            )

        # 7. No-op rejection
        if len(modified_slots) == 0:
            return AgentResponse(
                proposal=None,
                rationale="No schedule changes detected in regenerated plan.",
                raw_output=raw_text,
                metadata={
                    **base_meta,
                    "status": "target_not_modified",
                    "unintended_mutations_count": 0,
                },
            )

        # 8. Single target change translation
        target_mod = modified_slots[0]
        t_day = target_mod["day"]
        t_slot = target_mod["meal_type"]
        new_rec = target_mod["new_recipe_id"]
        new_serv = target_mod["new_servings"]
        orig_rec = target_mod["original_recipe_id"]
        orig_serv = target_mod["original_servings"]

        # Recipe ID changed: REPLACE_MEAL
        if new_rec != orig_rec and new_rec is not None:
            proposal = ActionProposal(
                operation=OperationType.REPLACE_MEAL,
                arguments={
                    "day": t_day,
                    "meal_type": t_slot,
                    "recipe_id": new_rec,
                    "servings": new_serv,
                },
                reason=f"Regenerated schedule replaces {t_day} {t_slot} with {new_rec}.",
                expected_plan_version=plan_version,
            )
            return AgentResponse(
                proposal=proposal,
                rationale=proposal.reason,
                raw_output=raw_text,
                metadata={
                    **base_meta,
                    "status": "ok",
                    "operation": "replace_meal",
                    "unintended_mutations_count": 0,
                },
            )

        # Only servings changed: CHANGE_SERVINGS
        if new_rec == orig_rec and new_serv != orig_serv and new_serv is not None:
            proposal = ActionProposal(
                operation=OperationType.CHANGE_SERVINGS,
                arguments={
                    "day": t_day,
                    "meal_type": t_slot,
                    "servings": new_serv,
                },
                reason=f"Regenerated schedule scales servings for {t_day} {t_slot} to {new_serv}.",
                expected_plan_version=plan_version,
            )
            return AgentResponse(
                proposal=proposal,
                rationale=proposal.reason,
                raw_output=raw_text,
                metadata={
                    **base_meta,
                    "status": "ok",
                    "operation": "change_servings",
                    "unintended_mutations_count": 0,
                },
            )

        # Catch-all
        return AgentResponse(
            proposal=None,
            rationale="Unrecognized modification pattern in target slot.",
            raw_output=raw_text,
            metadata={
                **base_meta,
                "status": "unsupported_modification",
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
