"""Abstract agent protocols, response contracts, and inspection interfaces.

Defines the common contracts implemented by both offline test agents (MockAgent)
and future real LLM adapters (Stage 6). Enforces the strict trust boundary:
agents propose actions via structured AgentResponse objects; they never mutate
authoritative state directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Protocol, runtime_checkable

from shopconnect_env.context.base import ContextResult
from shopconnect_env.validation.schemas import ActionProposal


@runtime_checkable
class ObjectInspector(Protocol):
    """Narrow, read-only inspection protocol implemented by Condition C context builders.

    The MockAgent (and future LLM adapters) use this protocol to query sub-objects
    selectively on demand without receiving monolithic state dumps.
    """

    def inspect_slot(self, day: str, meal_type: str) -> Dict[str, Any]:
        """Query meal details assigned to a specific day and slot."""
        ...

    def inspect_recipe(self, recipe_id: str) -> Dict[str, Any]:
        """Query high-level metadata, tags, servings, and costs for a recipe."""
        ...

    def inspect_recipe_ingredients(self, recipe_id: str) -> Dict[str, Any]:
        """Query detailed ingredient requirements and quantities for a recipe."""
        ...

    def inspect_ingredient(self, ingredient_id: str) -> Dict[str, Any]:
        """Query canonical ingredient classification, units, and allergen tags."""
        ...

    def inspect_plan(self) -> Dict[str, Any]:
        """Query active plan metadata, plan_version, and assigned meal count."""
        ...

    def inspect_preferences(self) -> Dict[str, Any]:
        """Query household dietary constraints and weekly budget limits."""
        ...

    def inspect(
        self,
        target_type: str,
        target_id: Optional[str] = None,
        day: Optional[str] = None,
        meal_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Unified inspection query dispatcher matching InspectObjectArgs query schema."""
        ...


def _freeze_value(val: Any) -> Any:
    """Recursively freeze mappings into MappingProxyType and sequences into tuples.

    Enforces deep immutability and defensive copying.
    Scalars (str, int, float, bool, None) remain unchanged.
    """
    if isinstance(val, Mapping):
        return MappingProxyType({k: _freeze_value(v) for k, v in val.items()})
    if isinstance(val, (list, tuple)):
        return tuple(_freeze_value(item) for item in val)
    if isinstance(val, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in val)
    return val


def _unfreeze_value(val: Any) -> Any:
    """Recursively convert MappingProxyType to dict and tuples to lists for serialization."""
    if isinstance(val, Mapping):
        return {k: _unfreeze_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_unfreeze_value(item) for item in val]
    if isinstance(val, (set, frozenset)):
        return [_unfreeze_value(item) for item in val]
    return val


@dataclass(frozen=True)
class AgentResponse:
    """Standardized response container emitted by any agent (Mock or LLM).

    Attributes:
        proposal: The structured ActionProposal, or None if the agent could
                  not formulate an actionable proposal.
        rationale: Brief human-readable action explanation.
        raw_output: Deterministic structured string emitted by the agent
                    (e.g. JSON string of the proposal or error message).
        inspection_calls_made: Total number of selective inspection queries
                               issued during action formulation (Condition C).
        metadata: Read-only mapping of diagnostic metadata (e.g. matched grammar rule,
                  condition_id, plan_version used). Protected from external modification
                  via recursive defensive copying and MappingProxyType freezing.
    """

    proposal: Optional[ActionProposal]
    rationale: str = ""
    raw_output: str = ""
    inspection_calls_made: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforce recursive immutability and defensive copying for metadata."""
        if isinstance(self.metadata, Mapping):
            frozen = _freeze_value(self.metadata)
        else:
            frozen = MappingProxyType({})
        object.__setattr__(self, "metadata", frozen)

    @property
    def has_proposal(self) -> bool:
        """Return True if an actionable ActionProposal is present."""
        return self.proposal is not None

    @property
    def metadata_dict(self) -> Dict[str, Any]:
        """Return an unfrozen, JSON-serializable dictionary copy of metadata."""
        return _unfreeze_value(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to a plain, JSON-serializable dictionary.

        Unfreezes nested MappingProxyType and tuple objects into standard dicts and lists.
        """
        return {
            "proposal": self.proposal.model_dump(mode="json") if self.proposal is not None else None,
            "rationale": self.rationale,
            "raw_output": self.raw_output,
            "inspection_calls_made": self.inspection_calls_made,
            "metadata": self.metadata_dict,
        }


class BaseAgent(ABC):
    """Abstract base protocol for meal planning agents."""

    @abstractmethod
    def propose_action(
        self,
        context: ContextResult,
        inspector: Optional[ObjectInspector] = None,
    ) -> AgentResponse:
        """Formulate an ActionProposal given a standardized ContextResult.

        The task instruction is read strictly from context.task_prompt.
        In Condition C, inspector provides read-only progressive disclosure.

        Args:
            context: Standardized ContextResult carrying task_prompt, system_prompt,
                     serialized_context, and signatures.
            inspector: Optional read-only query interface (supplied in Condition C).

        Returns:
            An AgentResponse containing the structured ActionProposal and metadata.
        """
        pass
