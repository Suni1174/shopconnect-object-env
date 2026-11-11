"""Evaluation metrics, cost calculation, latency percentiles, and summary aggregations.

Part of Stage 6 Phase 6.4:
- Versioned pricing configuration and verified cost calculation.
- Granular task-level metrics (correctness, state safety, usage, latency, cost).
- Condition-level aggregate summary computations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
import math
from typing import Any, Dict, List, Optional

from shopconnect_env.agents.providers import LLMUsage


@dataclass(frozen=True)
class PricingConfig:
    """Versioned pricing configuration for LLM token billing.

    Rates are specified as Decimal values in currency per 1,000,000 tokens.
    If any required rate is None, pricing is considered unverified/unknown
    and monetary cost calculations evaluate strictly to None.
    """

    version: str
    source: str
    effective_date: str
    currency: str = "USD"
    input_cost_per_million: Optional[Decimal] = None
    cached_input_cost_per_million: Optional[Decimal] = None
    output_cost_per_million: Optional[Decimal] = None
    verification_date: Optional[str] = None
    model_name: Optional[str] = None
    pricing_mode: str = "Standard"
    source_url: Optional[str] = None
    cache_write_30m_cost_per_million: Optional[Decimal] = None
    cache_write_12h_cost_per_million: Optional[Decimal] = None
    max_context_tokens_threshold: int = 272_000
    long_context_input_cost_per_million: Optional[Decimal] = None
    long_context_output_cost_per_million: Optional[Decimal] = None
    cache_retention: Optional[str] = "30m"

    def is_complete(self) -> bool:
        """Return True if base input and output rates are defined."""
        return (
            self.input_cost_per_million is not None
            and self.output_cost_per_million is not None
        )


def calculate_cost(
    usage: Optional[LLMUsage],
    pricing: Optional[PricingConfig],
) -> Optional[Decimal]:
    """Calculate monetary cost from provider-reported usage and verified pricing config.

    Requirements & Invariants:
    1. If pricing is None or incomplete (missing input or output rate), returns None.
    2. If usage is None, not provider-reported, or simulated (e.g. MockLLMProvider), returns None.
    3. If required token counts (input_tokens or output_tokens) are None, returns None.
    4. Long-context guardrail: prompts exceeding max_context_tokens_threshold (272,000)
       require long_context_input_cost_per_million and long_context_output_cost_per_million;
       returns None if not configured to prevent underbilling.
    5. Does not double-count reasoning tokens: reasoning tokens are already included
       in output_tokens (as reported by OpenAI Responses API) and are billed at output rate.
    6. Does not double-count cached tokens or cache writes: ordinary input tokens are
       computed as: ordinary_input = input_tokens - cached_tokens - cache_write_tokens.
    7. Cache-write components:
       - 30-minute cache writes billed at cache_write_30m_cost_per_million.
       - 12-hour cache writes billed at cache_write_12h_cost_per_million.
       - When detailed breakdown is present, the two components must sum to cache_write_tokens.
       - If cache-write details are missing or inconsistent, returns None (unknown cost).
       - Never adds cache_write_tokens again after pricing 30m and 12h components.
    8. Uses exact Decimal arithmetic throughout, quantizing final cost to 6 decimal places.
    """
    if pricing is None or not pricing.is_complete():
        return None
    if usage is None or not usage.is_provider_reported:
        return None
    if getattr(usage, "usage_provenance", None) == "simulated":
        return None
    if usage.input_tokens is None or usage.output_tokens is None:
        return None

    # 1. Long-context threshold check
    threshold = pricing.max_context_tokens_threshold
    is_long_context = usage.input_tokens > threshold
    if is_long_context:
        if pricing.long_context_input_cost_per_million is None or pricing.long_context_output_cost_per_million is None:
            return None
        input_rate = pricing.long_context_input_cost_per_million
        output_rate = pricing.long_context_output_cost_per_million
    else:
        input_rate = pricing.input_cost_per_million
        output_rate = pricing.output_cost_per_million

    # 2. Cached tokens handling (never double-count cached tokens)
    raw_cached = usage.cached_input_tokens or 0
    if raw_cached > 0 and pricing.cached_input_cost_per_million is None:
        return None
    cached_tokens = min(raw_cached, usage.input_tokens)

    # 3. Cache write tokens handling & consistency check
    raw_cw_total = usage.cache_write_tokens
    raw_cw_30m = usage.cache_write_tokens_30m
    raw_cw_12h = usage.cache_write_tokens_12h

    has_cw_breakdown = (raw_cw_30m is not None or raw_cw_12h is not None)
    has_cw_total = (raw_cw_total is not None and raw_cw_total > 0)

    # If cache_write_tokens > 0 but breakdown is missing: cannot infer retention tier -> return None
    if has_cw_total and not has_cw_breakdown:
        return None

    cw_30m = raw_cw_30m or 0
    cw_12h = raw_cw_12h or 0
    breakdown_sum = cw_30m + cw_12h

    # If breakdown is present, verify consistency with reported total
    if has_cw_breakdown:
        if raw_cw_total is not None and breakdown_sum != raw_cw_total:
            # Breakdown sum does not match reported total -> inconsistent, return None
            return None
        total_cw_tokens = breakdown_sum
    else:
        total_cw_tokens = 0

    # Ensure cached_tokens + total_cw_tokens does not exceed total input_tokens
    if cached_tokens + total_cw_tokens > usage.input_tokens:
        return None

    # 4. Ordinary input tokens: input_tokens - cached_tokens - cache_write_tokens
    ordinary_input_tokens = usage.input_tokens - cached_tokens - total_cw_tokens
    per_million = Decimal("1000000")

    ordinary_input_cost = (Decimal(ordinary_input_tokens) / per_million) * input_rate

    cached_cost = Decimal("0")
    if cached_tokens > 0 and pricing.cached_input_cost_per_million is not None:
        cached_cost = (Decimal(cached_tokens) / per_million) * pricing.cached_input_cost_per_million

    # 5. Cache write costs (priced exclusively through 30m and 12h rates; never added again)
    cw_30m_cost = Decimal("0")
    if cw_30m > 0:
        if pricing.cache_write_30m_cost_per_million is None:
            return None
        cw_30m_cost = (Decimal(cw_30m) / per_million) * pricing.cache_write_30m_cost_per_million

    cw_12h_cost = Decimal("0")
    if cw_12h > 0:
        if pricing.cache_write_12h_cost_per_million is None:
            return None
        cw_12h_cost = (Decimal(cw_12h) / per_million) * pricing.cache_write_12h_cost_per_million

    # 6. Output cost (reasoning tokens already included in output_tokens; never add reasoning tokens separately)
    output_cost = (Decimal(usage.output_tokens) / per_million) * output_rate

    total = ordinary_input_cost + cached_cost + cw_30m_cost + cw_12h_cost + output_cost
    return total.quantize(Decimal("0.000001"))


def create_gpt6_luna_pricing(
    version: str = "2026-11-openai-standard-v1",
    source: str = "OpenAI Official Responses API Pricing Schedule (Standard Mode)",
    source_url: str = "https://developers.openai.com/api/docs/models/gpt-6-luna",
    effective_date: str = "2026-11-01",
    verification_date: Optional[str] = "2026-11-11",
    model_name: str = "gpt-6-luna",
    pricing_mode: str = "Standard",
    input_cost_per_million: Decimal = Decimal("0.10"),
    cached_input_cost_per_million: Decimal = Decimal("0.01"),
    output_cost_per_million: Decimal = Decimal("0.50"),
    currency: str = "USD",
    cache_write_30m_cost_per_million: Optional[Decimal] = Decimal("0.125"),
    cache_write_12h_cost_per_million: Optional[Decimal] = Decimal("0.25"),
    max_context_tokens_threshold: int = 272_000,
    long_context_input_cost_per_million: Optional[Decimal] = None,
    long_context_output_cost_per_million: Optional[Decimal] = None,
    cache_retention: Optional[str] = "30m",
) -> PricingConfig:
    """Create verified pricing configuration for OpenAI GPT-6 Luna under Standard processing.

    Documented rates per million tokens:
    - Input: $0.10
    - Cached input: $0.01
    - 30m cache writes: $0.125
    - 12h cache writes: $0.25
    - Output: $0.50

    Official references:
    - https://developers.openai.com/api/docs/models/gpt-6-luna
    - https://developers.openai.com/api/docs/pricing
    """
    if pricing_mode != "Standard":
        raise ValueError(
            f"Pricing mode '{pricing_mode}' is not supported by this configuration. "
            "These rates apply strictly to Standard processing and must not be silently "
            "applied to Batch, Flex, Fast, or other processing modes."
        )
    return PricingConfig(
        version=version,
        source=source,
        effective_date=effective_date,
        currency=currency,
        input_cost_per_million=input_cost_per_million,
        cached_input_cost_per_million=cached_input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
        verification_date=verification_date,
        model_name=model_name,
        pricing_mode=pricing_mode,
        source_url=source_url,
        cache_write_30m_cost_per_million=cache_write_30m_cost_per_million,
        cache_write_12h_cost_per_million=cache_write_12h_cost_per_million,
        max_context_tokens_threshold=max_context_tokens_threshold,
        long_context_input_cost_per_million=long_context_input_cost_per_million,
        long_context_output_cost_per_million=long_context_output_cost_per_million,
        cache_retention=cache_retention,
    )


def calculate_percentile(values: List[Any], percentile: float) -> Optional[float]:
    """Calculate percentile from a list of values using standard linear interpolation.

    Filters out None and non-numeric values. Returns None when no valid observations exist.

    Args:
        values: Sequence of numerical or optional values.
        percentile: Float between 0.0 and 100.0 (e.g. 50.0 for p50, 95.0 for p95).

    Returns:
        The calculated percentile value as a float, or None if values contains no valid observations.
    """
    valid_vals = []
    for v in values:
        if v is not None:
            try:
                fv = float(v)
                if not math.isnan(fv):
                    valid_vals.append(fv)
            except (ValueError, TypeError):
                continue

    if not valid_vals:
        return None
    sorted_vals = sorted(valid_vals)
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])

    # Method matching numpy/scipy percentile (linear interpolation)
    k = (n - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return float(round(d0 + d1, 3))


@dataclass(frozen=True)
class RunMetrics:
    """Granular, comprehensive metrics for a single task-condition run."""

    # Identity
    task_id: str
    fixture_id: str
    condition_id: str

    # Correctness
    task_success: bool
    expected_outcome: str  # COMMIT, REJECT, CLARIFY
    actual_outcome: str  # COMMIT, REJECT, CLARIFY, ERROR
    target_accuracy: bool
    false_rejection: bool
    rejection_category_match: Optional[bool] = None

    # State Safety
    non_target_drift_count: int = 0
    unintended_assignments: int = 0
    non_target_recipe_overwrites: int = 0
    committed_constraint_violations: int = 0
    plan_mutated_on_rejection: bool = False
    basket_mutated_on_rejection: bool = False

    # LLM Usage
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    cache_write_tokens_30m: Optional[int] = None
    cache_write_tokens_12h: Optional[int] = None
    is_provider_reported: bool = False
    model_requests: int = 1
    inspection_calls: int = 0

    # Latency & Cost
    wall_clock_latency_ms: float = 0.0
    per_request_latencies_ms: List[float] = field(default_factory=list)
    cost_usd: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a clean, JSON-serializable dictionary."""
        d = asdict(self)
        if self.cost_usd is not None:
            d["cost_usd"] = str(self.cost_usd)
        return d


@dataclass(frozen=True)
class AggregateMetrics:
    """Summary metrics aggregated across all runs for a specific experimental condition."""

    condition_id: str
    total_runs: int
    successful_runs: int
    task_success_rate: float
    target_accuracy_rate: float
    false_rejection_rate: float
    rejection_accuracy_rate: float
    clarification_accuracy_rate: float

    # State safety
    mean_drift_count: float
    total_unintended_assignments: int
    total_recipe_overwrites: int
    total_policy_violations: int

    # Usage
    mean_input_tokens: Optional[float]
    mean_output_tokens: Optional[float]
    mean_total_tokens: Optional[float]
    mean_model_requests: float
    mean_inspection_calls: float

    # Latency & Cost
    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]
    total_cost_usd: Optional[Decimal]

    def to_dict(self) -> Dict[str, Any]:
        """Convert aggregate summary to dictionary."""
        d = asdict(self)
        if self.total_cost_usd is not None:
            d["total_cost_usd"] = str(self.total_cost_usd)
        return d


def compute_aggregate_metrics(
    runs: List[RunMetrics],
    condition_id: str,
) -> AggregateMetrics:
    """Aggregate individual run metrics for a condition into a formal statistical summary."""
    cond_runs = [r for r in runs if r.condition_id == condition_id]
    total = len(cond_runs)
    if total == 0:
        return AggregateMetrics(
            condition_id=condition_id,
            total_runs=0,
            successful_runs=0,
            task_success_rate=0.0,
            target_accuracy_rate=0.0,
            false_rejection_rate=0.0,
            rejection_accuracy_rate=0.0,
            clarification_accuracy_rate=0.0,
            mean_drift_count=0.0,
            total_unintended_assignments=0,
            total_recipe_overwrites=0,
            total_policy_violations=0,
            mean_input_tokens=None,
            mean_output_tokens=None,
            mean_total_tokens=None,
            mean_model_requests=0.0,
            mean_inspection_calls=0.0,
            latency_p50_ms=None,
            latency_p95_ms=None,
            total_cost_usd=None,
        )

    successful = sum(1 for r in cond_runs if r.task_success)
    target_acc = sum(1 for r in cond_runs if r.target_accuracy)

    # Sub-category rates
    commit_runs = [r for r in cond_runs if r.expected_outcome == "COMMIT"]
    reject_runs = [r for r in cond_runs if r.expected_outcome == "REJECT"]
    clarify_runs = [r for r in cond_runs if r.expected_outcome == "CLARIFY"]

    false_rejections = sum(1 for r in commit_runs if r.false_rejection)
    false_rejection_rate = (false_rejections / len(commit_runs)) if commit_runs else 0.0

    reject_matches = sum(1 for r in reject_runs if r.rejection_category_match is True)
    rejection_accuracy_rate = (reject_matches / len(reject_runs)) if reject_runs else 0.0

    clarify_successes = sum(1 for r in clarify_runs if r.task_success)
    clarification_accuracy_rate = (clarify_successes / len(clarify_runs)) if clarify_runs else 0.0

    # State safety
    mean_drift = sum(r.non_target_drift_count for r in cond_runs) / total
    total_unintended = sum(r.unintended_assignments for r in cond_runs)
    total_overwrites = sum(r.non_target_recipe_overwrites for r in cond_runs)
    total_violations = sum(r.committed_constraint_violations for r in cond_runs)

    # Usage
    has_tokens = any(r.total_tokens is not None for r in cond_runs)
    mean_input = (
        sum(r.input_tokens for r in cond_runs if r.input_tokens is not None) / total
        if has_tokens
        else None
    )
    mean_output = (
        sum(r.output_tokens for r in cond_runs if r.output_tokens is not None) / total
        if has_tokens
        else None
    )
    mean_total = (
        sum(r.total_tokens for r in cond_runs if r.total_tokens is not None) / total
        if has_tokens
        else None
    )

    mean_requests = sum(r.model_requests for r in cond_runs) / total
    mean_inspections = sum(r.inspection_calls for r in cond_runs) / total

    # Latencies
    latencies = [r.wall_clock_latency_ms for r in cond_runs]
    p50 = calculate_percentile(latencies, 50.0)
    p95 = calculate_percentile(latencies, 95.0)

    # Costs
    has_cost = any(r.cost_usd is not None for r in cond_runs)
    total_cost = None
    if has_cost:
        total_cost = sum(
            (r.cost_usd for r in cond_runs if r.cost_usd is not None),
            Decimal("0"),
        )

    return AggregateMetrics(
        condition_id=condition_id,
        total_runs=total,
        successful_runs=successful,
        task_success_rate=round(successful / total, 4),
        target_accuracy_rate=round(target_acc / total, 4),
        false_rejection_rate=round(false_rejection_rate, 4),
        rejection_accuracy_rate=round(rejection_accuracy_rate, 4),
        clarification_accuracy_rate=round(clarification_accuracy_rate, 4),
        mean_drift_count=round(mean_drift, 3),
        total_unintended_assignments=total_unintended,
        total_recipe_overwrites=total_overwrites,
        total_policy_violations=total_violations,
        mean_input_tokens=round(mean_input, 2) if mean_input is not None else None,
        mean_output_tokens=round(mean_output, 2) if mean_output is not None else None,
        mean_total_tokens=round(mean_total, 2) if mean_total is not None else None,
        mean_model_requests=round(mean_requests, 2),
        mean_inspection_calls=round(mean_inspections, 2),
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        total_cost_usd=total_cost,
    )
