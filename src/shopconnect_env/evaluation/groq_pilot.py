"""Controlled Groq Cloud API pilot execution harness and guardrails for Protocol v2.

Stage 6 Protocol v2:
- Safe, bounded execution of exactly 3 frozen benchmark tasks across 3 conditions (9 runs).
- Selected tasks:
  * 1 COMMIT: task_01_commit_replace_wed_dinner (meal_replacement)
  * 1 REJECT: task_14_reject_allergen_peanut_toast (allergen_conflict)
  * 1 CLARIFY: task_22_clarify_missing_meal_slot (ambiguous_request)
- Strict maximum 28 total provider requests across the full pilot.
- Sequential execution only (zero parallelism).
- SDK retries disabled / single-attempt execution (max_retries=0).
- Output token ceiling enforced (max_output_tokens=2048).
- Model: qwen/qwen3.8-27b (Groq hosted).
- Dual confirmation flags (--live AND --confirm-live) strictly required for live API calls.
- Offline preflight validation and simulated dry-run modes with zero network access.
- Dedicated artifacts under artifacts/protocol_v2_pilot/groq_<run_id>/.
- Distinctly tagged: REAL_API_PILOT (GROQ) — LIMITED_SAMPLE — NOT_FULL_EXPERIMENT.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Ensure src/ is in sys.path for direct CLI script execution
_src_root = str(Path(__file__).resolve().parent.parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

# =============================================================================
# 0. ENVIRONMENT & PROJECT ROOT RESOLUTION
# =============================================================================

def get_project_root() -> Path:
    """Resolve the project root directory robustly using pathlib."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "requirements.txt").exists() or (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parents[3]


_PROJECT_ENV_LOADED: bool = False


def load_project_env(
    env_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
) -> bool:
    """Load environment variables from project-root .env without overriding existing variables."""
    global _PROJECT_ENV_LOADED
    if _PROJECT_ENV_LOADED and not force_reload and env_path is None:
        return True

    if load_dotenv is None:
        return False

    if env_path is not None:
        target = Path(env_path).resolve()
    else:
        root = get_project_root()
        target = root / ".env"

    if target.exists() and target.is_file():
        success = bool(load_dotenv(dotenv_path=target, override=False))
        if env_path is None:
            _PROJECT_ENV_LOADED = True
        return success
    return False


# Automatically load project-root .env at module import
load_project_env()

from pydantic import ValidationError

from shopconnect_env.agents.condition_a_agent import ConditionAEnvelopeAdapter
from shopconnect_env.agents.providers import (
    AuthenticationError,
    ConnectionError,
    GroqProvider,
    LLMCompletionResult,
    LLMMessage,
    LLMProvider,
    LLMUsage,
    MockLLMProvider,
    ProviderError,
    RateLimitError,
    TimeoutError,
    usage_to_dict,
)
from shopconnect_env.context.base import ContextRequest
from shopconnect_env.context.full_state import FullStateContextBuilder
from shopconnect_env.context.object_context import ObjectContextBuilder
from shopconnect_env.context.tool_context import ToolContextBuilder
from shopconnect_env.domain.models import MealPlan, RejectionCategory, load_catalog_from_json
from shopconnect_env.evaluation.metrics import (
    PricingConfig,
    RunMetrics,
    compute_aggregate_metrics,
    create_groq_free_pricing,
)
from shopconnect_env.evaluation.runner import (
    ExperimentRunner,
    PlanFixture,
    RunRecord,
    compute_sha256,
    load_plan_fixtures,
)
from shopconnect_env.evaluation.tasks import (
    BenchmarkTask,
    ExpectedOutcome,
    load_benchmark_tasks,
)


# =============================================================================
# 1. PILOT CONSTANTS & SPECIFICATIONS
# =============================================================================

PILOT_TASK_IDS: Tuple[str, ...] = (
    "task_01_commit_replace_wed_dinner",
    "task_14_reject_allergen_peanut_toast",
    "task_22_clarify_missing_meal_slot",
)

PILOT_CONDITIONS: Tuple[str, ...] = ("A", "B", "C")

MAX_PILOT_RUNS: int = 9  # 3 tasks * 3 conditions
PILOT_TASKS_COUNT: int = len(PILOT_TASK_IDS)  # 3

# Per-condition request budgets for Groq Protocol v2 pilot
MAX_REQUESTS_PER_CONDITION_A: int = 1
MAX_REQUESTS_PER_CONDITION_B: int = 1
PROTOCOL_V2_MAX_REQUESTS_PER_CONDITION_C: int = 6
PROTOCOL_V2_MAX_INSPECTIONS_PER_CONDITION_C: int = 5

# Maximum possible total requests: 3 tasks × (1 A + 1 B + 6 C) = 24 requests
PROTOCOL_V2_MAX_THEORETICAL_TOTAL_REQUESTS: int = PILOT_TASKS_COUNT * (
    MAX_REQUESTS_PER_CONDITION_A + MAX_REQUESTS_PER_CONDITION_B + PROTOCOL_V2_MAX_REQUESTS_PER_CONDITION_C
)

# Global hard ceiling adapted to Groq free plan quotas (ceiling 28)
PROTOCOL_V2_GLOBAL_REQUEST_CEILING: int = 28
GLOBAL_REQUEST_CEILING: int = PROTOCOL_V2_GLOBAL_REQUEST_CEILING
DEFAULT_OUTPUT_TOKEN_CEILING: int = 2048
PILOT_MODEL: str = "qwen/qwen3.8-27b"
DEFAULT_BASE_URL: str = "https://api.groq.com/openai/v1"
PILOT_TAG: str = "REAL_API_PILOT (GROQ) — LIMITED_SAMPLE — NOT_FULL_EXPERIMENT"

# Standalone One-Request Connection Test Specifications
CONNECTION_TEST_PROMPT: str = "Reply with exactly: SHOPCONNECT_GROQ_CONNECTION_OK"
EXPECTED_CONNECTION_TEST_TOKEN: str = "SHOPCONNECT_GROQ_CONNECTION_OK"
CONNECTION_TEST_OUTPUT_TOKEN_CEILING: int = 64
DEFAULT_CONNECTION_TEST_ARTIFACT_DIR: str = "artifacts/protocol_v2_pilot"

# Standalone One-Request Tool-Calling Smoke Test Specifications
INSPECT_PLAN_TOOL_DEFINITION: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_plan",
        "description": "Inspect active plan metadata, plan_version, and total assigned meals.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}
INSPECT_PLAN_TOOL_CHOICE: Dict[str, Any] = {
    "type": "function",
    "function": {"name": "inspect_plan"},
}
TOOL_SMOKE_TEST_PROMPT: str = "Inspect the current active meal plan metadata and plan version."
TOOL_SMOKE_TEST_OUTPUT_TOKEN_CEILING: int = 128
DEFAULT_TOOL_SMOKE_TEST_ARTIFACT_DIR: str = "artifacts/protocol_v2_pilot"

# Standalone One-Request Condition A JSON Smoke Test Specifications
JSON_SMOKE_TEST_SYSTEM_PROMPT: str = (
    "You are a meal planning assistant. Respond strictly with a JSON object conforming to the schema. "
    "Do not include any explanation or commentary outside the JSON."
)
JSON_SMOKE_TEST_USER_PROMPT: str = (
    "User Request:\n"
    "Please replace Thursday dinner with Thai Peanut Noodles.\n\n"
    "Current Household Constraints:\n"
    "- Severe peanut allergy registered for child.\n\n"
    "Because of the allergen conflict, reject this request.\n"
    "Emit a single JSON object conforming to the Condition A response schema:\n"
    '{"decision": "REJECT", "rejection_category": "allergen_conflict", "reason": "<non-empty explanation of the peanut allergen conflict>"}\n'
    "Valid rejection_category values include: allergen_conflict, dietary_mismatch, budget_overflow, invalid_recipe, no_op, other."
)
JSON_SMOKE_TEST_OUTPUT_TOKEN_CEILING: int = 256
DEFAULT_JSON_SMOKE_TEST_ARTIFACT_DIR: str = "artifacts/protocol_v2_pilot"

# Safe Rate Limiting Target and Free Tier Quotas for Groq
GROQ_FREE_TIER_ITPM_LIMIT: int = 7000
GROQ_FREE_TIER_RPM_LIMIT: int = 30
CONFIGURED_SAFE_TARGET_RPM: int = 12
MIN_REQUEST_INTERVAL_SECONDS: float = 12.0  # 12.0s minimum interval between request starts (safe for 5 RPM / 7k ITPM)


# =============================================================================
# 2. SEQUENTIAL RATE LIMITER
# =============================================================================

class SequentialRateLimiter:
    """Enforces a safe sequential interval, quota-aware pacing, and 60-second sliding-window token budget."""

    def __init__(
        self,
        min_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS,
        clock: Optional[Callable[[], float]] = None,
        sleeper: Optional[Callable[[float], None]] = None,
        window_seconds: float = 60.0,
        itpm_limit: int = 7000,
        reserve_tokens: int = 1000,
        enable_sliding_window: bool = True,
        printer: Optional[Callable[[str], None]] = None,
        verbose: bool = True,
    ) -> None:
        self.min_interval_seconds = min_interval_seconds
        self.clock = clock or time.monotonic
        self.sleeper = sleeper or time.sleep
        self.window_seconds = window_seconds
        self.itpm_limit = itpm_limit
        self.reserve_tokens = reserve_tokens
        self.effective_budget = max(0, itpm_limit - reserve_tokens)
        self.enable_sliding_window = enable_sliding_window
        self.printer = printer
        self.verbose = verbose

        self.last_request_start: Optional[float] = None
        self.total_sleep_seconds: float = 0.0
        self.sleep_events: List[float] = []
        self.sleep_records: List[Dict[str, Any]] = []

        # Sliding window history: deque of dicts {"id": int, "timestamp": float, "tokens": int, "status": str}
        self.window_entries: deque = deque()
        self._next_reservation_id: int = 1
        self.last_reservation_id: Optional[int] = None

        self.pending_quota_delay_seconds: float = 0.0
        self.last_remaining_tokens: Optional[int] = None
        self.last_reset_tokens_seconds: Optional[float] = None
        self.last_header_receipt_time: Optional[float] = None
        self.quota_deadline: Optional[float] = None
        self.retry_after_deadline: Optional[float] = None
        self.token_reset_deadline: Optional[float] = None
        self._last_processed_headers: Optional[Mapping[str, Any]] = None

    def prune_window(self, now: Optional[float] = None) -> None:
        """Expire entries older than the rolling window_seconds."""
        current_time = now if now is not None else self.clock()
        cutoff = current_time - self.window_seconds
        while self.window_entries and self.window_entries[0]["timestamp"] <= cutoff:
            self.window_entries.popleft()

    def get_window_tokens(self, now: Optional[float] = None) -> int:
        """Return the sum of active reserved and confirmed tokens in the current window."""
        self.prune_window(now)
        return sum(entry["tokens"] for entry in self.window_entries)

    def reserve_slot(self, estimated_tokens: int, now: Optional[float] = None) -> int:
        """Reserve token allocation before dispatching a request."""
        current_time = now if now is not None else self.clock()
        res_id = self._next_reservation_id
        self._next_reservation_id += 1
        self.window_entries.append({
            "id": res_id,
            "timestamp": current_time,
            "tokens": max(0, estimated_tokens),
            "status": "reserved",
        })
        self.last_reservation_id = res_id
        return res_id

    def reconcile_reservation(self, reservation_id: int, actual_tokens: int) -> None:
        """Reconcile estimated reservation with actual provider-reported input tokens.

        Cached input tokens are part of input tokens, not additional, so actual_tokens
        strictly reflects total input tokens consumed.
        """
        for entry in self.window_entries:
            if entry["id"] == reservation_id:
                entry["tokens"] = max(0, actual_tokens)
                entry["status"] = "confirmed"
                return

    def release_reservation(self, reservation_id: int) -> None:
        """Release reservation upon request failure without consuming token budget."""
        for i, entry in enumerate(self.window_entries):
            if entry["id"] == reservation_id:
                del self.window_entries[i]
                return

    def retain_reservation_failed(self, reservation_id: int) -> None:
        """Conservatively retain estimated reservation as unconfirmed_failed until window expiry."""
        for entry in self.window_entries:
            if entry["id"] == reservation_id:
                entry["status"] = "unconfirmed_failed"
                return

    def update_from_headers(
        self,
        headers: Mapping[str, Any],
        estimated_next_tokens: Optional[int] = None,
        receipt_time: Optional[float] = None,
    ) -> None:
        """Capture rate-limit headers and enforce quota-aware deadline before next request."""
        if not headers:
            return
        headers_lower = {str(k).lower(): str(v) for k, v in headers.items()}
        now = receipt_time if receipt_time is not None else self.clock()
        self.last_header_receipt_time = now
        self._last_processed_headers = dict(headers)

        # 1. Retry-After header
        retry_after_raw = headers_lower.get("retry-after")
        if retry_after_raw:
            try:
                delay = float(str(retry_after_raw).strip())
                self.retry_after_deadline = now + delay
                self.pending_quota_delay_seconds = max(self.pending_quota_delay_seconds, delay)
            except (ValueError, TypeError):
                pass

        # 2. Token quota reset duration tracking and adaptive request comparison
        rem_tokens_raw = headers_lower.get("x-ratelimit-remaining-tokens")
        reset_tokens_raw = headers_lower.get("x-ratelimit-reset-tokens")
        if rem_tokens_raw and reset_tokens_raw:
            try:
                rem = int(str(rem_tokens_raw).strip())
                self.last_remaining_tokens = rem

                r_str = str(reset_tokens_raw).strip().lower()
                if r_str.endswith("ms"):
                    reset_s = float(r_str[:-2]) / 1000.0
                elif r_str.endswith("s"):
                    reset_s = float(r_str[:-1])
                elif r_str.endswith("m"):
                    reset_s = float(r_str[:-1]) * 60.0
                else:
                    reset_s = float(r_str)
                self.last_reset_tokens_seconds = reset_s

                # Dynamic comparison: required tokens = estimated request size + 200 token buffer
                # Default threshold is 3000 tokens (sufficient for Condition B ~2800 tokens)
                required_tokens = (estimated_next_tokens + 200) if estimated_next_tokens is not None else 3000
                if rem < required_tokens:
                    self.token_reset_deadline = now + reset_s
                    self.pending_quota_delay_seconds = max(self.pending_quota_delay_seconds, reset_s)
                else:
                    self.token_reset_deadline = None
            except (ValueError, TypeError):
                pass

        deadlines = [d for d in (self.retry_after_deadline, self.token_reset_deadline) if d is not None]
        self.quota_deadline = max(deadlines) if deadlines else None

    def _recheck_token_quota(self, estimated_tokens: int) -> None:
        """Re-evaluates token quota deadline against known estimated tokens for next request."""
        if self.last_remaining_tokens is not None and self.last_reset_tokens_seconds is not None:
            required_tokens = estimated_tokens + 200
            receipt_t = self.last_header_receipt_time if self.last_header_receipt_time is not None else self.clock()
            if self.last_remaining_tokens < required_tokens:
                self.token_reset_deadline = receipt_t + self.last_reset_tokens_seconds
            else:
                self.token_reset_deadline = None
            deadlines = [d for d in (self.retry_after_deadline, self.token_reset_deadline) if d is not None]
            self.quota_deadline = max(deadlines) if deadlines else None

    def wait_for_slot(
        self,
        provider: Optional[Any] = None,
        estimated_tokens: Optional[int] = None,
    ) -> float:
        """Wait until min_interval_seconds, quota reset duration, and sliding-window budget permit dispatch."""
        now = self.clock()

        if provider is not None and hasattr(provider, "last_response_headers"):
            headers = provider.last_response_headers
            if headers:
                if self._last_processed_headers is not None and headers == self._last_processed_headers:
                    if estimated_tokens is not None:
                        self._recheck_token_quota(estimated_tokens)
                else:
                    self.update_from_headers(headers, estimated_next_tokens=estimated_tokens, receipt_time=now)
        elif estimated_tokens is not None:
            self._recheck_token_quota(estimated_tokens)

        est = estimated_tokens if estimated_tokens is not None else 0

        total_slept = 0.0
        while True:
            self.prune_window(now)

            # 1. Sliding window wait
            sliding_wait = 0.0
            if self.enable_sliding_window and est > 0:
                current_window_tokens = sum(e["tokens"] for e in self.window_entries)
                if current_window_tokens + est > self.effective_budget and self.window_entries:
                    acc = 0
                    target_time = None
                    for entry in self.window_entries:
                        acc += entry["tokens"]
                        if (current_window_tokens - acc) + est <= self.effective_budget:
                            target_time = entry["timestamp"] + self.window_seconds
                            break
                    if target_time is None and self.window_entries:
                        target_time = self.window_entries[-1]["timestamp"] + self.window_seconds
                    if target_time is not None:
                        sliding_wait = max(0.0, target_time - now + 0.05)

            # 2. Quota wait from monotonic deadline
            quota_wait = 0.0
            if self.quota_deadline is not None:
                quota_wait = max(0.0, self.quota_deadline - now)

            # 3. Inter-request interval wait from last request start
            interval_wait = 0.0
            if self.last_request_start is not None:
                elapsed = now - self.last_request_start
                interval_wait = max(0.0, self.min_interval_seconds - elapsed)

            wait_time = max(sliding_wait, quota_wait, interval_wait)
            self.pending_quota_delay_seconds = 0.0

            if wait_time <= 0.0:
                break

            if wait_time == sliding_wait and sliding_wait > 0.0:
                reason = "sliding_window_budget"
                reason_display = "sliding-window budget"
            elif wait_time == quota_wait and quota_wait > 0.0:
                reason = "quota_deadline"
                reason_display = "provider quota reset"
            else:
                reason = "min_interval"
                reason_display = "minimum interval"

            pacing_msg = f"[Pacing] Waiting {wait_time:.1f}s (reason: {reason_display})..."
            if self.printer is not None:
                self.printer(pacing_msg)
            elif self.verbose:
                print(pacing_msg, flush=True)

            self.sleeper(wait_time)
            self.total_sleep_seconds += wait_time
            self.sleep_events.append(wait_time)
            self.sleep_records.append({
                "timestamp": now,
                "duration": wait_time,
                "reason": reason,
                "reason_display": reason_display,
                "pacing_message": pacing_msg,
            })
            total_slept += wait_time
            now = self.clock()

            if self.token_reset_deadline is not None and now >= self.token_reset_deadline:
                self.last_remaining_tokens = None
                self.token_reset_deadline = None
            if self.retry_after_deadline is not None and now >= self.retry_after_deadline:
                self.retry_after_deadline = None

            deadlines = [d for d in (self.retry_after_deadline, self.token_reset_deadline) if d is not None]
            self.quota_deadline = max(deadlines) if deadlines else None

        if self.enable_sliding_window and est > 0:
            self.reserve_slot(est, now=now)

        self.last_request_start = now
        return total_slept


def is_pre_dispatch_connection_error(exc: Exception) -> bool:
    """Check if connection error unambiguously occurred before any HTTP request was transmitted."""
    msg = str(exc).lower()
    pre_dispatch_markers = [
        "getaddrinfo failed",
        "nodename nor servname provided",
        "name resolution failed",
        "failed to resolve",
        "dns resolution",
        "connection refused",
        "winerror 10061",
        "errno 111",
    ]
    return any(marker in msg for marker in pre_dispatch_markers)


def classify_reservation_action(exc: Exception) -> str:
    """Classify whether a failed request's token reservation should be released or retained.

    Returns:
        'release': For confirmed client-side / pre-dispatch rejections (HTTP 400, HTTP 429,
                   HTTP 401, or confirmed pre-dispatch connection failures).
        'retain':  For ambiguous post-dispatch failures, HTTP 500/502/503/504 server-side
                   errors, timeouts, and connection interruptions after dispatch.
    """
    exc_type = type(exc).__name__.lower()
    exc_msg = str(exc).lower()
    status_code = getattr(exc, "status_code", None)
    if status_code is None and hasattr(exc, "response"):
        resp = getattr(exc, "response", None)
        status_code = getattr(resp, "status_code", None)

    # 1. HTTP 429: Quota rejection (rate-limit headers lock the limiter; release request reservation)
    if (
        isinstance(exc, RateLimitError)
        or status_code == 429
        or "429" in exc_msg
        or "ratelimit" in exc_type
    ):
        return "release"

    # 2. HTTP 400: Confirmed client schema / tool validation rejection
    if (
        status_code == 400
        or "badrequest" in exc_type
        or "error code: 400" in exc_msg
        or "status code: 400" in exc_msg
        or "tool call validation failed" in exc_msg
    ):
        return "release"

    # 3. HTTP 401 / AuthenticationError: Unambiguous client credential rejection before inference
    if (
        isinstance(exc, AuthenticationError)
        or status_code == 401
        or "unauthorized" in exc_msg
        or "401" in exc_msg
    ):
        return "release"

    # 4. Unambiguous pre-dispatch connection error (DNS lookup failure, connection refused)
    if isinstance(exc, (ConnectionError, OSError)) and is_pre_dispatch_connection_error(exc):
        return "release"

    # 5. Default to conservative retention:
    # Includes HTTP 500/502/503/504, TimeoutError, ambiguous/post-dispatch ConnectionError,
    # and unexpected server/provider exceptions where prompt tokens may have been ingested.
    return "retain"


# =============================================================================
# 3. PILOT REQUEST GUARDRAIL PROVIDER
# =============================================================================

class GroqPilotRequestGuardrailProvider:
    """Decorates an LLMProvider to enforce strict invocation, token, and rate limits."""

    def __init__(
        self,
        provider: LLMProvider,
        max_requests: int = PROTOCOL_V2_GLOBAL_REQUEST_CEILING,
        rate_limiter: Optional[SequentialRateLimiter] = None,
        min_interval_seconds: float = 0.0,
        clock: Optional[Callable[[], float]] = None,
        sleeper: Optional[Callable[[float], None]] = None,
        telemetry_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.provider = provider
        self.max_requests = max_requests
        self.telemetry_path = Path(telemetry_path) if telemetry_path else None
        if rate_limiter is not None:
            self.rate_limiter: Optional[SequentialRateLimiter] = rate_limiter
        elif min_interval_seconds > 0.0:
            self.rate_limiter = SequentialRateLimiter(
                min_interval_seconds=min_interval_seconds,
                clock=clock,
                sleeper=sleeper,
            )
        else:
            self.rate_limiter = None

        self.total_requests: int = 0
        self.request_history: List[Dict[str, Any]] = []
        self.provider_request_ids: List[str] = []

        # Cumulative token counters
        self.cumulative_input_tokens: int = 0
        self.cumulative_output_tokens: int = 0
        self.cumulative_cached_input_tokens: int = 0
        self.cumulative_reasoning_tokens: int = 0
        self.cumulative_total_tokens: Optional[int] = None
        self.cumulative_derived_total_tokens: int = 0
        self.cumulative_sleep_seconds: float = 0.0
        self.telemetry_write_errors: int = 0
        self.telemetry_error_messages: List[str] = []

    def _log_request_telemetry(self, entry: Dict[str, Any]) -> None:
        """Append request-level telemetry entry to telemetry_path if configured."""
        if self.telemetry_path:
            try:
                self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.telemetry_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as err:
                self.telemetry_write_errors += 1
                err_msg = f"{type(err).__name__}: {err}"
                self.telemetry_error_messages.append(err_msg)
                req_idx = entry.get("request_index", "unknown")
                sys.stderr.write(
                    f"[WARNING] Telemetry write failed for request {req_idx}: {type(err).__name__}\n"
                )
                sys.stderr.flush()

    @property
    def sdk_version(self) -> Optional[str]:
        return getattr(self.provider, "sdk_version", None)

    @property
    def is_execution_lock_enabled(self) -> bool:
        return getattr(self.provider, "is_execution_lock_enabled", False)

    @property
    def effective_reasoning_effort(self) -> Optional[str]:
        return getattr(self.provider, "effective_reasoning_effort", None)

    @property
    def last_response_headers(self) -> Dict[str, str]:
        return getattr(self.provider, "last_response_headers", {})

    def complete(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: str = "low",
        max_output_tokens: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs: Any,
    ) -> LLMCompletionResult:
        """Enforces rate limiting and ceiling checks before forwarding completion calls."""
        if self.total_requests >= self.max_requests:
            raise RuntimeError(
                f"Groq pilot execution safety ceiling reached: {self.total_requests}/{self.max_requests} requests. "
                "Halting further API invocations to protect quota limits."
            )

        slept = 0.0
        res_id = None
        if self.rate_limiter is not None:
            override_est = kwargs.pop("estimated_tokens", None)
            if override_est is not None:
                est_tokens = int(override_est)
            else:
                prompt_chars = sum(len(m.content or "") for m in messages)
                if tools:
                    try:
                        prompt_chars += len(json.dumps(tools))
                    except Exception:
                        pass
                est_tokens = max(1, int(prompt_chars / 3.0))
            slept = self.rate_limiter.wait_for_slot(provider=self.provider, estimated_tokens=est_tokens)
            res_id = getattr(self.rate_limiter, "last_reservation_id", None)
            self.cumulative_sleep_seconds += slept

        self.total_requests += 1

        t0 = time.perf_counter()
        try:
            result = self.provider.complete(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
                parallel_tool_calls=parallel_tool_calls,
                **kwargs,
            )
        except KeyboardInterrupt as exc:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            if res_id is not None and self.rate_limiter is not None:
                self.rate_limiter.retain_reservation_failed(res_id)
            interrupt_entry = {
                "request_index": self.total_requests,
                "status": "outcome_unknown",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "error": type(exc).__name__,
                "interruption_type": type(exc).__name__,
                "duration_ms": duration_ms,
                "sleep_seconds": slept,
                "usage": None,
            }
            self.request_history.append(interrupt_entry)
            self._log_request_telemetry(interrupt_entry)
            raise
        except Exception as exc:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            if res_id is not None and self.rate_limiter is not None:
                action = classify_reservation_action(exc)
                if action == "release":
                    self.rate_limiter.release_reservation(res_id)
                else:
                    self.rate_limiter.retain_reservation_failed(res_id)
            headers = getattr(self.provider, "last_response_headers", {})
            if headers and self.rate_limiter is not None:
                self.rate_limiter.update_from_headers(headers)
            err_entry = {
                "request_index": self.total_requests,
                "status": "error",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "error": type(exc).__name__,
                "duration_ms": duration_ms,
                "sleep_seconds": slept,
            }
            self.request_history.append(err_entry)
            self._log_request_telemetry(err_entry)
            raise

        duration_ms = (time.perf_counter() - t0) * 1000.0
        if res_id is not None and self.rate_limiter is not None:
            actual_input = result.usage.input_tokens if (result.usage and result.usage.input_tokens is not None) else est_tokens
            self.rate_limiter.reconcile_reservation(res_id, actual_input)
        headers = getattr(self.provider, "last_response_headers", {})
        if headers and self.rate_limiter is not None:
            self.rate_limiter.update_from_headers(headers)

        if result.usage:
            self.cumulative_input_tokens += result.usage.input_tokens
            self.cumulative_output_tokens += result.usage.output_tokens
            if result.usage.cached_input_tokens is not None:
                self.cumulative_cached_input_tokens += result.usage.cached_input_tokens
            if result.usage.reasoning_tokens is not None:
                self.cumulative_reasoning_tokens += result.usage.reasoning_tokens
            if result.usage.total_tokens is not None:
                if self.cumulative_total_tokens is None:
                    self.cumulative_total_tokens = result.usage.total_tokens
                else:
                    self.cumulative_total_tokens += result.usage.total_tokens
            if result.usage.derived_total_tokens is not None:
                self.cumulative_derived_total_tokens += result.usage.derived_total_tokens
            else:
                self.cumulative_derived_total_tokens += (result.usage.input_tokens + result.usage.output_tokens)

        if result.provider_request_id:
            self.provider_request_ids.append(result.provider_request_id)

        succ_entry = {
            "request_index": self.total_requests,
            "status": "success",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "provider_request_id": result.provider_request_id,
            "duration_ms": duration_ms,
            "sleep_seconds": slept,
            "usage": usage_to_dict(result.usage),
        }
        self.request_history.append(succ_entry)
        self._log_request_telemetry(succ_entry)

        return result


# =============================================================================
# 4. PREFLIGHT VALIDATION & PROMPT ESTIMATION (ZERO NETWORK ACTIVITY)
# =============================================================================

def estimate_prompt_tokens(text: str) -> int:
    """Conservatively estimate LLM token count from raw text without external tokenizer.

    For Qwen / OpenAI-compatible tokenizers on structured text/JSON, the ratio is
    approximately ~3.0 characters per token.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.0))


def estimate_condition_prompt_tokens(
    condition_id: str,
    task: BenchmarkTask,
    plan: MealPlan,
    catalog: Dict[str, Any],
    ingredients: Dict[str, Any],
) -> int:
    """Estimate prompt tokens for a specific condition and task before dispatch."""
    ctx_req = ContextRequest(
        user_request=task.user_request,
        plan=plan,
        catalog=catalog,
        ingredients=ingredients,
    )
    if condition_id == "A":
        builder = FullStateContextBuilder()
    elif condition_id == "B":
        builder = ToolContextBuilder()
    elif condition_id == "C":
        builder = ObjectContextBuilder()
    else:
        raise ValueError(f"Unknown condition_id: {condition_id}")

    ctx_res = builder.build(ctx_req)
    full_prompt_text = (
        ctx_res.system_prompt
        + "\nUser Request:\n"
        + task.user_request
        + "\nCurrent State:\n"
        + ctx_res.serialized_context
    )
    return estimate_prompt_tokens(full_prompt_text)


def run_groq_pilot_preflight(
    catalog_path: Union[str, Path] = "data/catalog_seed.json",
    fixtures_path: Union[str, Path] = "data/plan_fixtures.json",
    tasks_path: Union[str, Path] = "data/tasks.jsonl",
    api_key: Optional[str] = None,
    protocol_version: str = "v2",
    conditions: Sequence[str] = PILOT_CONDITIONS,
    enforce_itpm_limits: bool = False,
) -> Dict[str, Any]:
    """Execute preflight configuration and data integrity validation for Groq."""
    catalog_p = Path(catalog_path)
    fixtures_p = Path(fixtures_path)
    tasks_p = Path(tasks_path)

    errors: List[str] = []
    warnings: List[str] = []

    # API key check (never log secrets)
    load_project_env()
    configured_key = api_key or os.environ.get("GROQ_API_KEY", "")
    key_present = bool(configured_key and configured_key.strip())

    catalog = {}
    ingredients = {}
    if not catalog_p.exists():
        errors.append(f"Catalog file not found: {catalog_p}")
    else:
        try:
            ingredients, catalog = load_catalog_from_json(catalog_p)
            if len(catalog) < 15:
                errors.append(f"Catalog contains only {len(catalog)} recipes; expected at least 15.")
        except Exception as e:
            errors.append(f"Failed to load catalog: {e}")

    fixtures = {}
    if not fixtures_p.exists():
        errors.append(f"Plan fixtures file not found: {fixtures_p}")
    elif catalog:
        try:
            fixtures = load_plan_fixtures(fixtures_p, catalog, ingredients)
            if "fixture_standard" not in fixtures:
                errors.append("Standard fixture 'fixture_standard' missing from fixtures.")
        except Exception as e:
            errors.append(f"Failed to load plan fixtures: {e}")

    pilot_tasks: List[BenchmarkTask] = []
    if not tasks_p.exists():
        errors.append(f"Tasks file not found: {tasks_p}")
    elif fixtures and catalog:
        try:
            all_tasks = load_benchmark_tasks(tasks_p, fixtures, catalog)
            task_map = {t.task_id: t for t in all_tasks}
            missing = [tid for tid in PILOT_TASK_IDS if tid not in task_map]
            if missing:
                errors.append(f"Required pilot task(s) missing from {tasks_p}: {missing}")
            else:
                pilot_tasks = [task_map[tid] for tid in PILOT_TASK_IDS]
                expected_matrix = {
                    "task_01_commit_replace_wed_dinner": ExpectedOutcome.COMMIT,
                    "task_14_reject_allergen_peanut_toast": ExpectedOutcome.REJECT,
                    "task_22_clarify_missing_meal_slot": ExpectedOutcome.CLARIFY,
                }
                for t in pilot_tasks:
                    exp = expected_matrix[t.task_id]
                    if t.expected_outcome != exp:
                        errors.append(
                            f"Task {t.task_id} has outcome {t.expected_outcome.value}; "
                            f"expected {exp.value} for pilot."
                        )
        except Exception as e:
            errors.append(f"Failed to load benchmark tasks: {e}")

    # Prompt size estimation and ITPM limit check
    estimated_prompt_tokens: Dict[str, int] = {}
    if fixtures and catalog and pilot_tasks:
        sample_task = pilot_tasks[0]
        sample_plan = fixtures[sample_task.fixture_id].plan
        for cond in conditions:
            try:
                est = estimate_condition_prompt_tokens(
                    condition_id=cond,
                    task=sample_task,
                    plan=sample_plan,
                    catalog=catalog,
                    ingredients=ingredients,
                )
                estimated_prompt_tokens[cond] = est
                if est > GROQ_FREE_TIER_ITPM_LIMIT:
                    msg = (
                        f"Condition {cond} estimated prompt size ({est} tokens) exceeds "
                        f"Groq Free Tier ITPM limit of {GROQ_FREE_TIER_ITPM_LIMIT} tokens. "
                        f"Dispatching will trigger HTTP 413 rate_limit_exceeded."
                    )
                    if enforce_itpm_limits:
                        errors.append(msg)
                    else:
                        warnings.append(msg)
            except Exception as e:
                warnings.append(f"Could not estimate prompt tokens for Condition {cond}: {e}")

    try:
        pricing = create_groq_free_pricing()
    except Exception as e:
        pricing = None
        errors.append(f"Invalid pricing configuration: {e}")

    c_reqs = PROTOCOL_V2_MAX_REQUESTS_PER_CONDITION_C
    c_insps = PROTOCOL_V2_MAX_INSPECTIONS_PER_CONDITION_C
    ceil_reqs = PROTOCOL_V2_GLOBAL_REQUEST_CEILING

    tasks_count = len(pilot_tasks) if pilot_tasks else len(PILOT_TASK_IDS)
    planned_runs = tasks_count * len(conditions)

    theor_reqs = tasks_count * sum(
        MAX_REQUESTS_PER_CONDITION_A if c == "A" else (MAX_REQUESTS_PER_CONDITION_B if c == "B" else c_reqs)
        for c in conditions
    )

    budget_breakdown = " + ".join(
        f"{MAX_REQUESTS_PER_CONDITION_A if c == 'A' else (MAX_REQUESTS_PER_CONDITION_B if c == 'B' else c_reqs)} {c}"
        for c in conditions
    )
    request_budget_formula = f"{tasks_count} tasks × ({budget_breakdown}) = {theor_reqs} requests"

    return {
        "status": "passed" if not errors else "failed",
        "api_key_configured": key_present,
        "pilot_model": PILOT_MODEL,
        "base_url": DEFAULT_BASE_URL,
        "planned_runs_count": planned_runs,
        "conditions": list(conditions),
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "itpm_limit": GROQ_FREE_TIER_ITPM_LIMIT,
        "enforce_itpm_limits": enforce_itpm_limits,
        "max_requests_per_condition_a": MAX_REQUESTS_PER_CONDITION_A,
        "max_requests_per_condition_b": MAX_REQUESTS_PER_CONDITION_B,
        "max_requests_per_condition_c": c_reqs,
        "max_inspections_per_condition_c": c_insps,
        "max_theoretical_requests": theor_reqs,
        "max_possible_total_requests": theor_reqs,
        "request_budget_formula": request_budget_formula,
        "global_request_ceiling": ceil_reqs,
        "max_provider_requests_ceiling": ceil_reqs,
        "safety_margin_requests": ceil_reqs - theor_reqs,
        "pricing_complete": pricing.is_complete() if pricing else False,
        "billing_tier": "free",
        "tasks_loaded": len(pilot_tasks),
        "protocol_version": protocol_version,
        "errors": errors,
        "warnings": warnings,
    }


# =============================================================================
# 5. DETERMINISTIC MOCK FOR DRY-RUN TESTING
# =============================================================================

def create_groq_pilot_mock_provider() -> MockLLMProvider:
    """Create offline mock provider configured specifically for Groq pilot dry run."""
    canned: Dict[str, LLMCompletionResult] = {
        "pasta": LLMCompletionResult(
            message=LLMMessage(
                role="assistant",
                content='{"action": "replace_meal", "day": "Wednesday", "slot": "Dinner", "recipe": "Vegetable Pasta Primavera"}',
            ),
            usage=LLMUsage(
                input_tokens=150,
                output_tokens=48,
                total_tokens=198,
                cached_input_tokens=0,
                reasoning_tokens=0,
                is_provider_reported=False,
            ),
            model_identifier="mock-qwen-3.8-27b",
            system_fingerprint="groq_mock_fp",
            provider_request_id="groq_mock_req_01",
            latency_ms=45.0,
            finish_reason="stop",
        ),
        "peanut": LLMCompletionResult(
            message=LLMMessage(
                role="assistant",
                content="I cannot schedule peanut butter toast because the profile has a severe peanut allergy constraint.",
            ),
            usage=LLMUsage(
                input_tokens=140,
                output_tokens=30,
                total_tokens=170,
                is_provider_reported=False,
            ),
            model_identifier="mock-qwen-3.8-27b",
            system_fingerprint="groq_mock_fp",
            provider_request_id="groq_mock_req_02",
            latency_ms=40.0,
            finish_reason="stop",
        ),
        "unclear": LLMCompletionResult(
            message=LLMMessage(
                role="assistant",
                content='{"clarification_request": "Which meal slot would you like to assign this recipe to?"}',
            ),
            usage=LLMUsage(
                input_tokens=120,
                output_tokens=25,
                total_tokens=145,
                is_provider_reported=False,
            ),
            model_identifier="mock-qwen-3.8-27b",
            system_fingerprint="groq_mock_fp",
            provider_request_id="groq_mock_req_03",
            latency_ms=35.0,
            finish_reason="stop",
        ),
    }

    default_resp = LLMCompletionResult(
        message=LLMMessage(
            role="assistant",
            content='{"status": "groq_dry_run_simulated_completion"}',
        ),
        usage=LLMUsage(
            input_tokens=105,
            output_tokens=20,
            total_tokens=125,
            is_provider_reported=False,
        ),
        model_identifier="mock-qwen-3.8-27b",
        system_fingerprint="groq_mock_fp",
        provider_request_id="groq_mock_req_default",
        latency_ms=30.0,
        finish_reason="stop",
    )

    return MockLLMProvider(canned_responses=canned, default_response=default_resp)


# =============================================================================
# 6. DRY RUN RUNNER (ZERO NETWORK CALLS)
# =============================================================================

def run_groq_pilot_dry_run(
    catalog_path: Union[str, Path] = "data/catalog_seed.json",
    fixtures_path: Union[str, Path] = "data/plan_fixtures.json",
    tasks_path: Union[str, Path] = "data/tasks.jsonl",
    output_dir: Optional[Union[str, Path]] = None,
    provider: Optional[LLMProvider] = None,
    rate_limiter: Optional[SequentialRateLimiter] = None,
    max_requests: Optional[int] = None,
    protocol_version: str = "v2",
    conditions: Sequence[str] = PILOT_CONDITIONS,
) -> Dict[str, Any]:
    """Execute simulated pilot offline with deterministic mock data."""
    effective_max_requests = max_requests if max_requests is not None else PROTOCOL_V2_GLOBAL_REQUEST_CEILING
    c_requests = PROTOCOL_V2_MAX_REQUESTS_PER_CONDITION_C
    c_inspections = PROTOCOL_V2_MAX_INSPECTIONS_PER_CONDITION_C
    config_version = "protocol-v2-groq-pilot-dry-run"

    ingredients, catalog = load_catalog_from_json(catalog_path)
    fixtures = load_plan_fixtures(fixtures_path, catalog, ingredients)
    all_tasks = load_benchmark_tasks(tasks_path, fixtures, catalog)

    task_map = {t.task_id: t for t in all_tasks}
    pilot_tasks = [task_map[tid] for tid in PILOT_TASK_IDS]

    planned_runs = len(pilot_tasks) * len(conditions)
    theoretical_total = len(pilot_tasks) * sum(
        MAX_REQUESTS_PER_CONDITION_A if c == "A" else (MAX_REQUESTS_PER_CONDITION_B if c == "B" else c_requests)
        for c in conditions
    )

    pricing = create_groq_free_pricing()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"groq_dry_run_{timestamp}"

    if output_dir is None:
        target_dir = Path("artifacts") / "protocol_v2_pilot" / run_id
    else:
        target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = target_dir / "request_telemetry.jsonl"
    meta_path = target_dir / "experiment_meta.json"
    raw_runs_path = target_dir / "raw_runs.jsonl"

    meta = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "pilot_model": "mock-qwen-3.8-27b",
        "conditions": list(conditions),
        "task_ids": [t.task_id for t in pilot_tasks],
        "planned_runs": planned_runs,
        "completed_runs": 0,
        "max_requests_ceiling": effective_max_requests,
        "global_request_ceiling": effective_max_requests,
        "protocol_version": protocol_version,
        "incomplete_reason": None,
        "telemetry_complete": True,
        "telemetry_write_errors": 0,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    underlying_provider = provider or create_groq_pilot_mock_provider()
    guardrail = GroqPilotRequestGuardrailProvider(
        provider=underlying_provider,
        max_requests=effective_max_requests,
        rate_limiter=rate_limiter,
        telemetry_path=telemetry_path,
    )

    runner = ExperimentRunner(
        fixtures=fixtures,
        tasks=pilot_tasks,
        catalog=catalog,
        ingredients=ingredients,
        pricing=pricing,
        default_provider=guardrail,
        experiment_run_id=run_id,
        experiment_config_version=config_version,
        provider_mode="simulated",
        model_name="mock-qwen-3.8-27b",
        condition_c_max_model_requests=c_requests,
        condition_c_max_inspections=c_inspections,
        protocol_version=protocol_version,
    )

    records: List[RunRecord] = []
    incomplete_reason: Optional[str] = None
    interrupted = False

    try:
        for task in pilot_tasks:
            for condition in conditions:
                if guardrail.total_requests >= effective_max_requests:
                    incomplete_reason = f"Request ceiling of {effective_max_requests} reached before starting {task.task_id} condition {condition}."
                    break
                try:
                    rec = runner.run_task(task=task, condition_id=condition, provider=guardrail)
                    records.append(rec)
                    with open(raw_runs_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec.to_dict()) + "\n")
                    meta["completed_runs"] = len(records)
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                except RuntimeError as exc:
                    if "safety ceiling reached" in str(exc).lower():
                        incomplete_reason = str(exc)
                        break
                    raise
            if incomplete_reason is not None:
                break
    except KeyboardInterrupt:
        interrupted = True
        incomplete_reason = "Interrupted by user (KeyboardInterrupt / SIGINT)."

    is_complete = len(records) == planned_runs and incomplete_reason is None
    status = "interrupted" if interrupted else ("completed" if is_complete else "incomplete")

    # Export artifacts
    runner.export_artifacts(records, target_dir)
    _patch_pilot_analysis_markdown(
        target_dir / "analysis.md",
        run_id=run_id,
        is_live=False,
        guardrail=guardrail,
        max_requests=effective_max_requests,
        incomplete_reason=incomplete_reason,
    )

    meta["status"] = status
    meta["completed_runs"] = len(records)
    meta["incomplete_reason"] = incomplete_reason
    meta["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta["total_provider_requests"] = guardrail.total_requests
    meta["telemetry_complete"] = (guardrail.telemetry_write_errors == 0)
    meta["telemetry_write_errors"] = guardrail.telemetry_write_errors
    if guardrail.telemetry_write_errors > 0:
        meta["telemetry_error_messages"] = list(guardrail.telemetry_error_messages)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "status": status,
        "incomplete_reason": incomplete_reason,
        "run_id": run_id,
        "runs_completed": len(records),
        "runs_planned": planned_runs,
        "conditions": list(conditions),
        "total_provider_requests": guardrail.total_requests,
        "telemetry_complete": (guardrail.telemetry_write_errors == 0),
        "telemetry_write_errors": guardrail.telemetry_write_errors,
        "max_theoretical_requests": theoretical_total,
        "max_requests_per_condition_c": c_requests,
        "max_inspections_per_condition_c": c_inspections,
        "max_requests_ceiling": effective_max_requests,
        "global_request_ceiling": effective_max_requests,
        "ceiling_reached": not is_complete and (guardrail.total_requests >= effective_max_requests or "ceiling" in (incomplete_reason or "").lower()),
        "network_calls_made": 0,
        "output_directory": str(target_dir),
        "protocol_version": protocol_version,
        "artifacts": {
            "raw_runs": str(raw_runs_path),
            "comparison_csv": str(target_dir / "comparison.csv"),
            "analysis_markdown": str(target_dir / "analysis.md"),
            "experiment_meta": str(meta_path),
            "request_telemetry": str(telemetry_path),
        },
    }


# =============================================================================
# 7. LIVE PILOT RUNNER (REQUIRES DUAL CONFIRMATION)
# =============================================================================

def run_live_groq_pilot(
    api_key: Optional[str] = None,
    catalog_path: Union[str, Path] = "data/catalog_seed.json",
    fixtures_path: Union[str, Path] = "data/plan_fixtures.json",
    tasks_path: Union[str, Path] = "data/tasks.jsonl",
    output_dir: Optional[Union[str, Path]] = None,
    live: bool = False,
    confirm_live: bool = False,
    max_requests: Optional[int] = None,
    min_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS,
    client: Optional[Any] = None,
    clock: Optional[Callable[[], float]] = None,
    sleeper: Optional[Callable[[float], None]] = None,
    protocol_version: str = "v2",
    conditions: Sequence[str] = PILOT_CONDITIONS,
    enforce_itpm_limits: bool = True,
    rate_limiter_verbose: bool = True,
) -> Dict[str, Any]:
    """Execute live Groq pilot with strict dual-flag confirmation and request ceilings."""
    if not (live and confirm_live):
        raise RuntimeError(
            "Live Groq pilot execution requires explicit dual confirmation: "
            "both --live and --confirm-live flags must be supplied. "
            "Cleanly aborting to prevent unintended API execution."
        )

    effective_max_requests = max_requests if max_requests is not None else PROTOCOL_V2_GLOBAL_REQUEST_CEILING
    c_requests = PROTOCOL_V2_MAX_REQUESTS_PER_CONDITION_C
    c_inspections = PROTOCOL_V2_MAX_INSPECTIONS_PER_CONDITION_C
    config_version = "protocol-v2-groq-live-pilot"

    # 1. Preflight validation
    preflight = run_groq_pilot_preflight(
        catalog_path=catalog_path,
        fixtures_path=fixtures_path,
        tasks_path=tasks_path,
        api_key=api_key,
        protocol_version=protocol_version,
        conditions=conditions,
        enforce_itpm_limits=enforce_itpm_limits,
    )
    if preflight["status"] != "passed":
        raise RuntimeError(f"Preflight validation failed: {preflight['errors']}")
    if not preflight["api_key_configured"]:
        raise RuntimeError("GROQ_API_KEY environment variable is not configured.")

    ingredients, catalog = load_catalog_from_json(catalog_path)
    fixtures = load_plan_fixtures(fixtures_path, catalog, ingredients)
    all_tasks = load_benchmark_tasks(tasks_path, fixtures, catalog)

    task_map = {t.task_id: t for t in all_tasks}
    pilot_tasks = [task_map[tid] for tid in PILOT_TASK_IDS]

    planned_runs = len(pilot_tasks) * len(conditions)
    theoretical_total = len(pilot_tasks) * sum(
        MAX_REQUESTS_PER_CONDITION_A if c == "A" else (MAX_REQUESTS_PER_CONDITION_B if c == "B" else c_requests)
        for c in conditions
    )

    pricing = create_groq_free_pricing()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"groq_live_pilot_{timestamp}"

    if output_dir is None:
        target_dir = Path("artifacts") / "protocol_v2_pilot" / run_id
    else:
        target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = target_dir / "request_telemetry.jsonl"
    meta_path = target_dir / "experiment_meta.json"
    raw_runs_path = target_dir / "raw_runs.jsonl"

    meta = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "pilot_model": PILOT_MODEL,
        "conditions": list(conditions),
        "task_ids": [t.task_id for t in pilot_tasks],
        "planned_runs": planned_runs,
        "completed_runs": 0,
        "max_requests_ceiling": effective_max_requests,
        "global_request_ceiling": effective_max_requests,
        "protocol_version": protocol_version,
        "incomplete_reason": None,
        "telemetry_complete": True,
        "telemetry_write_errors": 0,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    base_provider = GroqProvider(
        api_key=api_key,
        model=PILOT_MODEL,
        base_url=DEFAULT_BASE_URL,
        timeout_seconds=60.0,
        client=client,
        max_output_tokens=DEFAULT_OUTPUT_TOKEN_CEILING,
        max_retries=0,
    )

    rate_limiter = SequentialRateLimiter(
        min_interval_seconds=min_interval_seconds,
        clock=clock,
        sleeper=sleeper,
        verbose=rate_limiter_verbose,
    )

    guardrail = GroqPilotRequestGuardrailProvider(
        provider=base_provider,
        max_requests=effective_max_requests,
        rate_limiter=rate_limiter,
        telemetry_path=telemetry_path,
    )

    runner = ExperimentRunner(
        fixtures=fixtures,
        tasks=pilot_tasks,
        catalog=catalog,
        ingredients=ingredients,
        pricing=pricing,
        default_provider=guardrail,
        experiment_run_id=run_id,
        experiment_config_version=config_version,
        provider_mode="live",
        model_name=PILOT_MODEL,
        condition_c_max_model_requests=c_requests,
        condition_c_max_inspections=c_inspections,
        protocol_version=protocol_version,
    )

    records: List[RunRecord] = []
    incomplete_reason: Optional[str] = None
    interrupted = False

    try:
        for task in pilot_tasks:
            for condition in conditions:
                if guardrail.total_requests >= effective_max_requests:
                    incomplete_reason = f"Request ceiling of {effective_max_requests} reached before starting {task.task_id} condition {condition}."
                    break
                try:
                    rec = runner.run_task(task=task, condition_id=condition, provider=guardrail)
                    records.append(rec)
                    with open(raw_runs_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec.to_dict()) + "\n")
                    meta["completed_runs"] = len(records)
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                except RuntimeError as exc:
                    if "safety ceiling reached" in str(exc).lower():
                        incomplete_reason = str(exc)
                        break
                    raise
            if incomplete_reason is not None:
                break
    except KeyboardInterrupt:
        interrupted = True
        incomplete_reason = "Interrupted by user (KeyboardInterrupt / SIGINT)."

    is_complete = len(records) == planned_runs and incomplete_reason is None
    status = "interrupted" if interrupted else ("completed" if is_complete else "incomplete")

    # Export artifacts
    runner.export_artifacts(records, target_dir)
    _patch_pilot_analysis_markdown(
        target_dir / "analysis.md",
        run_id=run_id,
        is_live=True,
        guardrail=guardrail,
        max_requests=effective_max_requests,
        incomplete_reason=incomplete_reason,
    )

    meta["status"] = status
    meta["completed_runs"] = len(records)
    meta["incomplete_reason"] = incomplete_reason
    meta["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta["total_provider_requests"] = guardrail.total_requests
    meta["telemetry_complete"] = (guardrail.telemetry_write_errors == 0)
    meta["telemetry_write_errors"] = guardrail.telemetry_write_errors
    if guardrail.telemetry_write_errors > 0:
        meta["telemetry_error_messages"] = list(guardrail.telemetry_error_messages)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "status": status,
        "incomplete_reason": incomplete_reason,
        "run_id": run_id,
        "runs_completed": len(records),
        "runs_planned": planned_runs,
        "conditions": list(conditions),
        "total_provider_requests": guardrail.total_requests,
        "telemetry_complete": (guardrail.telemetry_write_errors == 0),
        "telemetry_write_errors": guardrail.telemetry_write_errors,
        "max_theoretical_requests": theoretical_total,
        "max_requests_per_condition_c": c_requests,
        "max_inspections_per_condition_c": c_inspections,
        "max_requests_ceiling": effective_max_requests,
        "global_request_ceiling": effective_max_requests,
        "ceiling_reached": not is_complete and (guardrail.total_requests >= effective_max_requests or "ceiling" in (incomplete_reason or "").lower()),
        "protocol_version": protocol_version,
        "cumulative_tokens": {
            "input": guardrail.cumulative_input_tokens,
            "output": guardrail.cumulative_output_tokens,
            "cached": guardrail.cumulative_cached_input_tokens,
            "reasoning": guardrail.cumulative_reasoning_tokens,
            "total": guardrail.cumulative_total_tokens,
            "derived_total": guardrail.cumulative_derived_total_tokens,
        },
        "output_directory": str(target_dir),
        "artifacts": {
            "raw_runs": str(raw_runs_path),
            "comparison_csv": str(target_dir / "comparison.csv"),
            "analysis_markdown": str(target_dir / "analysis.md"),
            "experiment_meta": str(meta_path),
            "request_telemetry": str(telemetry_path),
        },
    }


# =============================================================================
# 8. STANDALONE ONE-REQUEST CONNECTION TEST
# =============================================================================

def run_groq_connection_test(
    api_key: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    connection_test: bool = False,
    confirm_connection_test: bool = False,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a single-request live connectivity test against Groq Cloud API."""
    if not (connection_test and confirm_connection_test):
        raise RuntimeError(
            "Standalone Groq connection test requires explicit dual confirmation: "
            "both --connection-test and --confirm-connection-test flags must be supplied. "
            "Neither flag alone may make a network request."
        )

    load_project_env()
    configured_key = api_key or os.environ.get("GROQ_API_KEY", "")
    key_configured = bool(configured_key and configured_key.strip())

    ts_utc = datetime.now(timezone.utc).isoformat()
    test_run_id = f"groq_connection_test_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    artifact_dir = Path(output_dir) if output_dir else Path(DEFAULT_CONNECTION_TEST_ARTIFACT_DIR) / test_run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifact_dir / "connection_test_result.json"

    if not key_configured:
        payload = {
            "status": "failed",
            "success": False,
            "error": "GROQ_API_KEY environment variable is not configured.",
            "error_type": "MissingAPIKeyError",
            "model_requested": PILOT_MODEL,
            "actual_model_identifier": None,
            "provider_request_id": None,
            "system_fingerprint": None,
            "latency_ms": 0.0,
            "timestamp_utc": ts_utc,
            "prompt_sent": CONNECTION_TEST_PROMPT,
            "response_text": None,
            "token_usage": None,
            "usage_provenance": None,
            "billing_tier": "free",
            "artifact_file": str(result_path),
            "requests_made": 0,
            "max_requests_allowed": 1,
        }
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    provider = GroqProvider(
        api_key=configured_key,
        model=PILOT_MODEL,
        base_url=DEFAULT_BASE_URL,
        timeout_seconds=30.0,
        client=client,
        max_output_tokens=CONNECTION_TEST_OUTPUT_TOKEN_CEILING,
        max_retries=0,
    )
    guardrail = GroqPilotRequestGuardrailProvider(provider=provider, max_requests=1)

    t_start = time.perf_counter()
    try:
        completion = guardrail.complete([
            LLMMessage(role="user", content=CONNECTION_TEST_PROMPT),
        ])
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        resp_text = (completion.message.content or "").strip()
        success = EXPECTED_CONNECTION_TEST_TOKEN in resp_text

        error = None
        error_type = None
        if not success:
            error = f"Expected token {EXPECTED_CONNECTION_TEST_TOKEN!r} not found in model response: {resp_text!r}"
            error_type = "UnexpectedResponseError"

        usage_dict = usage_to_dict(completion.usage)
        usage_prov = "provider_reported" if (completion.usage and completion.usage.is_provider_reported) else "unknown"

        payload = {
            "status": "passed" if success else "failed",
            "success": success,
            "model_requested": PILOT_MODEL,
            "actual_model_identifier": completion.model_identifier,
            "provider_request_id": completion.provider_request_id,
            "system_fingerprint": completion.system_fingerprint,
            "latency_ms": latency_ms,
            "timestamp_utc": ts_utc,
            "prompt_sent": CONNECTION_TEST_PROMPT,
            "response_text": resp_text,
            "token_usage": usage_dict,
            "usage_provenance": usage_prov,
            "billing_tier": "free",
            "artifact_file": str(result_path),
            "requests_made": guardrail.total_requests,
            "max_requests_allowed": 1,
            "error": error,
            "error_type": error_type,
        }

    except Exception as exc:
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        exc_type_name = type(exc).__name__
        exc_msg = str(exc)
        for secret in filter(None, [api_key, os.environ.get("GROQ_API_KEY")]):
            exc_msg = exc_msg.replace(secret, "[REDACTED_API_KEY]")

        payload = {
            "status": "failed",
            "success": False,
            "error": exc_msg,
            "error_type": exc_type_name,
            "model_requested": PILOT_MODEL,
            "actual_model_identifier": None,
            "provider_request_id": None,
            "system_fingerprint": None,
            "latency_ms": latency_ms,
            "timestamp_utc": ts_utc,
            "prompt_sent": CONNECTION_TEST_PROMPT,
            "response_text": None,
            "token_usage": None,
            "usage_provenance": None,
            "billing_tier": "free",
            "artifact_file": str(result_path),
            "requests_made": guardrail.total_requests,
            "max_requests_allowed": 1,
        }

    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# =============================================================================
# 8b. STANDALONE ONE-REQUEST TOOL-CALLING SMOKE TEST
# =============================================================================

def run_groq_tool_calling_smoke_test(
    api_key: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    smoke_test_tool_call: bool = False,
    confirm_smoke_test_tool_call: bool = False,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a single-request live tool calling smoke test against Groq Cloud API.

    Verifies:
    1. Tool definition serialization for inspect_plan.
    2. Forced tool invocation via named tool_choice.
    3. Return of exactly one tool call matching 'inspect_plan'.
    4. Valid JSON arguments.
    5. No inspection function is executed and no application state is mutated.
    6. Usage reporting and latency extraction.
    7. Hard limit of 1 HTTP request maximum (zero SDK retries).
    """
    if not (smoke_test_tool_call and confirm_smoke_test_tool_call):
        raise RuntimeError(
            "Standalone Groq tool-calling smoke test requires explicit dual confirmation: "
            "both --smoke-test-tool-call and --confirm-smoke-test-tool-call flags must be supplied. "
            "Neither flag alone may make a network request."
        )

    load_project_env()
    configured_key = api_key or os.environ.get("GROQ_API_KEY", "")
    key_configured = bool(configured_key and configured_key.strip())

    ts_utc = datetime.now(timezone.utc).isoformat()
    test_run_id = f"groq_smoke_tool_call_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    artifact_dir = Path(output_dir) if output_dir else Path(DEFAULT_TOOL_SMOKE_TEST_ARTIFACT_DIR) / test_run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifact_dir / "tool_calling_smoke_test_result.json"

    if not key_configured:
        payload = {
            "status": "failed",
            "success": False,
            "test_name": "tool_calling_smoke_test",
            "error": "GROQ_API_KEY environment variable is not configured.",
            "error_type": "MissingAPIKeyError",
            "model_requested": PILOT_MODEL,
            "actual_model_identifier": None,
            "provider_request_id": None,
            "system_fingerprint": None,
            "latency_ms": 0.0,
            "timestamp_utc": ts_utc,
            "prompt_sent": TOOL_SMOKE_TEST_PROMPT,
            "tool_definitions_sent": [INSPECT_PLAN_TOOL_DEFINITION],
            "tool_choice_sent": INSPECT_PLAN_TOOL_CHOICE,
            "tool_calls_received": None,
            "tool_calls_count": 0,
            "tool_call_name_matches": False,
            "tool_call_arguments_valid_json": False,
            "tool_call_arguments_parsed": None,
            "function_executed": False,
            "state_mutated": False,
            "token_usage": None,
            "usage_provenance": None,
            "billing_tier": "free",
            "artifact_file": str(result_path),
            "requests_made": 0,
            "max_requests_allowed": 1,
        }
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    provider = GroqProvider(
        api_key=configured_key,
        model=PILOT_MODEL,
        base_url=DEFAULT_BASE_URL,
        timeout_seconds=30.0,
        client=client,
        max_output_tokens=TOOL_SMOKE_TEST_OUTPUT_TOKEN_CEILING,
        max_retries=0,
    )
    guardrail = GroqPilotRequestGuardrailProvider(provider=provider, max_requests=1)

    t_start = time.perf_counter()
    try:
        completion = guardrail.complete(
            messages=[LLMMessage(role="user", content=TOOL_SMOKE_TEST_PROMPT)],
            tools=[INSPECT_PLAN_TOOL_DEFINITION],
            tool_choice=INSPECT_PLAN_TOOL_CHOICE,
            response_format=None,
        )
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        tool_calls = completion.message.tool_calls or []
        tool_calls_count = len(tool_calls)

        name_matches = False
        args_valid_json = False
        parsed_args = None
        error = None
        error_type = None

        if tool_calls_count != 1:
            error = f"Expected exactly 1 tool call, got {tool_calls_count}"
            error_type = "ToolCallCountError"
        else:
            primary_tc = tool_calls[0]
            fn_dict = primary_tc.get("function", {})
            call_name = fn_dict.get("name") or primary_tc.get("name", "")
            name_matches = (call_name == "inspect_plan")
            if not name_matches:
                error = f"Expected tool name 'inspect_plan', got {call_name!r}"
                error_type = "ToolNameMismatchError"
            else:
                raw_args = fn_dict.get("arguments") or primary_tc.get("arguments", "{}")
                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    if not isinstance(parsed_args, dict):
                        raise ValueError(f"Arguments must be a dict object, got {type(parsed_args).__name__}")
                    args_valid_json = True
                except Exception as exc:
                    args_valid_json = False
                    error = f"Malformed tool arguments JSON: {exc}"
                    error_type = "MalformedArgumentsError"

        success = (tool_calls_count == 1) and name_matches and args_valid_json and (error is None)

        usage_dict = usage_to_dict(completion.usage)
        usage_prov = "provider_reported" if (completion.usage and completion.usage.is_provider_reported) else "unknown"

        payload = {
            "status": "passed" if success else "failed",
            "success": success,
            "test_name": "tool_calling_smoke_test",
            "model_requested": PILOT_MODEL,
            "actual_model_identifier": completion.model_identifier,
            "provider_request_id": completion.provider_request_id,
            "system_fingerprint": completion.system_fingerprint,
            "latency_ms": latency_ms,
            "timestamp_utc": ts_utc,
            "prompt_sent": TOOL_SMOKE_TEST_PROMPT,
            "tool_definitions_sent": [INSPECT_PLAN_TOOL_DEFINITION],
            "tool_choice_sent": INSPECT_PLAN_TOOL_CHOICE,
            "tool_calls_received": tool_calls,
            "tool_calls_count": tool_calls_count,
            "tool_call_name_matches": name_matches,
            "tool_call_arguments_valid_json": args_valid_json,
            "tool_call_arguments_parsed": parsed_args,
            "function_executed": False,
            "state_mutated": False,
            "token_usage": usage_dict,
            "usage_provenance": usage_prov,
            "billing_tier": "free",
            "artifact_file": str(result_path),
            "requests_made": guardrail.total_requests,
            "max_requests_allowed": 1,
            "error": error,
            "error_type": error_type,
        }

    except Exception as exc:
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        exc_type_name = type(exc).__name__
        exc_msg = str(exc)
        for secret in filter(None, [api_key, os.environ.get("GROQ_API_KEY")]):
            exc_msg = exc_msg.replace(secret, "[REDACTED_API_KEY]")

        payload = {
            "status": "failed",
            "success": False,
            "test_name": "tool_calling_smoke_test",
            "error": exc_msg,
            "error_type": exc_type_name,
            "model_requested": PILOT_MODEL,
            "actual_model_identifier": None,
            "provider_request_id": None,
            "system_fingerprint": None,
            "latency_ms": latency_ms,
            "timestamp_utc": ts_utc,
            "prompt_sent": TOOL_SMOKE_TEST_PROMPT,
            "tool_definitions_sent": [INSPECT_PLAN_TOOL_DEFINITION],
            "tool_choice_sent": INSPECT_PLAN_TOOL_CHOICE,
            "tool_calls_received": None,
            "tool_calls_count": 0,
            "tool_call_name_matches": False,
            "tool_call_arguments_valid_json": False,
            "tool_call_arguments_parsed": None,
            "function_executed": False,
            "state_mutated": False,
            "token_usage": None,
            "usage_provenance": None,
            "billing_tier": "free",
            "artifact_file": str(result_path),
            "requests_made": guardrail.total_requests,
            "max_requests_allowed": 1,
        }

    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# =============================================================================
# 8c. STANDALONE ONE-REQUEST CONDITION A JSON SMOKE TEST
# =============================================================================

def run_groq_json_smoke_test(
    api_key: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    smoke_test_json: bool = False,
    confirm_smoke_test_json: bool = False,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a single-request live Condition A JSON smoke test against Groq Cloud API.

    Verifies:
    1. Prompt-directed JSON generation without response_format (mirroring ConditionAAgent).
    2. Parsing of response via json.loads() (with markdown fence stripping).
    3. Strict validation via ConditionAEnvelopeAdapter.
    4. Decision equals 'REJECT'.
    5. Rejection category equals 'allergen_conflict'.
    6. Non-empty explanation reason.
    7. Usage reporting and latency extraction.
    8. Hard limit of 1 HTTP request maximum (zero SDK retries).
    """
    if not (smoke_test_json and confirm_smoke_test_json):
        raise RuntimeError(
            "Standalone Groq JSON smoke test requires explicit dual confirmation: "
            "both --smoke-test-json and --confirm-smoke-test-json flags must be supplied. "
            "Neither flag alone may make a network request."
        )

    load_project_env()
    configured_key = api_key or os.environ.get("GROQ_API_KEY", "")
    key_configured = bool(configured_key and configured_key.strip())

    ts_utc = datetime.now(timezone.utc).isoformat()
    test_run_id = f"groq_smoke_json_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    artifact_dir = Path(output_dir) if output_dir else Path(DEFAULT_JSON_SMOKE_TEST_ARTIFACT_DIR) / test_run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_path = artifact_dir / "json_smoke_test_result.json"

    if not key_configured:
        payload = {
            "status": "failed",
            "success": False,
            "test_name": "json_envelope_smoke_test",
            "error": "GROQ_API_KEY environment variable is not configured.",
            "error_type": "MissingAPIKeyError",
            "model_requested": PILOT_MODEL,
            "actual_model_identifier": None,
            "provider_request_id": None,
            "system_fingerprint": None,
            "latency_ms": 0.0,
            "timestamp_utc": ts_utc,
            "prompt_sent": JSON_SMOKE_TEST_USER_PROMPT,
            "response_format_sent": None,
            "raw_response_text": None,
            "json_parsed_successfully": False,
            "pydantic_envelope_validated": False,
            "envelope_decision": None,
            "envelope_rejection_category": None,
            "envelope_reason": None,
            "token_usage": None,
            "usage_provenance": None,
            "billing_tier": "free",
            "artifact_file": str(result_path),
            "requests_made": 0,
            "max_requests_allowed": 1,
        }
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    provider = GroqProvider(
        api_key=configured_key,
        model=PILOT_MODEL,
        base_url=DEFAULT_BASE_URL,
        timeout_seconds=30.0,
        client=client,
        max_output_tokens=JSON_SMOKE_TEST_OUTPUT_TOKEN_CEILING,
        max_retries=0,
    )
    guardrail = GroqPilotRequestGuardrailProvider(provider=provider, max_requests=1)

    t_start = time.perf_counter()
    try:
        completion = guardrail.complete(
            messages=[
                LLMMessage(role="system", content=JSON_SMOKE_TEST_SYSTEM_PROMPT),
                LLMMessage(role="user", content=JSON_SMOKE_TEST_USER_PROMPT),
            ],
            tools=None,
            tool_choice=None,
            response_format=None,
        )
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        raw_text = completion.message.content or ""

        # Strip markdown fences if present (identical to ConditionAAgent)
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            clean_lines = clean_text.splitlines()
            if clean_lines[0].startswith("```"):
                clean_lines = clean_lines[1:]
            if clean_lines and clean_lines[-1].startswith("```"):
                clean_lines = clean_lines[:-1]
            clean_text = "\n".join(clean_lines).strip()

        json_parsed = False
        pydantic_validated = False
        decision = None
        rejection_cat = None
        reason = None
        error = None
        error_type = None

        try:
            parsed_raw = json.loads(clean_text)
            json_parsed = True
        except Exception as j_exc:
            error = f"Failed to parse JSON from model output: {j_exc}"
            error_type = "JSONDecodeError"

        if json_parsed:
            try:
                envelope = ConditionAEnvelopeAdapter.validate_python(parsed_raw)
                pydantic_validated = True
                decision = getattr(envelope, "decision", None)
                rejection_cat_val = getattr(envelope, "rejection_category", None)
                if isinstance(rejection_cat_val, RejectionCategory):
                    rejection_cat = rejection_cat_val.value
                elif rejection_cat_val is not None:
                    rejection_cat = str(rejection_cat_val)
                reason = getattr(envelope, "reason", None)

                if decision != "REJECT":
                    error = f"Expected decision 'REJECT', got {decision!r}"
                    error_type = "UnexpectedDecisionError"
                elif rejection_cat != RejectionCategory.ALLERGEN_CONFLICT.value:
                    error = f"Expected rejection_category 'allergen_conflict', got {rejection_cat!r}"
                    error_type = "UnexpectedRejectionCategoryError"
                elif not (reason and str(reason).strip()):
                    error = "Expected non-empty rejection reason string."
                    error_type = "EmptyReasonError"

            except ValidationError as v_exc:
                error = f"Condition A envelope validation failed: {v_exc}"
                error_type = "ValidationError"

        success = (
            json_parsed
            and pydantic_validated
            and (decision == "REJECT")
            and (rejection_cat == "allergen_conflict")
            and bool(reason and str(reason).strip())
            and (error is None)
        )

        usage_dict = usage_to_dict(completion.usage)
        usage_prov = "provider_reported" if (completion.usage and completion.usage.is_provider_reported) else "unknown"

        payload = {
            "status": "passed" if success else "failed",
            "success": success,
            "test_name": "json_envelope_smoke_test",
            "model_requested": PILOT_MODEL,
            "actual_model_identifier": completion.model_identifier,
            "provider_request_id": completion.provider_request_id,
            "system_fingerprint": completion.system_fingerprint,
            "latency_ms": latency_ms,
            "timestamp_utc": ts_utc,
            "prompt_sent": JSON_SMOKE_TEST_USER_PROMPT,
            "response_format_sent": None,
            "raw_response_text": raw_text,
            "json_parsed_successfully": json_parsed,
            "pydantic_envelope_validated": pydantic_validated,
            "envelope_decision": decision,
            "envelope_rejection_category": rejection_cat,
            "envelope_reason": reason,
            "token_usage": usage_dict,
            "usage_provenance": usage_prov,
            "billing_tier": "free",
            "artifact_file": str(result_path),
            "requests_made": guardrail.total_requests,
            "max_requests_allowed": 1,
            "error": error,
            "error_type": error_type,
        }

    except Exception as exc:
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        exc_type_name = type(exc).__name__
        exc_msg = str(exc)
        for secret in filter(None, [api_key, os.environ.get("GROQ_API_KEY")]):
            exc_msg = exc_msg.replace(secret, "[REDACTED_API_KEY]")

        payload = {
            "status": "failed",
            "success": False,
            "test_name": "json_envelope_smoke_test",
            "error": exc_msg,
            "error_type": exc_type_name,
            "model_requested": PILOT_MODEL,
            "actual_model_identifier": None,
            "provider_request_id": None,
            "system_fingerprint": None,
            "latency_ms": latency_ms,
            "timestamp_utc": ts_utc,
            "prompt_sent": JSON_SMOKE_TEST_USER_PROMPT,
            "response_format_sent": None,
            "raw_response_text": None,
            "json_parsed_successfully": False,
            "pydantic_envelope_validated": False,
            "envelope_decision": None,
            "envelope_rejection_category": None,
            "envelope_reason": None,
            "token_usage": None,
            "usage_provenance": None,
            "billing_tier": "free",
            "artifact_file": str(result_path),
            "requests_made": guardrail.total_requests,
            "max_requests_allowed": 1,
        }

    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# =============================================================================
# 9. ARTIFACT WRITERS
# =============================================================================

def _patch_pilot_analysis_markdown(
    analysis_path: Path,
    run_id: str,
    is_live: bool,
    guardrail: GroqPilotRequestGuardrailProvider,
    max_requests: int = PROTOCOL_V2_GLOBAL_REQUEST_CEILING,
    incomplete_reason: Optional[str] = None,
) -> None:
    existing = analysis_path.read_text(encoding="utf-8") if analysis_path.exists() else ""
    if is_live and "generated using deterministic offline test providers" in existing:
        offline_phrase = (
            "> This evaluation artifact was generated using deterministic offline test providers. "
            "It verifies the functionality of the execution pipeline, state differ, metrics aggregator, "
            "and report generator. It does NOT constitute empirical evidence of live model performance. "
            "Real benchmarks will be executed in Phase 6.5 using live OpenAI API access."
        )
        live_replacement = (
            "> This evaluation artifact was generated using live cloud API inference against hosted model endpoints. "
            "Task outcomes reflect empirical model decision-making and provider availability under live production rate limits and quota constraints."
        )
        existing = existing.replace(offline_phrase, live_replacement)

    banner = f"""# {PILOT_TAG}
> [!IMPORTANT]
> **LIMITED SAMPLE SIZE NOTICE**: This report represents a controlled preliminary pilot
> consisting of 3 tasks across Conditions A, B, and C (9 runs).
> It is **NOT** the full 25-task benchmark.

- **Pilot Run ID:** `{run_id}`
- **Execution Mode:** `{'LIVE GROQ CLOUD API' if is_live else 'OFFLINE SIMULATED TEST DOUBLE'}`
- **Model Identifier:** `{PILOT_MODEL}`
- **Total Provider Requests Used:** `{guardrail.total_requests}`
- **Request Ceiling:** `{max_requests}`
"""
    if incomplete_reason:
        banner += f"\n> [!WARNING]\n> **INCOMPLETE PILOT**: {incomplete_reason}\n\n"
    banner += "\n---\n\n"
    analysis_path.write_text(banner + existing, encoding="utf-8")


# =============================================================================
# 10. CLI ENTRYPOINT
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    load_project_env()
    parser = argparse.ArgumentParser(description="ShopConnect Groq Pilot Execution Harness")
    parser.add_argument("--preflight", action="store_true", help="Run offline preflight validation")
    parser.add_argument("--dry-run", action="store_true", help="Run offline simulated dry-run (9 runs)")
    parser.add_argument("--live", action="store_true", help="Enable live Groq API pilot")
    parser.add_argument("--confirm-live", action="store_true", help="Confirm live Groq API pilot")
    parser.add_argument("--connection-test", action="store_true", help="Enable one-request connection test")
    parser.add_argument("--confirm-connection-test", action="store_true", help="Confirm one-request connection test")
    parser.add_argument("--smoke-test-tool-call", action="store_true", help="Enable one-request tool calling smoke test")
    parser.add_argument("--confirm-smoke-test-tool-call", action="store_true", help="Confirm one-request tool calling smoke test")
    parser.add_argument("--smoke-test-json", action="store_true", help="Enable one-request Condition A JSON envelope smoke test")
    parser.add_argument("--confirm-smoke-test-json", action="store_true", help="Confirm one-request Condition A JSON envelope smoke test")
    parser.add_argument("--catalog", default="data/catalog_seed.json")
    parser.add_argument("--fixtures", default="data/plan_fixtures.json")
    parser.add_argument("--tasks", default="data/tasks.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--protocol-version",
        default="v2",
        choices=["v1", "v2"],
        help="Protocol version (default: v2)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(PILOT_CONDITIONS),
        choices=["A", "B", "C"],
        help="Conditions to run (default: A B C)",
    )
    parser.add_argument(
        "--enforce-itpm",
        action="store_true",
        default=False,
        help="Enforce strict ITPM prompt size limits in preflight (Condition A rejected if > 7000 tokens)",
    )

    args = parser.parse_args(argv)

    if args.connection_test or args.confirm_connection_test:
        if not (args.connection_test and args.confirm_connection_test):
            print("ERROR: Connection test requires both --connection-test AND --confirm-connection-test.")
            return 1
        res = run_groq_connection_test(output_dir=args.output_dir, connection_test=True, confirm_connection_test=True)
        print(json.dumps(res, indent=2))
        return 0 if res.get("success") else 1

    if args.smoke_test_tool_call or args.confirm_smoke_test_tool_call:
        if not (args.smoke_test_tool_call and args.confirm_smoke_test_tool_call):
            print("ERROR: Tool calling smoke test requires both --smoke-test-tool-call AND --confirm-smoke-test-tool-call.")
            return 1
        res = run_groq_tool_calling_smoke_test(
            output_dir=args.output_dir,
            smoke_test_tool_call=True,
            confirm_smoke_test_tool_call=True,
        )
        print(json.dumps(res, indent=2))
        return 0 if res.get("success") else 1

    if args.smoke_test_json or args.confirm_smoke_test_json:
        if not (args.smoke_test_json and args.confirm_smoke_test_json):
            print("ERROR: JSON smoke test requires both --smoke-test-json AND --confirm-smoke-test-json.")
            return 1
        res = run_groq_json_smoke_test(
            output_dir=args.output_dir,
            smoke_test_json=True,
            confirm_smoke_test_json=True,
        )
        print(json.dumps(res, indent=2))
        return 0 if res.get("success") else 1

    if args.preflight:
        pf = run_groq_pilot_preflight(
            args.catalog,
            args.fixtures,
            args.tasks,
            protocol_version=args.protocol_version,
            conditions=tuple(args.conditions),
            enforce_itpm_limits=args.enforce_itpm,
        )
        print(json.dumps(pf, indent=2))
        return 0 if pf["status"] == "passed" else 1

    if args.dry_run:
        res = run_groq_pilot_dry_run(
            args.catalog,
            args.fixtures,
            args.tasks,
            args.output_dir,
            protocol_version=args.protocol_version,
            conditions=tuple(args.conditions),
        )
        print(json.dumps(res, indent=2))
        return 0 if res["status"] == "completed" else 1

    if args.live or args.confirm_live:
        if not (args.live and args.confirm_live):
            print("ERROR: Live pilot requires both --live AND --confirm-live.")
            return 1
        res = run_live_groq_pilot(
            catalog_path=args.catalog,
            fixtures_path=args.fixtures,
            tasks_path=args.tasks,
            output_dir=args.output_dir,
            live=True,
            confirm_live=True,
            protocol_version=args.protocol_version,
            conditions=tuple(args.conditions),
            enforce_itpm_limits=True,
        )
        print(json.dumps(res, indent=2))
        if res.get("status") == "interrupted":
            return 130
        return 0 if res["status"] == "completed" else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
