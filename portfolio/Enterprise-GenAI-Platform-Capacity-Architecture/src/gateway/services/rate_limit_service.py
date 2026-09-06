"""Distributed rate limiting service using Redis sliding window counters.

Algorithm: Sliding Window (approximated with fixed-TTL buckets)
─────────────────────────────────────────────────────────────────
Each request increments a counter keyed to a 60-second window.
The counter expires after 60 seconds (sliding the window forward).

Tradeoff vs Token Bucket:
  Sliding Window:
    ✓ Simple to reason about (X requests in the last 60 seconds)
    ✓ Atomic in Redis with a single INCR + EXPIRE Lua script
    ✗ Allows burst of 2x limit at window boundaries
        (N requests at t=59s + N requests at t=61s = 2N in 2 seconds)

  Token Bucket:
    ✓ Smoother — limits burst more precisely
    ✓ Allows short bursts up to bucket size, then refills at a steady rate
    ✗ Requires storing bucket fill level + last_refill timestamp
    ✗ More complex Lua script for atomic check-and-consume

  For an LLM API gateway, sliding window is typically sufficient because:
  - LLM requests are inherently slow (seconds, not milliseconds)
  - The boundary-burst window is small relative to inference time
  - Simplicity is a maintenance advantage

Rate limit keys:
  rl:rpm:{user_id}         — requests per minute
  rl:tpm:{user_id}         — tokens per minute
  conc:{user_id}           — current concurrent request count
"""

from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, status

from src.gateway.core.logging import get_logger
from src.gateway.infrastructure.redis import (
    acquire_concurrency_slot,
    release_concurrency_slot,
    sliding_window_check,
)
from src.gateway.models.quota import Quota

logger = get_logger("rate_limit_service")


@dataclass
class RateLimitResult:
    allowed: bool
    current_rpm: int
    limit_rpm: int
    retry_after_seconds: Optional[int] = None
    reason: Optional[str] = None


class RateLimitService:
    RPM_WINDOW = 60       # seconds
    TPM_WINDOW = 60       # seconds

    def _rpm_key(self, user_id: str) -> str:
        return f"rl:rpm:{user_id}"

    def _tpm_key(self, user_id: str) -> str:
        return f"rl:tpm:{user_id}"

    def _concurrency_key(self, user_id: str) -> str:
        return f"conc:{user_id}"

    async def check_rpm(self, user_id: str, quota: Quota) -> RateLimitResult:
        """Check and increment the requests-per-minute counter."""
        count, allowed = await sliding_window_check(
            key=self._rpm_key(user_id),
            window_seconds=self.RPM_WINDOW,
            max_count=quota.rpm,
        )
        return RateLimitResult(
            allowed=allowed,
            current_rpm=count,
            limit_rpm=quota.rpm,
            retry_after_seconds=self.RPM_WINDOW if not allowed else None,
            reason="rpm_exceeded" if not allowed else None,
        )

    async def check_tpm(self, user_id: str, token_estimate: int, quota: Quota) -> bool:
        """Check (but do not consume) token-per-minute budget.

        Consumption happens after inference when actual token counts are known.
        This pre-check uses the request's estimated input token count as a
        conservative lower bound to fail fast before expensive inference.
        """
        _, allowed = await sliding_window_check(
            key=self._tpm_key(user_id),
            window_seconds=self.TPM_WINDOW,
            max_count=quota.tpm,
        )
        return allowed

    async def record_tokens(self, user_id: str, tokens: int) -> None:
        """Record actual token usage against the TPM counter post-inference."""
        await sliding_window_check(
            key=self._tpm_key(user_id),
            window_seconds=self.TPM_WINDOW,
            max_count=999_999_999,  # just increment, don't enforce here
        )

    async def acquire_slot(self, user_id: str, quota: Quota) -> bool:
        """Atomically acquire a concurrency slot.

        Returns True if acquired. Caller MUST call release_slot() in a
        finally block to prevent slot leaks.

        Why limit concurrency?
        - Each active inference request holds GPU VRAM (KV-cache pages).
        - Unlimited concurrent requests → GPU OOM or starvation for other users.
        - This is admission control at the API layer, complementing the
          vLLM scheduler's internal queue.
        """
        _, acquired = await acquire_concurrency_slot(
            key=self._concurrency_key(user_id),
            max_concurrent=quota.max_concurrent,
            ttl_seconds=300,  # safety TTL — cleans up leaked slots after 5min
        )
        if not acquired:
            logger.warning(
                "concurrency_limit_exceeded",
                user_id=user_id,
                max_concurrent=quota.max_concurrent,
            )
        return acquired

    async def release_slot(self, user_id: str) -> None:
        """Release a previously acquired concurrency slot."""
        await release_concurrency_slot(self._concurrency_key(user_id))

    def raise_429(self, result: RateLimitResult) -> None:
        """Raise a standardised 429 response with rate limit headers."""
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "reason": result.reason,
                "limit": result.limit_rpm,
                "current": result.current_rpm,
            },
            headers={
                "Retry-After": str(result.retry_after_seconds or 60),
                "X-RateLimit-Limit": str(result.limit_rpm),
                "X-RateLimit-Remaining": str(max(0, result.limit_rpm - result.current_rpm)),
                "X-RateLimit-Reset": str(result.retry_after_seconds or 60),
            },
        )
