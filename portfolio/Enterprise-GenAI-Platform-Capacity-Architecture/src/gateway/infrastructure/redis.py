"""Redis connection pool and atomic operation helpers.

Why Redis alongside PostgreSQL?
- Redis provides sub-millisecond counter increments — critical for rate limiting
  where every request must check and update counters before proceeding.
- PostgreSQL is durable but ~5-10ms per write. Using it for per-request counters
  would become a bottleneck under load.
- Redis is ephemeral (can be lost on restart) — acceptable for short-lived counters.
- PostgreSQL stores durable usage events after the fact.
- The two systems are complementary, not redundant.

Atomic operations via Lua scripts:
- Redis is single-threaded, but network round-trips between CHECK and INCR
  create a race condition for concurrent requests.
- Lua scripts run atomically on the server, eliminating the race.
"""

from typing import Any

import redis.asyncio as aioredis
from redis.asyncio import Redis

from src.gateway.core.config import get_settings
from src.gateway.core.logging import get_logger

logger = get_logger("redis")

_redis_pool: Redis | None = None


def get_redis() -> Redis:
    """Return singleton Redis client with connection pool."""
    global _redis_pool
    if _redis_pool is None:
        settings = get_settings()
        _redis_pool = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=settings.redis_pool_size,
        )
        logger.info("redis_pool_created", url=settings.redis_url)
    return _redis_pool


async def close_redis() -> None:
    """Graceful shutdown — release all connections in the pool."""
    global _redis_pool
    if _redis_pool:
        await _redis_pool.aclose()
        _redis_pool = None
        logger.info("redis_pool_closed")


# ── Lua Scripts (atomic rate-limit operations) ─────────────────────────────────

# Sliding window rate limit check + increment.
# KEYS[1] = rate limit key  (e.g. "rl:rpm:user_id:window_bucket")
# ARGV[1] = window TTL in seconds
# ARGV[2] = max allowed count
# Returns: [current_count, allowed (1=yes, 0=no)]
_SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local ttl = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])

local current = redis.call('INCR', key)
if current == 1 then
    redis.call('EXPIRE', key, ttl)
end

if current > limit then
    return {current, 0}
else
    return {current, 1}
end
"""

# Atomic concurrent slot acquire.
# KEYS[1] = concurrency counter key
# ARGV[1] = max concurrent
# ARGV[2] = TTL in seconds (safety net to prevent leaked slots)
# Returns: [current_count, acquired (1=yes, 0=no)]
_ACQUIRE_SLOT_SCRIPT = """
local key = KEYS[1]
local max = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])

local current = redis.call('INCR', key)
if current == 1 then
    redis.call('EXPIRE', key, ttl)
end

if current > max then
    redis.call('DECR', key)
    return {current - 1, 0}
else
    return {current, 1}
end
"""


async def sliding_window_check(
    key: str,
    window_seconds: int,
    max_count: int,
) -> tuple[int, bool]:
    """Atomically increment a sliding window counter and check the limit.

    Returns (current_count, is_allowed).
    The window resets after window_seconds from the first request in that bucket.
    """
    r = get_redis()
    result: list[int] = await r.eval(
        _SLIDING_WINDOW_SCRIPT, 1, key, window_seconds, max_count
    )
    return int(result[0]), bool(result[1])


async def acquire_concurrency_slot(
    key: str,
    max_concurrent: int,
    ttl_seconds: int = 300,
) -> tuple[int, bool]:
    """Atomically increment the concurrent request counter.

    Returns (current_count, acquired).
    Call release_concurrency_slot() in a finally block.
    TTL is a safety net so crashed workers don't permanently block slots.
    """
    r = get_redis()
    result: list[int] = await r.eval(
        _ACQUIRE_SLOT_SCRIPT, 1, key, max_concurrent, ttl_seconds
    )
    return int(result[0]), bool(result[1])


async def release_concurrency_slot(key: str) -> None:
    """Decrement the concurrent request counter. Always call in finally."""
    r = get_redis()
    await r.decr(key)


async def increment_token_counter(key: str, tokens: int, ttl_seconds: int) -> int:
    """Add token count to a rolling counter. Returns new total."""
    r = get_redis()
    pipe = r.pipeline()
    await pipe.incrby(key, tokens)
    await pipe.expire(key, ttl_seconds, nx=True)  # only set TTL if not already set
    results: list[Any] = await pipe.execute()
    return int(results[0])


async def get_counter(key: str) -> int:
    """Read a counter value (0 if key doesn't exist)."""
    r = get_redis()
    value = await r.get(key)
    return int(value) if value else 0
