"""Quota service — daily and monthly token budget enforcement.

Redis counters are fast but ephemeral.
PostgreSQL provides durable daily/monthly totals for billing and auditing.

Daily quota keys in Redis:
  quota:daily:{user_id}:{YYYY-MM-DD}   TTL = 25 hours

Monthly quota keys in Redis:
  quota:monthly:{user_id}:{YYYY-MM}    TTL = 32 days

We use Redis as the hot check for daily/monthly limits so we don't hit
PostgreSQL on every request. If Redis loses data (restart), counters rebuild
from PostgreSQL UsageEvents on the next warm-up.
"""

from datetime import datetime, timezone, timedelta

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.gateway.core.config import get_settings
from src.gateway.core.logging import get_logger
from src.gateway.infrastructure.redis import get_counter, increment_token_counter
from src.gateway.models.quota import Quota
from src.gateway.models.usage_event import UsageEvent
from src.gateway.models.user import User

logger = get_logger("quota_service")


def _daily_key(user_id: str) -> str:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"quota:daily:{user_id}:{date_str}"


def _monthly_key(user_id: str) -> str:
    month_str = datetime.now(timezone.utc).strftime("%Y-%m")
    return f"quota:monthly:{user_id}:{month_str}"


class QuotaService:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def get_quota_by_user_id(self, user_id: str) -> Quota:
        """Load the user's quota config by user_id, falling back to defaults."""
        result = await self._db.execute(
            select(Quota).where(Quota.user_id == user_id)
        )
        quota = result.scalar_one_or_none()
        if quota:
            return quota

        user_res = await self._db.execute(select(User).where(User.id == user_id))
        user = user_res.scalar_one_or_none()
        plan = user.plan if user else "developer"
        settings = get_settings()
        defaults = _plan_defaults(plan, settings)
        quota = Quota(user_id=user_id, **defaults)
        self._db.add(quota)
        await self._db.flush()
        return quota

    async def get_quota(self, user: User) -> Quota:
        """Load the user's quota config, falling back to defaults."""
        result = await self._db.execute(
            select(Quota).where(Quota.user_id == user.id)
        )
        quota = result.scalar_one_or_none()
        if quota:
            return quota

        # Auto-create default quota for the user's plan
        settings = get_settings()
        defaults = _plan_defaults(user.plan, settings)
        quota = Quota(user_id=user.id, **defaults)
        self._db.add(quota)
        await self._db.flush()
        return quota

    async def check_daily_quota(self, user_id: str, quota: Quota) -> None:
        """Raise 429 if the user has exhausted their daily token budget."""
        used = await get_counter(_daily_key(user_id))
        if used >= quota.daily_tokens:
            logger.warning(
                "daily_quota_exceeded",
                user_id=user_id,
                used=used,
                limit=quota.daily_tokens,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "daily_quota_exceeded",
                    "used_tokens": used,
                    "daily_limit": quota.daily_tokens,
                    "resets_at": "midnight UTC",
                },
                headers={"Retry-After": str(_seconds_until_midnight())},
            )

    async def check_monthly_quota(self, user_id: str, quota: Quota) -> None:
        """Raise 429 if the user has exhausted their monthly token budget."""
        used = await get_counter(_monthly_key(user_id))
        if used >= quota.monthly_tokens:
            logger.warning(
                "monthly_quota_exceeded",
                user_id=user_id,
                used=used,
                limit=quota.monthly_tokens,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "monthly_quota_exceeded",
                    "used_tokens": used,
                    "monthly_limit": quota.monthly_tokens,
                    "resets_at": "1st of next month UTC",
                },
            )

    async def consume_tokens(self, user_id: str, tokens: int) -> None:
        """Atomically increment daily and monthly Redis counters."""
        daily_key = _daily_key(user_id)
        monthly_key = _monthly_key(user_id)

        await increment_token_counter(daily_key, tokens, ttl_seconds=90_000)    # 25h
        await increment_token_counter(monthly_key, tokens, ttl_seconds=2_764_800)  # 32d

    async def get_daily_usage(self, user_id: str) -> int:
        return await get_counter(_daily_key(user_id))

    async def get_monthly_usage(self, user_id: str) -> int:
        return await get_counter(_monthly_key(user_id))

    async def get_total_usage_from_db(self, user_id: str) -> dict:
        """Durable total from PostgreSQL — used for usage API and billing."""
        result = await self._db.execute(
            select(
                func.count(UsageEvent.id).label("total_requests"),
                func.coalesce(func.sum(UsageEvent.total_tokens), 0).label("total_tokens"),
                func.coalesce(func.sum(UsageEvent.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(UsageEvent.output_tokens), 0).label("output_tokens"),
                func.avg(UsageEvent.latency_ms).label("avg_latency_ms"),
                func.avg(UsageEvent.ttft_ms).label("avg_ttft_ms"),
            ).where(UsageEvent.user_id == user_id)
        )
        row = result.one()
        return {
            "total_requests": row.total_requests,
            "total_tokens": int(row.total_tokens),
            "input_tokens": int(row.input_tokens),
            "output_tokens": int(row.output_tokens),
            "avg_latency_ms": float(row.avg_latency_ms) if row.avg_latency_ms else None,
            "avg_ttft_ms": float(row.avg_ttft_ms) if row.avg_ttft_ms else None,
        }


def _seconds_until_midnight() -> int:
    now = datetime.now(timezone.utc)
    midnight = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return int((midnight - now).total_seconds())


def _plan_defaults(plan: str, settings) -> dict:
    plans = {
        "free": {
            "plan": "free",
            "rpm": settings.default_rpm,
            "tpm": settings.default_tpm,
            "daily_tokens": settings.default_daily_tokens,
            "monthly_tokens": settings.default_monthly_tokens,
            "max_concurrent": settings.default_max_concurrent,
        },
        "pro": {
            "plan": "pro",
            "rpm": 60,
            "tpm": 500_000,
            "daily_tokens": 1_000_000,
            "monthly_tokens": 20_000_000,
            "max_concurrent": 10,
        },
        "enterprise": {
            "plan": "enterprise",
            "rpm": 600,
            "tpm": 5_000_000,
            "daily_tokens": 50_000_000,
            "monthly_tokens": 500_000_000,
            "max_concurrent": 50,
        },
    }
    return plans.get(plan, plans["free"])
