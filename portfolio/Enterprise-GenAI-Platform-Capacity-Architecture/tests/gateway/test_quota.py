"""Tests: Token-based quotas — within/exceeded/daily reset."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException

from src.gateway.services.quota_service import QuotaService, _daily_key, _monthly_key


@pytest.mark.asyncio
async def test_daily_quota_within_limit(db_session, test_user):
    """Within daily token budget → check passes without raising."""
    service = QuotaService(db_session)
    quota = await service.get_quota(test_user)

    with patch(
        "src.gateway.services.quota_service.get_counter",
        new_callable=AsyncMock,
        return_value=0,  # 0 tokens used today
    ):
        # Should not raise
        await service.check_daily_quota(test_user.id, quota)


@pytest.mark.asyncio
async def test_daily_quota_exceeded_raises_429(db_session, test_user):
    """Exceeding daily token budget → 429 with correct detail."""
    service = QuotaService(db_session)
    quota = await service.get_quota(test_user)

    with patch(
        "src.gateway.services.quota_service.get_counter",
        new_callable=AsyncMock,
        return_value=quota.daily_tokens + 1,  # over limit
    ):
        with pytest.raises(HTTPException) as exc_info:
            await service.check_daily_quota(test_user.id, quota)

    exc = exc_info.value
    assert exc.status_code == 429
    assert exc.detail["error"] == "daily_quota_exceeded"
    assert "Retry-After" in exc.headers


@pytest.mark.asyncio
async def test_monthly_quota_exceeded_raises_429(db_session, test_user):
    """Exceeding monthly token budget → 429."""
    service = QuotaService(db_session)
    quota = await service.get_quota(test_user)

    with patch(
        "src.gateway.services.quota_service.get_counter",
        new_callable=AsyncMock,
        return_value=quota.monthly_tokens + 1,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await service.check_monthly_quota(test_user.id, quota)

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["error"] == "monthly_quota_exceeded"


@pytest.mark.asyncio
async def test_token_consumption_increments_counters(db_session, test_user):
    """consume_tokens() must call increment on both daily and monthly keys."""
    service = QuotaService(db_session)

    with patch(
        "src.gateway.services.quota_service.increment_token_counter",
        new_callable=AsyncMock,
        return_value=500,
    ) as mock_incr:
        await service.consume_tokens(test_user.id, 500)

    assert mock_incr.call_count == 2  # daily + monthly


@pytest.mark.asyncio
async def test_new_user_gets_default_quota(db_session, test_user):
    """Users with no explicit quota record get default free-tier limits."""
    from src.gateway.models.user import User
    import uuid

    new_user = User(
        id=str(uuid.uuid4()),
        email="new@example.com",
        plan="free",
        is_active=True,
    )
    db_session.add(new_user)
    await db_session.flush()

    service = QuotaService(db_session)
    quota = await service.get_quota(new_user)

    assert quota.plan == "free"
    assert quota.rpm == 10
    assert quota.daily_tokens == 50_000
    assert quota.max_concurrent == 2


@pytest.mark.asyncio
async def test_pro_plan_gets_higher_quota(db_session, test_user):
    """Pro plan user (from fixture) gets pro-tier limits."""
    service = QuotaService(db_session)
    quota = await service.get_quota(test_user)

    assert quota.plan == "pro"
    assert quota.rpm == 60
    assert quota.daily_tokens == 1_000_000
    assert quota.max_concurrent == 10
