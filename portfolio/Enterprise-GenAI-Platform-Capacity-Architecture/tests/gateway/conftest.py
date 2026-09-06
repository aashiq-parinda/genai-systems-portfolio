"""Pytest fixtures for the gateway test suite.

Uses:
  - AsyncClient (httpx) for async route testing
  - SQLite in-memory via aiosqlite (no Docker needed)
  - FakeRedis (fakeredis) to avoid a real Redis dependency in tests
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from unittest.mock import AsyncMock, patch

from src.gateway.main import create_app
from src.gateway.infrastructure.database import Base, get_session_factory, get_db
from src.gateway.models.user import User
from src.gateway.models.quota import Quota
from src.gateway.models.api_key import APIKey
from src.gateway.core.config import get_settings
from src.gateway.core.security import generate_api_key

from sqlalchemy.pool import StaticPool
import src.gateway.infrastructure.database as db_module

# ── SQLite test DB ─────────────────────────────────────────────────────────────

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def db_engine():
    engine = create_async_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    old_engine = db_module._engine
    old_factory = db_module._session_factory
    db_module._engine = engine
    db_module._session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
    db_module._engine = old_engine
    db_module._session_factory = old_factory


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine):
    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()


# ── Test user and API key ──────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    user = User(
        email="test@example.com",
        verified_email=True,
        verified_phone=False,
        plan="pro",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()

    quota = Quota(
        user_id=user.id,
        plan="pro",
        rpm=60,
        tpm=500_000,
        daily_tokens=1_000_000,
        monthly_tokens=20_000_000,
        max_concurrent=10,
    )
    db_session.add(quota)
    await db_session.commit()
    return user


@pytest_asyncio.fixture
async def test_api_key(db_session: AsyncSession, test_user: User) -> tuple[str, APIKey]:
    """Returns (raw_key, APIKey model)."""
    settings = get_settings()
    raw_key, key_hash, key_prefix = generate_api_key(settings)
    api_key = APIKey(
        user_id=test_user.id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name="test-key",
        scopes="chat,admin",
        is_active=True,
    )
    db_session.add(api_key)
    await db_session.commit()
    return raw_key, api_key


# ── FastAPI test client ────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(db_session: AsyncSession, test_api_key):
    """Async test client with DB and Redis mocked."""
    raw_key, api_key = test_api_key

    # Override the DB dependency to use the test session
    async def override_get_db():
        yield db_session

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db

    # Patch Redis to avoid real connection in tests
    with patch("src.gateway.infrastructure.redis._redis_pool") as mock_redis:
        mock_redis.eval = AsyncMock(return_value=[1, 1])  # allow by default
        mock_redis.incr = AsyncMock(return_value=1)
        mock_redis.decr = AsyncMock(return_value=0)
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.incrby = AsyncMock(return_value=1)
        mock_redis.expire = AsyncMock(return_value=True)
        mock_redis.ping = AsyncMock(return_value=True)
        pipeline_mock = AsyncMock()
        pipeline_mock.execute = AsyncMock(return_value=[1, True])
        mock_redis.pipeline = AsyncMock(return_value=pipeline_mock)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac, raw_key
