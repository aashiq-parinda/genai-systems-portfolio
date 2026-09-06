"""Async SQLAlchemy 2 database engine and session factory.

Uses asyncpg for PostgreSQL in production, aiosqlite for pytest.
The DATABASE_URL env var controls which driver is used.

Why async? LLM API servers are I/O bound — waiting for inference, Redis,
and DB writes. Async I/O lets a single-threaded event loop handle hundreds
of concurrent requests without thread overhead.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.gateway.core.config import get_settings
from src.gateway.core.logging import get_logger

logger = get_logger("database")


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def _build_engine():
    settings = get_settings()
    url = settings.database_url

    # SQLite (pytest / dev) doesn't support pool configuration
    is_sqlite = "sqlite" in url
    kwargs = {} if is_sqlite else {
        "pool_size": settings.database_pool_size,
        "max_overflow": settings.database_max_overflow,
        "pool_pre_ping": True,  # drop stale connections silently
    }

    engine = create_async_engine(url, echo=False, **kwargs)
    logger.info("database_engine_created", url_scheme=url.split("://")[0])
    return engine


# Module-level singletons — created once at import time, reused across requests.
_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _session_factory


@asynccontextmanager
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for manual session use (outside request scope)."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — injects an AsyncSession into route handlers."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_all_tables() -> None:
    """Create all tables. Used in tests and dev startup. Alembic handles prod."""
    engine = get_engine()
    # Import all models so Base.metadata knows about them
    from src.gateway.models import api_key, audit_event, quota, usage_event, user  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_tables_created")


async def dispose_engine() -> None:
    """Clean up the connection pool on shutdown."""
    if _engine:
        await _engine.dispose()
        logger.info("database_engine_disposed")
