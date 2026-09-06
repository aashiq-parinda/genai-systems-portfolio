"""API Gateway FastAPI application factory.

This is the public API key gateway — distinct from the enterprise control plane
at src/control_plane/gateway.py which uses JWT + Bot Registry authentication.

Both share DynamicModelRouter and EnterpriseGuardrails.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.gateway.core.config import get_settings
from src.gateway.core.logging import configure_logging, get_logger
from src.gateway.infrastructure.database import create_all_tables, dispose_engine
from src.gateway.infrastructure.redis import close_redis
from src.gateway.infrastructure.vllm_client import close_vllm_client
from src.gateway.middleware.auth import AuthMiddleware
from src.gateway.middleware.request_id import RequestIDMiddleware
from src.gateway.api.routes.auth import router as auth_router
from src.gateway.api.routes.chat import router as chat_router
from src.gateway.api.routes.health import (
    health_router,
    models_router,
    usage_router,
)

logger = get_logger("gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    configure_logging()
    settings = get_settings()
    logger.info(
        "gateway_starting",
        env=settings.gateway_env,
        inference_backend=settings.inference_backend,
        port=settings.gateway_port,
    )

    # In development, auto-create tables.
    # In production, Alembic migrations handle schema changes.
    if settings.gateway_env != "production":
        await create_all_tables()
        await _seed_dev_user()

    logger.info("gateway_ready")
    yield

    # Graceful shutdown
    logger.info("gateway_shutting_down")
    await dispose_engine()
    await close_redis()
    await close_vllm_client()
    logger.info("gateway_stopped")


async def _seed_dev_user() -> None:
    """Create a test user + API key for local development.

    Prints the API key to stdout so developers can use it immediately
    after `docker compose up` without manual DB setup.
    """
    from src.gateway.infrastructure.database import db_session
    from src.gateway.models.user import User
    from src.gateway.models.quota import Quota
    from src.gateway.services.auth_service import AuthService
    from src.gateway.schemas.auth import CreateKeyRequest
    from sqlalchemy import select

    async with db_session() as db:
        result = await db.execute(select(User).where(User.email == "dev@localhost"))
        existing = result.scalar_one_or_none()
        if existing:
            return  # already seeded

        user = User(
            email="dev@localhost",
            verified_email=True,
            verified_phone=False,
            plan="pro",
            is_active=True,
        )
        db.add(user)
        await db.flush()

        quota = Quota(user_id=user.id, plan="pro", rpm=60, tpm=500_000,
                      daily_tokens=1_000_000, monthly_tokens=20_000_000, max_concurrent=10)
        db.add(quota)
        await db.flush()

        auth_service = AuthService(db)
        key_response = await auth_service.create_key(
            user=user,
            req=CreateKeyRequest(name="dev-key", scopes=["chat", "admin"]),
            request_id="startup",
        )

    print("\n" + "=" * 70)
    print("  DEV API KEY (only shown once):")
    print(f"  {key_response.raw_key}")
    print("=" * 70)
    print(f"  export LLM_API_KEY={key_response.raw_key}")
    print("=" * 70 + "\n")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Production LLM API Gateway",
        description=(
            "OpenAI-compatible API gateway for self-hosted LLMs. "
            "Part of the Enterprise GenAI Platform portfolio. "
            "See docs/07_API_GATEWAY_LAYER.md for architecture details."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware (order matters — outermost runs first) ──────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # ── Routes ─────────────────────────────────────────────────────────────────
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(models_router)
    app.include_router(usage_router)
    app.include_router(health_router)

    return app


# Uvicorn / Docker entrypoint
app = create_app()
