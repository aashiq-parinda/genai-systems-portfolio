"""Models, usage, and health routes."""

import time

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.gateway.infrastructure.database import get_db
from src.gateway.infrastructure.redis import get_redis
from src.gateway.infrastructure.vllm_client import VLLMError, get_vllm_client
from src.gateway.schemas.common import HealthResponse, ModelInfo, ModelList
from src.gateway.schemas.usage import QuotaStatus, UsageSummary
from src.gateway.services.model_router import GatewayModelRouter
from src.gateway.services.quota_service import QuotaService

_model_router = GatewayModelRouter()

# ── Models ─────────────────────────────────────────────────────────────────────
models_router = APIRouter(prefix="/v1", tags=["Models"])


@models_router.get("/models", response_model=ModelList)
async def list_models():
    """List available inference models."""
    models = _model_router.list_models()
    return ModelList(data=[ModelInfo(**m) for m in models])


# ── Usage ──────────────────────────────────────────────────────────────────────
usage_router = APIRouter(prefix="/v1", tags=["Usage"])


@usage_router.get("/usage", response_model=UsageSummary)
async def get_usage(request: Request, db: AsyncSession = Depends(get_db)):
    """Return aggregated usage statistics for the authenticated user."""
    user = request.state.user
    quota_service = QuotaService(db)
    totals = await quota_service.get_total_usage_from_db(user.id)
    daily = await quota_service.get_daily_usage(user.id)
    monthly = await quota_service.get_monthly_usage(user.id)

    return UsageSummary(
        user_id=user.id,
        period="all_time",
        total_requests=totals["total_requests"],
        successful_requests=totals["total_requests"],  # simplified for education
        failed_requests=0,
        total_tokens=totals["total_tokens"],
        input_tokens=totals["input_tokens"],
        output_tokens=totals["output_tokens"],
        avg_latency_ms=totals["avg_latency_ms"],
        avg_ttft_ms=totals["avg_ttft_ms"],
    )


@usage_router.get("/usage/quota", response_model=QuotaStatus)
async def get_quota_status(request: Request, db: AsyncSession = Depends(get_db)):
    """Return live quota status including current usage against limits."""
    user = request.state.user
    quota_service = QuotaService(db)
    quota = await quota_service.get_quota(user)
    daily_used = await quota_service.get_daily_usage(user.id)
    monthly_used = await quota_service.get_monthly_usage(user.id)

    return QuotaStatus(
        plan=quota.plan,
        rpm_limit=quota.rpm,
        rpm_used=0,  # live RPM from Redis could be added here
        tpm_limit=quota.tpm,
        daily_tokens_limit=quota.daily_tokens,
        daily_tokens_used=daily_used,
        monthly_tokens_limit=quota.monthly_tokens,
        monthly_tokens_used=monthly_used,
        max_concurrent=quota.max_concurrent,
        current_concurrent=0,  # live concurrency from Redis could be added here
    )


# ── Health ─────────────────────────────────────────────────────────────────────
health_router = APIRouter(tags=["Health"])


@health_router.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe — returns 200 if the process is running."""
    return HealthResponse(status="healthy")


@health_router.get("/ready", response_model=HealthResponse)
async def ready(db: AsyncSession = Depends(get_db)):
    """Readiness probe — checks Postgres, Redis, and inference backend."""
    checks: dict = {}

    # PostgreSQL
    try:
        from sqlalchemy import text
        await db.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"

    # Redis
    try:
        r = get_redis()
        await r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    # Inference backend
    try:
        client = get_vllm_client()
        await client.list_models()
        checks["inference"] = "ok"
    except VLLMError as e:
        checks["inference"] = f"degraded: {e.detail}"
    except Exception as e:
        checks["inference"] = f"error: {e}"

    all_ok = all("ok" in v for v in checks.values())
    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        checks=checks,
    )


@health_router.get("/metrics")
async def metrics():
    """Prometheus-compatible text metrics endpoint."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
