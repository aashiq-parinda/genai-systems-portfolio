"""Auth middleware — validates Bearer API key and attaches user context.

This middleware runs on every request to protected routes. It:
  1. Extracts the Bearer token from Authorization header
  2. Validates it against the DB (via AuthService)
  3. Attaches (user, api_key) to request.state
  4. Logs failures WITHOUT including the raw token

Public paths (health, metrics, docs) bypass this middleware.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.gateway.core.logging import get_logger
from src.gateway.infrastructure.database import get_session_factory
from src.gateway.services.auth_service import AuthService
from src.gateway.services.identity_service import IdentityService, make_ip_bucket

logger = get_logger("auth_middleware")

# Routes that don't require authentication
PUBLIC_PATHS = {
    "/health",
    "/ready",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # Allow public routes through
        if request.url.path in PUBLIC_PATHS or request.url.path.startswith("/docs"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "missing_auth", "detail": "Authorization: Bearer llm_xxx required"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        raw_key = auth_header[7:]  # strip "Bearer "
        request_id = getattr(request.state, "request_id", "unknown")

        factory = get_session_factory()
        async with factory() as db:
            auth_service = AuthService(db)
            result = await auth_service.validate_raw_key(raw_key)

            if result is None:
                ip = request.client.host if request.client else "unknown"
                ip_bucket = make_ip_bucket(ip)
                await auth_service.record_failed_auth(request_id, ip_bucket)
                await db.commit()
                # Never log the raw key — log only that auth failed
                logger.warning("auth_rejected", request_id=request_id, ip_bucket=ip_bucket)
                return JSONResponse(
                    status_code=401,
                    content={"error": "invalid_api_key", "detail": "API key invalid or revoked"},
                    headers={"WWW-Authenticate": "Bearer"},
                )

            api_key, user = result

            # Attach to request state for downstream handlers
            request.state.user = user
            request.state.api_key = api_key
            request.state.ip_bucket = make_ip_bucket(
                request.client.host if request.client else "0.0.0.0"
            )

            await db.commit()

        return await call_next(request)
