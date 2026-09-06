"""Request ID middleware.

Injects a unique request_id (UUID4) into:
  - request.state.request_id (available to all downstream handlers)
  - structlog context (appears in every log line for this request)
  - X-Request-ID response header (for client-side correlation)
"""

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        # Bind request_id to structlog context so every log line in this
        # request automatically includes it — no manual passing required.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
