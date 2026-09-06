"""Common / shared schemas."""

from typing import Any, Optional
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str  # "healthy" | "degraded" | "unhealthy"
    service: str = "api-gateway"
    checks: dict[str, Any] = {}


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
