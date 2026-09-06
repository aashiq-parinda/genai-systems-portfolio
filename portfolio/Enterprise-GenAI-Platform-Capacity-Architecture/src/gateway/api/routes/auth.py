"""Auth routes — API key management."""

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.gateway.infrastructure.database import get_db
from src.gateway.schemas.auth import APIKeyInfo, CreateKeyRequest, CreateKeyResponse
from src.gateway.services.auth_service import AuthService

router = APIRouter(prefix="/v1/auth", tags=["Authentication"])


@router.post("/keys", response_model=CreateKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    req: CreateKeyRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Create a new API key. The raw key is shown exactly once — store it securely."""
    user = request.state.user
    service = AuthService(db)
    return await service.create_key(
        user=user,
        req=req,
        request_id=request.state.request_id,
    )


@router.get("/keys", response_model=list[APIKeyInfo])
async def list_api_keys(request: Request, db: AsyncSession = Depends(get_db)):
    """List all API keys for the authenticated user (no raw keys returned)."""
    service = AuthService(db)
    return await service.list_keys(request.state.user)


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Revoke an API key. The key is deactivated but retained for audit history."""
    service = AuthService(db)
    await service.revoke_key(
        user=request.state.user,
        key_id=key_id,
        request_id=request.state.request_id,
    )
