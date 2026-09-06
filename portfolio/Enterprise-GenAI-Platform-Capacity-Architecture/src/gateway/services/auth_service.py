"""Auth service — API key CRUD and validation.

This is the only service that knows about key hashing.
Routes never touch key_hash directly.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.gateway.core.config import get_settings
from src.gateway.core.logging import get_logger
from src.gateway.core.security import generate_api_key, verify_api_key
from src.gateway.models.api_key import APIKey
from src.gateway.models.audit_event import AuditEvent
from src.gateway.models.user import User
from src.gateway.schemas.auth import CreateKeyRequest, CreateKeyResponse, APIKeyInfo

logger = get_logger("auth_service")


class AuthService:
    def __init__(self, db: AsyncSession):
        self._db = db
        self._settings = get_settings()

    async def create_key(
        self,
        user: User,
        req: CreateKeyRequest,
        request_id: str,
    ) -> CreateKeyResponse:
        """Generate and store a new API key. Returns raw key exactly once."""
        raw_key, key_hash, key_prefix = generate_api_key(self._settings)

        api_key = APIKey(
            user_id=user.id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=req.name,
            scopes=",".join(req.scopes),
        )
        self._db.add(api_key)

        audit = AuditEvent(
            request_id=request_id,
            user_id=user.id,
            api_key_id=api_key.id,
            action="api_key_created",
            detail=f"scopes={req.scopes}",
        )
        self._db.add(audit)
        await self._db.flush()

        logger.info(
            "api_key_created",
            user_id=user.id,
            key_prefix=key_prefix,
            request_id=request_id,
        )

        return CreateKeyResponse(
            id=api_key.id,
            raw_key=raw_key,  # shown once — never logged
            key_prefix=key_prefix,
            name=api_key.name,
            scopes=api_key.scopes_list,
            created_at=api_key.created_at,
        )

    async def revoke_key(
        self,
        user: User,
        key_id: str,
        request_id: str,
    ) -> None:
        """Mark a key as inactive. Does not delete — preserves audit trail."""
        result = await self._db.execute(
            select(APIKey).where(APIKey.id == key_id, APIKey.user_id == user.id)
        )
        api_key = result.scalar_one_or_none()
        if not api_key:
            raise HTTPException(status_code=404, detail="API key not found.")
        if not api_key.is_active:
            raise HTTPException(status_code=409, detail="API key already revoked.")

        api_key.is_active = False
        api_key.revoked_at = datetime.now(timezone.utc)

        self._db.add(AuditEvent(
            request_id=request_id,
            user_id=user.id,
            api_key_id=key_id,
            action="api_key_revoked",
        ))
        logger.info("api_key_revoked", key_id=key_id, user_id=user.id)

    async def list_keys(self, user: User) -> list[APIKeyInfo]:
        result = await self._db.execute(
            select(APIKey).where(APIKey.user_id == user.id).order_by(APIKey.created_at.desc())
        )
        keys = result.scalars().all()
        return [
            APIKeyInfo(
                id=k.id,
                key_prefix=k.key_prefix,
                name=k.name,
                scopes=k.scopes_list,
                is_active=k.is_active,
                created_at=k.created_at,
                last_used_at=k.last_used_at,
            )
            for k in keys
        ]

    async def validate_raw_key(self, raw_key: str) -> Optional[tuple[APIKey, User]]:
        """Validate a raw Bearer token. Returns (APIKey, User) or None.

        Called on every authenticated request. The DB lookup uses key_prefix
        to narrow the search before doing the hash comparison.
        """
        if not raw_key.startswith(self._settings.api_key_prefix):
            return None

        key_prefix = raw_key[:12]
        result = await self._db.execute(
            select(APIKey).where(
                APIKey.key_prefix == key_prefix,
                APIKey.is_active == True,  # noqa: E712
            )
        )
        candidates = result.scalars().all()

        for candidate in candidates:
            if verify_api_key(raw_key, candidate.key_hash, self._settings):
                # Update last_used_at asynchronously — not in the critical path
                candidate.last_used_at = datetime.now(timezone.utc)

                user_result = await self._db.execute(
                    select(User).where(User.id == candidate.user_id)
                )
                user = user_result.scalar_one_or_none()
                if user and user.is_active:
                    return candidate, user

        return None

    async def record_failed_auth(self, request_id: str, ip_bucket: str) -> None:
        """Write an audit record for failed authentication attempts."""
        self._db.add(AuditEvent(
            request_id=request_id,
            action="auth_failed",
            ip_bucket=ip_bucket,
        ))
        logger.warning("auth_failed", request_id=request_id, ip_bucket=ip_bucket)
