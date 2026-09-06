"""API key generation, hashing, and verification.

Design decisions:
- Keys use a recognisable prefix (llm_) so they're easy to grep in logs/audits.
- The raw key is PBKDF2-SHA256 hashed before storage. We never store the
  plaintext — only the hash. The full key is shown exactly once at creation.
- We store only a non-sensitive prefix (e.g. "llm_a3f2...") in the DB so
  admins can identify which key a user is referring to in support tickets.
- PBKDF2 is appropriate here because the key is already high-entropy (32
  random bytes). bcrypt would be overkill and slower with no security benefit
  for uniformly random secrets.
"""

import hashlib
import hmac
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.gateway.core.config import Settings, get_settings

_bearer_scheme = HTTPBearer(auto_error=False)


# ── Key Generation ─────────────────────────────────────────────────────────────

def generate_api_key(settings: Settings) -> tuple[str, str, str]:
    """Generate a new API key.

    Returns:
        (raw_key, key_hash, key_prefix)
        - raw_key: shown to the user exactly once, never stored
        - key_hash: stored in PostgreSQL
        - key_prefix: first 12 chars of raw key — used as a human-readable ID
    """
    token = secrets.token_urlsafe(32)  # 256 bits of entropy
    raw_key = f"{settings.api_key_prefix}{token}"
    key_hash = _hash_key(raw_key, settings)
    key_prefix = raw_key[:12]  # "llm_" + 8 chars — safe to store and display
    return raw_key, key_hash, key_prefix


def _hash_key(raw_key: str, settings: Settings) -> str:
    """PBKDF2-SHA256 hash of the raw key. Safe to store in PostgreSQL."""
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        raw_key.encode("utf-8"),
        settings.secret_key.encode("utf-8"),  # salt from env — never hardcoded
        settings.api_key_hash_iterations,
    )
    return dk.hex()


def verify_api_key(raw_key: str, stored_hash: str, settings: Settings) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    candidate_hash = _hash_key(raw_key, settings)
    return hmac.compare_digest(candidate_hash, stored_hash)


# ── FastAPI Dependency ─────────────────────────────────────────────────────────

async def require_bearer_token(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_bearer_scheme),
    ],
) -> str:
    """Extract raw Bearer token from Authorization header.

    Raises 401 if header is missing or malformed.
    The token is NOT validated here — validation happens in AuthService
    so we can attach user context to request.state centrally.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Expected: Bearer llm_xxx",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
