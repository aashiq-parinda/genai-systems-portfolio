"""Auth schemas — API key creation and response."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CreateKeyRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=128, description="Human-readable label")
    scopes: list[str] = Field(
        default=["chat"],
        description="Permissions: chat | embeddings | admin",
    )


class CreateKeyResponse(BaseModel):
    """Returned exactly once at key creation. The raw_key is never stored."""
    id: str
    raw_key: str = Field(description="Full API key — shown once. Store it securely.")
    key_prefix: str = Field(description="Safe-to-display prefix for identification")
    name: Optional[str]
    scopes: list[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class APIKeyInfo(BaseModel):
    """Safe summary — no raw key, no hash."""
    id: str
    key_prefix: str
    name: Optional[str]
    scopes: list[str]
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime]

    model_config = {"from_attributes": True}
