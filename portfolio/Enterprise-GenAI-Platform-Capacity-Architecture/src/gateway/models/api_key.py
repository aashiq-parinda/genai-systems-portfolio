"""APIKey ORM model.

Security design:
- key_hash: PBKDF2-SHA256 hash of the raw key — the only value stored.
- key_prefix: first 12 chars of the raw key (e.g. "llm_a3f2xk9p").
  Safe to display in admin UIs so users can identify their keys.
- The raw key is NEVER stored. It is shown exactly once at creation time.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.gateway.infrastructure.database import Base


class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # The safe-to-display prefix. Never the full key.
    key_prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    # PBKDF2-SHA256 hash — the only key material stored in the DB.
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)

    name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    # Comma-separated scopes: "chat", "embeddings", "admin"
    scopes: Mapped[str] = mapped_column(String(256), default="chat")

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="api_keys")  # noqa: F821
    usage_events: Mapped[list["UsageEvent"]] = relationship(  # noqa: F821
        "UsageEvent", back_populates="api_key", lazy="select"
    )

    @property
    def scopes_list(self) -> list[str]:
        return [s.strip() for s in self.scopes.split(",") if s.strip()]

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes_list or "admin" in self.scopes_list

    def __repr__(self) -> str:
        return f"<APIKey id={self.id} prefix={self.key_prefix} active={self.is_active}>"
