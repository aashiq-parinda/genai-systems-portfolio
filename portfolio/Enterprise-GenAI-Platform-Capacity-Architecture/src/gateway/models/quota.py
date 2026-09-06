"""Quota ORM model — per-user plan limits."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.gateway.infrastructure.database import Base


class Quota(Base):
    """Per-user quota configuration.

    Token-based limits (not just RPM) are essential for LLM APIs because:
    - A single request can consume 4,096+ tokens.
    - RPM alone doesn't bound GPU resource consumption.
    - Downstream billing is token-based, not request-based.
    - A slow user with large context windows can exhaust resources faster
      than a fast user with short prompts.
    """
    __tablename__ = "quotas"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True
    )

    plan: Mapped[str] = mapped_column(String(32), default="free")

    # Request-rate limits
    rpm: Mapped[int] = mapped_column(Integer, default=10)       # requests/minute
    tpm: Mapped[int] = mapped_column(Integer, default=50_000)   # tokens/minute

    # Token volume limits
    daily_tokens: Mapped[int] = mapped_column(Integer, default=50_000)
    monthly_tokens: Mapped[int] = mapped_column(Integer, default=1_000_000)

    # Concurrency limit — critical for GPU capacity management.
    # Unlimited concurrent requests → GPU OOM or starvation for other users.
    max_concurrent: Mapped[int] = mapped_column(Integer, default=2)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:
        return f"<Quota user_id={self.user_id} plan={self.plan} rpm={self.rpm}>"
