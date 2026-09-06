"""User ORM model."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.gateway.infrastructure.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    # Verification flags — coarse identity signals used by RiskScorer.
    # We never store phone numbers; only the boolean verified flag.
    verified_email: Mapped[bool] = mapped_column(Boolean, default=False)
    verified_phone: Mapped[bool] = mapped_column(Boolean, default=False)

    plan: Mapped[str] = mapped_column(String(32), default="free")  # free | pro | enterprise
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    api_keys: Mapped[list["APIKey"]] = relationship(  # noqa: F821
        "APIKey", back_populates="user", lazy="select"
    )
    usage_events: Mapped[list["UsageEvent"]] = relationship(  # noqa: F821
        "UsageEvent", back_populates="user", lazy="select"
    )

    @property
    def account_age_days(self) -> int:
        """Coarse identity signal: how long this account has existed."""
        delta = datetime.now(timezone.utc) - self.created_at
        return delta.days

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} plan={self.plan}>"
