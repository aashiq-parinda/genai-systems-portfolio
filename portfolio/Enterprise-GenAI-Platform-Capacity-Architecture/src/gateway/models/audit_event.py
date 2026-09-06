"""AuditEvent ORM model — immutable security audit log."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.gateway.infrastructure.database import Base


class AuditEvent(Base):
    """Immutable security audit record.

    Written for auth events (key creation, revocation, failed auth),
    policy decisions (rate limit, quota block), and risk escalations.

    IP addresses are stored as HASHED buckets — not raw IPs — to preserve
    privacy while still enabling abuse pattern detection.
    """
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    request_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)

    user_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    api_key_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    # e.g. "api_key_created", "api_key_revoked", "auth_failed",
    #       "rate_limit_exceeded", "quota_exceeded", "risk_escalated"
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Coarse IP bucket for rate-limit correlation (not the raw IP)
    ip_bucket: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # "low" | "medium" | "high"
    risk_level: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    def __repr__(self) -> str:
        return f"<AuditEvent action={self.action} user_id={self.user_id} risk={self.risk_level}>"
