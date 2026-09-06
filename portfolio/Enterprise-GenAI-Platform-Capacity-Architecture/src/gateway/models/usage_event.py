"""UsageEvent ORM model — durable per-request accounting record.

Every inference request (success or failure) produces one UsageEvent.
Redis holds fast rolling counters; PostgreSQL holds the permanent record.

Why both?
- Redis counters answer "how many tokens has this user consumed in the last
  minute?" in <1ms — essential for rate limiting on the hot path.
- PostgreSQL answers "how many tokens did this user consume in March?" —
  essential for billing, quota resets, and abuse investigations.
- If Redis restarts, counters reset but the durable history is safe.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.gateway.infrastructure.database import Base


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    request_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    api_key_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True
    )

    model: Mapped[str] = mapped_column(String(128), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)

    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # TTFT = Time To First Token — the most LLM-specific latency metric.
    # For streaming, this is time from request start to first token received.
    # For non-streaming, TTFT ≈ total latency.
    ttft_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    # "success" | "rate_limited" | "quota_exceeded" | "inference_error" | "timeout"
    status_label: Mapped[str] = mapped_column(String(32), default="success")

    streaming: Mapped[bool] = mapped_column(default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="usage_events")  # noqa: F821
    api_key: Mapped[Optional["APIKey"]] = relationship("APIKey", back_populates="usage_events")  # noqa: F821

    def __repr__(self) -> str:
        return (
            f"<UsageEvent request_id={self.request_id} model={self.model} "
            f"tokens={self.total_tokens} status={self.status_code}>"
        )
