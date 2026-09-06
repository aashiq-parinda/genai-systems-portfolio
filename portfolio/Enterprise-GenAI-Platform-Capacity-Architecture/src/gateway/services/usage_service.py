"""Usage accounting service.

Writes a UsageEvent to PostgreSQL after every inference request
(success or failure). This is the durable record of what happened.

Write strategy:
  - Use asyncio background task (fire-and-forget) so the DB write doesn't
    block the streaming response returning to the client.
  - If the DB write fails, log the error — we prefer losing a usage record
    over blocking the response.
  - For billing-critical deployments, use a queue (Kafka / SQS) instead.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from src.gateway.core.logging import get_logger
from src.gateway.models.usage_event import UsageEvent

logger = get_logger("usage_service")


class UsageService:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def record(
        self,
        *,
        request_id: str,
        user_id: Optional[str],
        api_key_id: Optional[str],
        model: str,
        input_tokens: int,
        output_tokens: int,
        status_code: int,
        status_label: str = "success",
        latency_ms: Optional[float] = None,
        ttft_ms: Optional[float] = None,
        streaming: bool = False,
    ) -> None:
        """Write a usage event to PostgreSQL.

        Called after every inference attempt. total_tokens = input + output.
        """
        event = UsageEvent(
            request_id=request_id,
            user_id=user_id,
            api_key_id=api_key_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            status_code=status_code,
            status_label=status_label,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            streaming=streaming,
            created_at=datetime.now(timezone.utc),
        )
        try:
            self._db.add(event)
            await self._db.flush()
            logger.info(
                "usage_event_recorded",
                request_id=request_id,
                user_id=user_id,
                model=model,
                total_tokens=event.total_tokens,
                status_code=status_code,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
            )
        except Exception as exc:
            # Log but don't raise — usage write failure must not break the response.
            logger.error(
                "usage_event_write_failed",
                request_id=request_id,
                error=str(exc),
            )
