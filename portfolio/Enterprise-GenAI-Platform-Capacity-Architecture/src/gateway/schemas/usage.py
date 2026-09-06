"""Usage and quota schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class UsageEventOut(BaseModel):
    id: str
    request_id: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: Optional[float]
    ttft_ms: Optional[float]
    status_code: int
    status_label: str
    streaming: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class UsageSummary(BaseModel):
    user_id: str
    period: str  # "today" | "this_month" | "all_time"
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    avg_latency_ms: Optional[float]
    avg_ttft_ms: Optional[float]


class QuotaStatus(BaseModel):
    plan: str
    rpm_limit: int
    rpm_used: int
    tpm_limit: int
    daily_tokens_limit: int
    daily_tokens_used: int
    monthly_tokens_limit: int
    monthly_tokens_used: int
    max_concurrent: int
    current_concurrent: int
