"""
ClaudeOps: Telemetry, FinOps Accounting & Audit Receipts.

Directly implements:
- Accurate Prompt Caching FinOps calculations (90% read discounts)
- Model cost tracking across Claude tiers (Haiku / Sonnet / Opus)
- Immutable JSON audit receipts for human review and compliance
"""

import time
import uuid
from typing import Dict, Any, List
from config import ClaudeModelTier, PRICING_REGISTRY


class FinOpsTelemetryLogger:
    """Calculates enterprise token costs and maintains audit trails."""

    def __init__(self, run_name: str, model: ClaudeModelTier = ClaudeModelTier.SONNET):
        self.run_id = f"claudeops-{uuid.uuid4().hex[:8]}"
        self.run_name = run_name
        self.model = model
        self.start_time = time.time()
        self.events: List[Dict[str, Any]] = []

    def record_step(self, step_name: str, details: Dict[str, Any]):
        """Records an intermediate execution step."""
        self.events.append({
            "timestamp": time.time() - self.start_time,
            "step": step_name,
            "details": details
        })

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_read_tokens: int = 0
    ) -> Dict[str, float]:
        """
        Calculates exact cost in USD, showing savings from Prompt Caching.
        """
        pricing = PRICING_REGISTRY[self.model]

        cost_uncached_input = (input_tokens / 1_000_000) * pricing.input_per_m
        cost_cached_input = (cached_read_tokens / 1_000_000) * pricing.cache_read_per_m
        cost_output = (output_tokens / 1_000_000) * pricing.output_per_m

        actual_total = cost_uncached_input + cost_cached_input + cost_output

        # Cost if caching hadn't been used (paying full input rate for cached tokens)
        hypothetical_uncached_total = (
            ((input_tokens + cached_read_tokens) / 1_000_000) * pricing.input_per_m
            + cost_output
        )

        savings_usd = max(0.0, hypothetical_uncached_total - actual_total)
        savings_pct = (savings_usd / hypothetical_uncached_total * 100.0) if hypothetical_uncached_total > 0 else 0.0

        return {
            "actual_total_usd": round(actual_total, 6),
            "uncached_hypothetical_usd": round(hypothetical_uncached_total, 6),
            "savings_usd": round(savings_usd, 6),
            "savings_percentage": round(savings_pct, 2)
        }

    def generate_audit_receipt(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_read_tokens: int,
        hitl_interventions: int,
        final_status: str
    ) -> Dict[str, Any]:
        """
        Generates an immutable audit receipt to present to engineering leadership.
        """
        duration = round(time.time() - self.start_time, 3)
        cost_metrics = self.calculate_cost(input_tokens, output_tokens, cached_read_tokens)

        return {
            "receipt_id": self.run_id,
            "run_name": self.run_name,
            "model_tier": self.model.value,
            "duration_seconds": duration,
            "final_status": final_status,
            "tokens": {
                "uncached_input": input_tokens,
                "cached_read": cached_read_tokens,
                "output": output_tokens,
                "total": input_tokens + cached_read_tokens + output_tokens
            },
            "finops": cost_metrics,
            "hitl_interventions_count": hitl_interventions,
            "timeline_events_count": len(self.events)
        }
