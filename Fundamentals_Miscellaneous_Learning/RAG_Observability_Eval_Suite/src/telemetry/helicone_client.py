import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from ..config import settings

@dataclass
class HeliconeProxyRequest:
    request_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    status_code: int
    scenario_tag: str
    user_id: str
    session_id: str
    headers: Dict[str, str] = field(default_factory=dict)
    cached: bool = False
    timestamp: float = field(default_factory=time.time)

class HeliconeProxyClient:
    """
    Simulates / integrates with Helicone's edge proxy layer.
    Tracks fast token counts, estimated dollar costs, TTFT latency, and custom property tags.
    """
    # Pricing per 1,000,000 tokens (GPT-4o-mini standard)
    COST_PER_M_INPUT = 0.150
    COST_PER_M_OUTPUT = 0.600

    def __init__(self):
        self.api_key = settings.helicone_api_key
        self.base_url = settings.helicone_base_url
        self.request_logs: List[HeliconeProxyRequest] = []

    def build_headers(
        self,
        scenario: str = "baseline",
        user_id: str = "support_customer",
        session_id: str = "sess_001",
        cache_enabled: bool = True
    ) -> Dict[str, str]:
        """Generate official Helicone gateway headers."""
        return {
            "Helicone-Auth": f"Bearer {self.api_key}",
            "Helicone-Property-Environment": "production",
            "Helicone-Property-Scenario": scenario,
            "Helicone-Property-Domain": "customer-support-rag",
            "Helicone-User-Id": user_id,
            "Helicone-Session-Id": session_id,
            "Helicone-Cache-Enabled": str(cache_enabled).lower()
        }

    def record_proxy_call(
        self,
        request_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        status_code: int = 200,
        scenario_tag: str = "baseline",
        user_id: str = "customer_user",
        session_id: str = "sess_001",
        cached: bool = False
    ) -> HeliconeProxyRequest:
        """Calculate costs and log edge proxy metrics."""
        # Calculate cost in USD
        input_cost = (prompt_tokens / 1_000_000.0) * self.COST_PER_M_INPUT
        output_cost = (completion_tokens / 1_000_000.0) * self.COST_PER_M_OUTPUT
        total_cost = round(input_cost + output_cost, 6)

        headers = self.build_headers(scenario=scenario_tag, user_id=user_id, session_id=session_id)

        req = HeliconeProxyRequest(
            request_id=request_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=total_cost,
            latency_ms=round(latency_ms, 2),
            status_code=status_code,
            scenario_tag=scenario_tag,
            user_id=user_id,
            session_id=session_id,
            headers=headers,
            cached=cached
        )
        self.request_logs.append(req)
        return req

    def get_aggregated_stats(self) -> Dict[str, Any]:
        """Return cumulative metrics surfaced by Helicone dashboard."""
        if not self.request_logs:
            return {"total_requests": 0, "total_cost_usd": 0.0, "avg_latency_ms": 0.0}

        total_cost = sum(r.cost_usd for r in self.request_logs)
        total_tokens = sum(r.total_tokens for r in self.request_logs)
        avg_latency = sum(r.latency_ms for r in self.request_logs) / len(self.request_logs)
        
        return {
            "total_requests": len(self.request_logs),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_ms": round(avg_latency, 2),
            "error_rate_percentage": round(
                (sum(1 for r in self.request_logs if r.status_code >= 400) / len(self.request_logs)) * 100, 2
            )
        }

helicone_client = HeliconeProxyClient()
