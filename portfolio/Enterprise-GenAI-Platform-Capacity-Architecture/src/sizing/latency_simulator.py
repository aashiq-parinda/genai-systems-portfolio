"""Latency & Throughput SLA Simulator for Distributed LLM Inference.

Simulates TTFT (Time-To-First-Token) and ITL (Inter-Token Latency) across
concurrency loads under continuous batching, prefill vs decode phase splits,
and network gateway overheads.
"""

from dataclasses import dataclass
from typing import Dict, List
import math
import random


@dataclass
class LatencyProfile:
    """Latency distribution percentiles in milliseconds."""
    concurrency_load: int
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    itl_p50_ms: float
    itl_p95_ms: float
    itl_p99_ms: float
    e2e_p50_s: float
    e2e_p95_s: float
    e2e_p99_s: float
    sla_violation_rate: float


class LatencySimulator:
    """Analytical SLA latency simulator."""

    def __init__(
        self,
        base_prefill_ms_per_1k_tokens: float = 85.0,
        base_decode_ms_per_token: float = 6.5,
        gateway_network_overhead_ms: float = 35.0,
        guardrail_overhead_ms: float = 45.0,
    ):
        self.base_prefill_rate = base_prefill_ms_per_1k_tokens
        self.base_decode_rate = base_decode_ms_per_token
        self.gateway_overhead = gateway_network_overhead_ms
        self.guardrail_overhead = guardrail_overhead_ms

    def simulate_profile(
        self,
        concurrency: int,
        active_gpu_replicas: int,
        prompt_tokens: int = 2048,
        generation_tokens: int = 512,
        seed: int = 42
    ) -> LatencyProfile:
        """Simulate latency metrics for given concurrency and replica count."""
        random.seed(seed)
        
        # Load factor per replica
        concurrency_per_replica = concurrency / max(1, active_gpu_replicas)
        # Optimal capacity threshold before queuing delay kicks in
        optimal_threshold = 250.0
        load_ratio = concurrency_per_replica / optimal_threshold

        # Base Prefill Time (TTFT base)
        raw_prefill = (prompt_tokens / 1000.0) * self.base_prefill_rate
        # Queuing delay scales non-linearly with load ratio (M/M/c queuing model approximation)
        queuing_delay = 0.0
        if load_ratio > 1.0:
            queuing_delay = 180.0 * (load_ratio ** 1.8)
        else:
            queuing_delay = 25.0 * load_ratio

        base_ttft = raw_prefill + self.gateway_overhead + self.guardrail_overhead + queuing_delay
        
        ttft_p50 = base_ttft * 1.0
        ttft_p95 = base_ttft * 1.42
        ttft_p99 = base_ttft * 1.85

        # Decode Time (ITL base)
        # Continuous batching introduces small scheduling jitter
        itl_base = self.base_decode_rate * (1.0 + min(0.65, 0.15 * math.log2(max(1.0, load_ratio))))
        itl_p50 = itl_base * 1.0
        itl_p95 = itl_base * 1.28
        itl_p99 = itl_base * 1.60

        # End-to-End Latency in seconds
        e2e_p50 = (ttft_p50 + (generation_tokens * itl_p50)) / 1000.0
        e2e_p95 = (ttft_p95 + (generation_tokens * itl_p95)) / 1000.0
        e2e_p99 = (ttft_p99 + (generation_tokens * itl_p99)) / 1000.0

        # Target SLA: P95 TTFT <= 1000ms (1.0s), P95 E2E <= 5.0s
        sla_violated = 0.0
        if ttft_p95 > 1000.0 or e2e_p95 > 5.0:
            sla_violated = min(1.0, round((ttft_p95 - 1000.0) / 1000.0 + (e2e_p95 - 5.0) / 5.0, 4))

        return LatencyProfile(
            concurrency_load=concurrency,
            ttft_p50_ms=round(ttft_p50, 1),
            ttft_p95_ms=round(ttft_p95, 1),
            ttft_p99_ms=round(ttft_p99, 1),
            itl_p50_ms=round(itl_p50, 1),
            itl_p95_ms=round(itl_p95, 1),
            itl_p99_ms=round(itl_p99, 1),
            e2e_p50_s=round(e2e_p50, 2),
            e2e_p95_s=round(e2e_p95, 2),
            e2e_p99_s=round(e2e_p99, 2),
            sla_violation_rate=sla_violated,
        )
