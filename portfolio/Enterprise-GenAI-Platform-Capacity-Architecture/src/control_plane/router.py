"""Dynamic Model Routing and Workload Distribution Engine.

Analyzes incoming enterprise prompt complexity, semantic classification,
and SLA requirements to route between Small Language Models (SLMs)
and Frontier Reasoning Models (Claude-X / 70B+), minimizing inference TCO.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import re


@dataclass
class RouteDecision:
    """Routing outcome and rationale."""
    selected_model: str
    tier: str  # "SLM" or "FRONTIER"
    complexity_score: float  # 0.0 (trivial) to 1.0 (deep reasoning)
    estimated_tokens_prompt: int
    estimated_tokens_completion: int
    estimated_cost_usd: float
    routing_reason: str


class DynamicModelRouter:
    """Intelligent semantic and complexity router for enterprise workloads."""

    COMPLEXITY_TRIGGERS = [
        r"(?i)\b(compare|contrast|synthesize|derive|architect|root\s+cause|step-by-step|investigate)\b",
        r"(?i)\b(analyze|analysis|liability|indemnification|contract\s+(?:risk|clause)|security\s+audit|compliance)\b",
        r"(?i)\b(write\s+(python|c\+\+|rust|sql|algorithm|pipeline|code))\b",
        r"(?i)\b(why\s+did|mathematical\s+proof|optimize\s+performance|debug|diagnose)\b",
    ]

    SIMPLE_TRIGGERS = [
        r"(?i)\b(summarize\s+briefly|extract\s+entities|classify|yes\s+or\s+no)\b",
        r"(?i)\b(format\s+as\s+json|translate\s+to|what\s+is\s+the\s+status|ticket\s+number)\b",
        r"(?i)\b(lookup|faq|operating\s+hours|contact\s+info|leave\s+balance)\b",
    ]

    def __init__(
        self,
        slm_model_name: str = "Enterprise-SLM-8B",
        frontier_model_name: str = "Claude-X-Frontier-70B",
        slm_cost_per_1k: float = 0.0002,
        frontier_cost_per_1k: float = 0.0035,
    ):
        self.slm_model = slm_model_name
        self.frontier_model = frontier_model_name
        self.slm_cost = slm_cost_per_1k
        self.frontier_cost = frontier_cost_per_1k

        self.complex_regexes = [re.compile(p) for p in self.COMPLEXITY_TRIGGERS]
        self.simple_regexes = [re.compile(p) for p in self.SIMPLE_TRIGGERS]

    def estimate_token_count(self, text: str) -> int:
        """Heuristic word-to-token estimator (~1.3 tokens per word)."""
        words = len(text.split())
        return max(1, int(words * 1.3))

    def evaluate_complexity(self, prompt: str, conversation_turn_count: int = 1) -> float:
        """Calculate complexity score between 0.0 and 1.0."""
        score = 0.20  # baseline
        
        # Length signal
        token_count = self.estimate_token_count(prompt)
        if token_count > 600:
            score += 0.35
        elif token_count > 200:
            score += 0.15

        # Complex keyword signals
        complex_matches = sum(1 for r in self.complex_regexes if r.search(prompt))
        score += min(0.40, complex_matches * 0.20)

        # Simple keyword reductions
        simple_matches = sum(1 for r in self.simple_regexes if r.search(prompt))
        score -= min(0.30, simple_matches * 0.15)

        # Multi-turn reasoning depth
        if conversation_turn_count > 4:
            score += 0.15

        return max(0.0, min(1.0, round(score, 2)))

    def route_request(
        self,
        prompt: str,
        force_frontier: bool = False,
        conversation_turns: int = 1
    ) -> RouteDecision:
        """Route request to the most cost-effective model tier meeting quality SLAs."""
        prompt_tokens = self.estimate_token_count(prompt)
        expected_output_tokens = min(1024, max(128, int(prompt_tokens * 0.6)))

        if force_frontier:
            cost = ((prompt_tokens + expected_output_tokens) / 1000.0) * self.frontier_cost
            return RouteDecision(
                selected_model=self.frontier_model,
                tier="FRONTIER",
                complexity_score=1.0,
                estimated_tokens_prompt=prompt_tokens,
                estimated_tokens_completion=expected_output_tokens,
                estimated_cost_usd=round(cost, 6),
                routing_reason="Explicit override requested by enterprise policy.",
            )

        complexity = self.evaluate_complexity(prompt, conversation_turns)

        # Route decision threshold: complexity >= 0.50 goes to Frontier
        if complexity >= 0.50:
            selected_model = self.frontier_model
            tier = "FRONTIER"
            cost = ((prompt_tokens + expected_output_tokens) / 1000.0) * self.frontier_cost
            reason = f"High complexity score ({complexity}) requires multi-step frontier reasoning."
        else:
            selected_model = self.slm_model
            tier = "SLM"
            cost = ((prompt_tokens + expected_output_tokens) / 1000.0) * self.slm_cost
            reason = f"Low-to-moderate complexity ({complexity}) fulfilled by optimized 8B SLM."

        return RouteDecision(
            selected_model=selected_model,
            tier=tier,
            complexity_score=complexity,
            estimated_tokens_prompt=prompt_tokens,
            estimated_tokens_completion=expected_output_tokens,
            estimated_cost_usd=round(cost, 6),
            routing_reason=reason,
        )
