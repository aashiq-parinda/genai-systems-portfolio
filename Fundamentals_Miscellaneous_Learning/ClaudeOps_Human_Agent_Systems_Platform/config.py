"""
ClaudeOps: Central Configuration & Anthropics Operational Constants.

Covers:
- Model Family Sizing (Haiku vs Sonnet vs Opus)
- Prompt Caching FinOps parameters (90% read discount)
- Human-in-the-Loop Risk Categories & Thresholds
- AI Fluency confidence escalation limits
"""

from enum import Enum
from dataclasses import dataclass


class ClaudeModelTier(str, Enum):
    """
    Claude Model Family Selection (from Claude Platform 101).
    - HAIKU: High-speed classification, intent triage, fast sub-agent filtering.
    - SONNET: Flagship for agentic coding, tool calling, and deep reasoning.
    - OPUS: High-order strategy and ambiguous executive synthesis.
    """
    HAIKU = "claude-3-5-haiku-20241022"
    SONNET = "claude-3-5-sonnet-20241022"
    OPUS = "claude-3-opus-20240229"


@dataclass(frozen=True)
class ModelPricing:
    """Cost per million tokens ($ USD) for accurate FinOps calculations."""
    input_per_m: float
    output_per_m: float
    cache_write_per_m: float
    cache_read_per_m: float  # ~90% discount relative to standard input


# Anthropic Public Pricing Rates (as taught in Claude Platform 101)
PRICING_REGISTRY = {
    ClaudeModelTier.HAIKU: ModelPricing(
        input_per_m=0.80,
        output_per_m=4.00,
        cache_write_per_m=1.00,
        cache_read_per_m=0.08  # 90% discount
    ),
    ClaudeModelTier.SONNET: ModelPricing(
        input_per_m=3.00,
        output_per_m=15.00,
        cache_write_per_m=3.75,
        cache_read_per_m=0.30  # 90% discount
    ),
    ClaudeModelTier.OPUS: ModelPricing(
        input_per_m=15.00,
        output_per_m=75.00,
        cache_write_per_m=18.75,
        cache_read_per_m=1.50  # 90% discount
    ),
}


class ActionRiskCategory(str, Enum):
    """
    HITL Action Risk Tiers (from Building Effective Human-Agent Teams).
    Determines whether execution is autonomous or requires human gate approval.
    """
    READ_ONLY = "READ_ONLY"             # e.g., inspect file, list dir -> 100% Autonomous
    EPHEMERAL_WRITE = "EPHEMERAL_WRITE" # e.g., scratchpad, test file -> Autonomous with logging
    STATE_MUTATING = "STATE_MUTATING"   # e.g., rewrite source code, update DB -> REQUIRES APPROVAL
    CRITICAL_RISK = "CRITICAL_RISK"     # e.g., delete files, deploy prod -> REQUIRES TWO-STEP CONFIRMATION


class EscalationReason(str, Enum):
    """
    Confidence & Error Escalation Triggers (from AI Fluency).
    """
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AMBIGUOUS_SPECIFICATION = "AMBIGUOUS_SPECIFICATION"
    TOOL_RETRY_EXHAUSTED = "TOOL_RETRY_EXHAUSTED"
    SYCOPHANCY_OR_SAFETY_FLAG = "SYCOPHANCY_OR_SAFETY_FLAG"


# Global Operational Thresholds
CONFIDENCE_THRESHOLD: float = 0.85      # Below 85% confidence triggers HITL escalation
MAX_AGENT_ITERATIONS: int = 8           # Prevents runaway ReAct loops
CACHE_TTL_SECONDS: int = 300            # Anthropic 5-minute ephemeral cache window
DEFAULT_TEMPERATURE_DETERMINISTIC: float = 0.0
DEFAULT_TEMPERATURE_CREATIVE: float = 0.4
