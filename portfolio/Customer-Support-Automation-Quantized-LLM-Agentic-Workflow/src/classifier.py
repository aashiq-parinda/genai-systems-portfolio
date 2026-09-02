"""
Compatibility wrapper for classifier module
"""
from src.core.classifier import (
    TicketClassifier,
    RiskLevel,
    TicketCategory,
    PROMPT_INJECTION_PATTERNS,
    CRITICAL_RISK_KEYWORDS,
    HIGH_RISK_KEYWORDS
)

__all__ = [
    "TicketClassifier",
    "RiskLevel",
    "TicketCategory",
    "PROMPT_INJECTION_PATTERNS",
    "CRITICAL_RISK_KEYWORDS",
    "HIGH_RISK_KEYWORDS"
]
