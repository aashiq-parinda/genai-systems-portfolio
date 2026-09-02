"""
Production Ticket Classifier with Multi-tier Intent Detection, Risk Guardrails & Jailbreak Defense
"""
import re
from enum import Enum
from typing import Dict, Any, List, Optional


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TicketCategory(str, Enum):
    ORDER_STATUS = "Order Status"
    REFUND_REQUEST = "Refund Request"
    TECHNICAL_SUPPORT = "Technical Support"
    ACCOUNT_SECURITY = "Account Security"
    BILLING_DISPUTE = "Billing Dispute"
    GENERAL_INQUIRY = "General Inquiry"
    ESCALATION_NEEDED = "Escalation Needed"


PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"system\s*prompt",
    r"you\s+are\s+now\s+(an?\s+)?unrestricted",
    r"dan\s+mode",
    r"developer\s+mode\s+(enabled|on)",
    r"jailbreak",
    r"disregard\s+(the\s+)?rules",
    r"bypass\s+safety\s+filter",
    r"reveal\s+your\s+(secret|internal|hidden)\s+instructions?",
    r"format\s+as\s+json\s+and\s+print\s+system\s+context",
]

CRITICAL_RISK_KEYWORDS = [
    "subpoena",
    "lawsuit",
    "legal action",
    "attorney",
    "arbitration",
    "chargeback",
    "unauthorized transaction",
    "identity theft",
    "data breach",
    "security vulnerability",
    "gdpr deletion request",
]

HIGH_RISK_KEYWORDS = [
    "fraud",
    "stolen card",
    "hacked account",
    "wire fraud",
    "formal complaint",
    "police report",
    "refund dispute",
]


class TicketClassifier:
    """
    Production-grade Classifier & Risk Policy Engine.
    Enforces deterministic safety guardrails before running downstream generation.
    """

    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold
        self._compiled_injection_regexes = [
            re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS
        ]

    def detect_prompt_injection(self, text: str) -> Optional[str]:
        """Detects adversarial jailbreaks and system prompt extraction attacks."""
        for pattern in self._compiled_injection_regexes:
            match = pattern.search(text)
            if match:
                return f"Adversarial prompt injection trigger detected: '{match.group(0)}'"
        return None

    def assess_risk(self, query: str) -> Dict[str, Any]:
        """Evaluates security, legal, and financial risk level of the incoming ticket."""
        query_lower = query.lower()

        # Check prompt injection
        injection_reason = self.detect_prompt_injection(query)
        if injection_reason:
            return {
                "risk_level": RiskLevel.CRITICAL,
                "is_high_risk": True,
                "requires_escalation": True,
                "reason": injection_reason,
            }

        # Check critical keywords
        for kw in CRITICAL_RISK_KEYWORDS:
            if kw in query_lower:
                return {
                    "risk_level": RiskLevel.CRITICAL,
                    "is_high_risk": True,
                    "requires_escalation": True,
                    "reason": f"Critical legal/compliance trigger detected: '{kw}'",
                }

        # Check high risk keywords
        for kw in HIGH_RISK_KEYWORDS:
            if kw in query_lower:
                return {
                    "risk_level": RiskLevel.HIGH,
                    "is_high_risk": True,
                    "requires_escalation": True,
                    "reason": f"High financial/security risk trigger detected: '{kw}'",
                }

        # Check medium risk (e.g. general refund inquiry with negative sentiment)
        if any(term in query_lower for term in ["angry", "unacceptable", "scam", "rip off"]):
            return {
                "risk_level": RiskLevel.MEDIUM,
                "is_high_risk": False,
                "requires_escalation": False,
                "reason": "Elevated customer sentiment detected - monitoring required.",
            }

        return {
            "risk_level": RiskLevel.LOW,
            "is_high_risk": False,
            "requires_escalation": False,
            "reason": None,
        }

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classifies incoming query intent with risk guardrails and confidence calibration.
        """
        risk_assessment = self.assess_risk(query)
        if risk_assessment["requires_escalation"]:
            return {
                "category": TicketCategory.ESCALATION_NEEDED.value,
                "confidence": 1.0,
                "escalated": True,
                "risk_level": risk_assessment["risk_level"].value,
                "reason": risk_assessment["reason"],
            }

        query_lower = query.lower()

        # Intent scoring
        scores: Dict[str, float] = {
            TicketCategory.ORDER_STATUS.value: 0.0,
            TicketCategory.REFUND_REQUEST.value: 0.0,
            TicketCategory.TECHNICAL_SUPPORT.value: 0.0,
            TicketCategory.ACCOUNT_SECURITY.value: 0.0,
            TicketCategory.BILLING_DISPUTE.value: 0.0,
            TicketCategory.GENERAL_INQUIRY.value: 0.2,
        }

        # Keyword feature matching
        if any(k in query_lower for k in ["order", "track", "tracking", "shipment", "delivery", "delivered", "package", "where is my"]):
            scores[TicketCategory.ORDER_STATUS.value] += 0.92
        if any(k in query_lower for k in ["refund", "return", "money back", "reimburse", "exchange"]):
            scores[TicketCategory.REFUND_REQUEST.value] += 0.89
        if any(k in query_lower for k in ["bug", "error", "crash", "not working", "fail", "technical", "app issue", "500"]):
            scores[TicketCategory.TECHNICAL_SUPPORT.value] += 0.91
        if any(k in query_lower for k in ["password", "2fa", "two-factor", "login issue", "reset password", "locked out"]):
            scores[TicketCategory.ACCOUNT_SECURITY.value] += 0.88
        if any(k in query_lower for k in ["invoice", "receipt", "overcharged", "billing statement", "charged twice"]):
            scores[TicketCategory.BILLING_DISPUTE.value] += 0.86

        top_category = max(scores, key=lambda k: scores[k])
        top_confidence = scores[top_category]

        # If highest confidence falls below threshold, escalate to human agent
        if top_confidence < self.confidence_threshold:
            return {
                "category": TicketCategory.ESCALATION_NEEDED.value,
                "confidence": round(top_confidence, 2),
                "escalated": True,
                "risk_level": RiskLevel.MEDIUM.value,
                "reason": f"Classification confidence ({top_confidence:.2f}) below threshold ({self.confidence_threshold:.2f})",
            }

        return {
            "category": top_category,
            "confidence": round(min(top_confidence, 0.99), 2),
            "escalated": False,
            "risk_level": risk_assessment["risk_level"].value,
            "reason": None,
        }
