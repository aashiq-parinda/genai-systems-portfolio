"""
Legal Contract Risk Taxonomy & Clause Policy Engine
"""
from enum import Enum
from typing import Dict, Any, List, Optional


class RiskTier(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# Standard LEDGAR Clause Categories mapped to baseline Risk Tiers
LEDGAR_CATEGORY_RISK_MAPPING: Dict[str, RiskTier] = {
    "Indemnification": RiskTier.CRITICAL,
    "Limitation of Liability": RiskTier.CRITICAL,
    "Termination": RiskTier.HIGH,
    "Governing Law": RiskTier.HIGH,
    "Non-Compete": RiskTier.HIGH,
    "Intellectual Property": RiskTier.HIGH,
    "Confidentiality": RiskTier.MEDIUM,
    "Warranties": RiskTier.MEDIUM,
    "Dispute Resolution": RiskTier.MEDIUM,
    "Assignment": RiskTier.MEDIUM,
    "Payment Terms": RiskTier.MEDIUM,
    "Notices": RiskTier.LOW,
    "Severability": RiskTier.LOW,
    "Entire Agreement": RiskTier.LOW,
    "Counterparts": RiskTier.LOW,
    "Miscellaneous": RiskTier.LOW,
}

# High Risk Term Triggers that elevate clause severity
HIGH_RISK_TRIGGERS = [
    {
        "pattern": "unlimited liability",
        "category": "Limitation of Liability",
        "elevated_risk": RiskTier.CRITICAL,
        "reason": "Unlimited liability clause detected — exposes business to uncapped damages.",
    },
    {
        "pattern": "sole discretion",
        "category": "Termination",
        "elevated_risk": RiskTier.HIGH,
        "reason": "Unilateral termination at sole discretion without reciprocal cure rights.",
    },
    {
        "pattern": "automatic renewal",
        "category": "Termination",
        "elevated_risk": RiskTier.HIGH,
        "reason": "Evergreen auto-renewal trap without mandatory prior notice period.",
    },
    {
        "pattern": "hold harmless and defend",
        "category": "Indemnification",
        "elevated_risk": RiskTier.CRITICAL,
        "reason": "Broad unilateral indemnity obligation without gross negligence carve-outs.",
    },
    {
        "pattern": "exclusive jurisdiction",
        "category": "Governing Law",
        "elevated_risk": RiskTier.HIGH,
        "reason": "Foreign or unfavorable exclusive court venue required for legal dispute.",
    },
    {
        "pattern": "perpetual, irrevocable",
        "category": "Intellectual Property",
        "elevated_risk": RiskTier.HIGH,
        "reason": "Perpetual broad IP license grant over customer proprietary data/materials.",
    },
]


def assess_clause_risk(category: str, clause_text: str) -> Dict[str, Any]:
    """
    Evaluates clause risk tier and inspects text for specific high-risk triggers.
    """
    text_lower = clause_text.lower()
    base_tier = LEDGAR_CATEGORY_RISK_MAPPING.get(category, RiskTier.LOW)

    matched_triggers = []
    final_tier = base_tier

    for trigger in HIGH_RISK_TRIGGERS:
        if trigger["pattern"] in text_lower:
            matched_triggers.append(trigger["reason"])
            if trigger["elevated_risk"] == RiskTier.CRITICAL or final_tier != RiskTier.CRITICAL:
                final_tier = trigger["elevated_risk"]

    return {
        "risk_tier": final_tier.value,
        "is_high_risk": final_tier in [RiskTier.CRITICAL, RiskTier.HIGH],
        "triggers": matched_triggers,
    }
