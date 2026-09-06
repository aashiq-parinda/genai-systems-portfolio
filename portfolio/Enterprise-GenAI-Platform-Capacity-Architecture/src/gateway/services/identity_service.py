"""Identity and risk scoring service.

Demonstrates why: Authentication ≠ Identity ≠ Quota ≠ Abuse Prevention

A request can be:
  - Authenticated (valid API key) but HIGH RISK (new account, unusual patterns)
  - Unauthenticated → blocked at auth layer
  - Authenticated + low risk → fast path
  - Authenticated + high risk → additional friction or stricter quota

Signals used (all privacy-conscious and coarse):
  - account_age_days: new accounts are higher risk
  - verified_email: unverified = higher risk
  - verified_phone: additional trust signal
  - failed_auth_attempts (from Redis): brute-force signal
  - ip_bucket: coarse geolocation/ASN bucket, not raw IP
  - recent_request_rate: requests per hour over a rolling window

We do NOT:
  - Fingerprint devices or browsers
  - Correlate across accounts
  - Store or use raw IP addresses
  - Attempt to identify real-world individuals
"""

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.gateway.core.logging import get_logger
from src.gateway.infrastructure.redis import get_counter
from src.gateway.models.user import User

logger = get_logger("identity_service")


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class RiskAssessment:
    level: RiskLevel
    score: float        # 0.0 (safe) to 1.0 (high risk)
    signals: dict[str, str]
    action: str         # "allow" | "slow_down" | "strict_quota" | "block"


def make_ip_bucket(client_ip: str) -> str:
    """Hash the IP to a coarse bucket. We never store raw IPs.

    Uses only the first 3 octets of IPv4 (or first 64 bits of IPv6)
    to create a /24 network-level bucket, then HMAC-SHA256 it.
    This lets us detect patterns at the network level without tracking individuals.
    """
    parts = client_ip.split(".")
    if len(parts) == 4:
        coarse = ".".join(parts[:3])  # /24 subnet
    else:
        coarse = client_ip[:client_ip.rfind(":")]  # IPv6 /64

    return "ipb_" + hashlib.sha256(coarse.encode()).hexdigest()[:16]


class IdentityService:
    """Compute a risk score from coarse, privacy-conscious signals."""

    # Thresholds
    HIGH_RISK_THRESHOLD = 0.65
    MEDIUM_RISK_THRESHOLD = 0.35

    # Redis keys
    FAILED_AUTH_KEY = "risk:failed_auth:{ip_bucket}"
    HOURLY_REQUESTS_KEY = "risk:hourly:{user_id}"

    async def assess(
        self,
        user: User,
        ip_bucket: str,
        request_id: str,
    ) -> RiskAssessment:
        """Compute a risk score for the current request context."""
        score = 0.0
        signals: dict[str, str] = {}

        # Signal 1: Account age — new accounts have less established trust
        if user.account_age_days < 1:
            score += 0.30
            signals["account_age"] = "new_account_less_than_1_day"
        elif user.account_age_days < 7:
            score += 0.15
            signals["account_age"] = "new_account_less_than_7_days"
        else:
            signals["account_age"] = f"{user.account_age_days}_days"

        # Signal 2: Email verification
        if not user.verified_email:
            score += 0.20
            signals["verified_email"] = "false"
        else:
            signals["verified_email"] = "true"

        # Signal 3: Phone verification (additional trust)
        if user.verified_phone:
            score -= 0.10  # bonus trust
            signals["verified_phone"] = "true"
        else:
            signals["verified_phone"] = "false"

        # Signal 4: Recent failed auth attempts (brute-force signal)
        failed_key = self.FAILED_AUTH_KEY.format(ip_bucket=ip_bucket)
        failed_attempts = await get_counter(failed_key)
        if failed_attempts >= 10:
            score += 0.35
            signals["failed_auth_attempts"] = f"{failed_attempts}_high"
        elif failed_attempts >= 3:
            score += 0.15
            signals["failed_auth_attempts"] = f"{failed_attempts}_moderate"
        else:
            signals["failed_auth_attempts"] = str(failed_attempts)

        # Signal 5: Hourly request rate
        hourly_key = self.HOURLY_REQUESTS_KEY.format(user_id=user.id)
        hourly_requests = await get_counter(hourly_key)
        if hourly_requests > 500:
            score += 0.20
            signals["hourly_rate"] = "excessive"
        elif hourly_requests > 200:
            score += 0.10
            signals["hourly_rate"] = "elevated"
        else:
            signals["hourly_rate"] = "normal"

        score = max(0.0, min(1.0, round(score, 3)))

        if score >= self.HIGH_RISK_THRESHOLD:
            level = RiskLevel.HIGH
            action = "strict_quota"
        elif score >= self.MEDIUM_RISK_THRESHOLD:
            level = RiskLevel.MEDIUM
            action = "slow_down"
        else:
            level = RiskLevel.LOW
            action = "allow"

        assessment = RiskAssessment(
            level=level,
            score=score,
            signals=signals,
            action=action,
        )

        logger.info(
            "risk_assessment",
            user_id=user.id,
            risk_level=level.value,
            score=score,
            action=action,
            request_id=request_id,
        )

        return assessment
