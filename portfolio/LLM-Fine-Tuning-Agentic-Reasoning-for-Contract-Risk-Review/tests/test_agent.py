"""
Unit tests for ContractRiskReviewAgent & Review DAG
"""
import pytest
from src.core.agent import ContractRiskReviewAgent, ReviewStatus
from src.core.risk_taxonomy import RiskTier


@pytest.fixture
def agent():
    return ContractRiskReviewAgent()


def test_agent_review_full_contract(agent):
    sample_contract = """
Section 1. Indemnification. Vendor agrees to indemnify and hold harmless Customer from all losses and liabilities.

Section 2. Limitation of Liability. Neither party shall be liable for consequential damages, and total liability is capped at 12 months fees paid.

Section 3. Termination. Either party may terminate with 30 days notice for material breach.

Section 4. Severability. If any term is invalid, the remainder of this Agreement survives.
"""
    report = agent.review_contract(sample_contract)
    assert report.total_clauses == 4
    assert report.overall_risk_tier in [RiskTier.CRITICAL.value, RiskTier.HIGH.value]
    assert report.high_risk_clause_count >= 2
    assert report.auto_finalized_count >= 1
    assert len(report.clause_analyses) == 4
    assert "overall" in report.executive_summary.lower()


def test_agent_critical_trigger_escalation(agent):
    critical_clause = """
1.1 Liability Cap. Customer agrees to unlimited liability for any system defects or breaches under this agreement.
"""
    report = agent.review_contract(critical_clause)
    assert report.overall_risk_tier == RiskTier.CRITICAL.value
    assert report.clause_analyses[0].status == ReviewStatus.CRITICAL_RISK_ESCALATED
    assert "unlimited liability" in report.clause_analyses[0].risk_triggers[0].lower()
