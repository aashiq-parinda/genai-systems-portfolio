"""
Unit tests for Precedent Vector Retriever
"""
import pytest
from src.core.precedent_retriever import PrecedentRetriever, PrecedentClause


@pytest.fixture
def retriever():
    return PrecedentRetriever()


def test_retrieve_indemnification_precedent(retriever):
    clause = "Customer agrees to defend and indemnify against third-party lawsuits."
    prec = retriever.retrieve_precedent("Indemnification", clause)
    assert prec is not None
    assert prec["category"] == "Indemnification"
    assert "precedent_text" in prec
    assert "standard_risk_guidance" in prec
    assert prec["similarity_score"] > 0


def test_retrieve_custom_precedent():
    custom = [
        PrecedentClause(
            id="test_prec_1",
            category="Custom Category",
            source="Custom Source",
            precedent_text="Custom precedent language for testing.",
            standard_risk_guidance="Standard guidance test."
        )
    ]
    r = PrecedentRetriever(precedents=custom)
    res = r.retrieve_precedent("Custom Category", "testing language")
    assert res["precedent_id"] == "test_prec_1"
