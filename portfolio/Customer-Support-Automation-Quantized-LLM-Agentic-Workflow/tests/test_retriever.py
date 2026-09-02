"""
Unit tests for HybridPolicyRetriever (BM25 + Dense RRF Fusion)
"""
import pytest
from src.core.retriever import HybridPolicyRetriever, Document


@pytest.fixture
def retriever():
    return HybridPolicyRetriever()


def test_retriever_shipping_query(retriever):
    docs = retriever.retrieve("When will my order ship and deliver?", category_filter="Order Status", top_k=1)
    assert len(docs) == 1
    assert "Shipping" in docs[0]["title"]
    assert docs[0]["category"] == "Order Status"
    assert docs[0]["hybrid_score"] > 0


def test_retriever_return_policy(retriever):
    docs = retriever.retrieve("Can I return opened items for a refund within 30 days?", category_filter="Refund Request", top_k=1)
    assert len(docs) == 1
    assert "Return Policy" in docs[0]["title"]
    assert docs[0]["bm25_score"] >= 0


def test_retriever_category_fallback(retriever):
    # Empty query should return empty list gracefully
    docs = retriever.retrieve("")
    assert docs == []


def test_retriever_custom_documents():
    custom_docs = [
        Document(id="test_1", category="Custom", title="Custom Policy", content="Custom content for testing.")
    ]
    custom_retriever = HybridPolicyRetriever(documents=custom_docs)
    res = custom_retriever.retrieve("custom query", top_k=1)
    assert len(res) == 1
    assert res[0]["id"] == "test_1"
