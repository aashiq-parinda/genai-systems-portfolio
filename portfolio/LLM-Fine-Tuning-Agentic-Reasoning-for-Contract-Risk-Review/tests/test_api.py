"""
Unit & Integration tests for FastAPI Contract Review Endpoints
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "contract-risk-review-agent"


def test_analyze_contract_endpoint():
    contract_payload = {
        "contract_text": "Section 1. Governing Law. This agreement is governed by the laws of Delaware.\n\nSection 2. Notices. All notices must be delivered in writing.",
        "confidence_threshold": 0.75
    }
    res = client.post("/v1/contracts/analyze", json=contract_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_clauses"] == 2
    assert "overall_risk_tier" in data
    assert len(data["clause_analyses"]) == 2


def test_classify_clause_endpoint():
    clause_payload = {
        "clause_text": "Vendor agrees to defend and indemnify Customer against any third-party claims."
    }
    res = client.post("/v1/contracts/classify-clause", json=clause_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_category"] == "Indemnification"
    assert data["risk_tier"] == "CRITICAL"
    assert data["precedent"] is not None


def test_fairness_report_endpoint():
    res = client.get("/v1/contracts/fairness-report")
    assert res.status_code == 200
    data = res.json()
    assert "macro_f1" in data
    assert "per_category_metrics" in data
