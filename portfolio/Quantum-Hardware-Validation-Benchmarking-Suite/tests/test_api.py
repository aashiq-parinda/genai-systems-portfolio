"""
Unit & Integration tests for FastAPI Quantum API
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
    assert data["service"] == "quantum-validation-suite"


def test_simulate_circuit_endpoint():
    payload = {
        "num_qubits": 2,
        "gates": [
            {"gate": "H", "qubit": 0},
            {"gate": "CNOT", "control": 0, "target": 1}
        ],
        "shots": 500
    }
    res = client.post("/v1/quantum/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["num_qubits"] == 2
    assert "00" in data["counts"]
    assert "11" in data["counts"]
    assert data["statevector_norm"] == 1.0


def test_validate_principles_endpoint():
    res = client.post("/v1/quantum/validate-principles")
    assert res.status_code == 200
    data = res.json()
    assert data["all_passed"] is True
    assert data["total_principles_tested"] >= 6


def test_benchmark_noise_endpoint():
    payload = {"circuit_type": "bell", "depolarizing_prob": 0.02}
    res = client.post("/v1/quantum/benchmark-noise", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["ideal_fidelity"] == 1.0
    assert data["noisy_fidelity"] > 0.90


def test_mitigate_error_endpoint():
    payload = {
        "noise_scales": [1.0, 2.0, 3.0],
        "measured_expectations": [0.95, 0.90, 0.85]
    }
    res = client.post("/v1/quantum/mitigate-error", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert abs(data["zne_mitigated_expectation"] - 1.0) < 1e-3


def test_divincenzo_scorecard_endpoint():
    res = client.get("/v1/quantum/divincenzo-scorecard")
    assert res.status_code == 200
    data = res.json()
    assert len(data["criteria"]) == 7
