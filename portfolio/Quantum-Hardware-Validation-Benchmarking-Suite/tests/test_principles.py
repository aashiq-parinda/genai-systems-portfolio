"""
Unit tests for Quantum Principles Validator
"""
import pytest
from src.core.principles_validator import QuantumPrinciplesValidator
from src.core.simulator import QuantumCircuit


def test_normalization():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.h(1)
    qc.cnot(1, 2)
    res = QuantumPrinciplesValidator.validate_normalization(qc)
    assert res["passed"] is True
    assert abs(res["measured_norm"] - 1.0) < 1e-6


def test_born_rule():
    res = QuantumPrinciplesValidator.validate_born_rule(shots=3000)
    assert res["passed"] is True
    assert res["statistical_delta"] < 0.05


def test_interference():
    res = QuantumPrinciplesValidator.validate_interference()
    assert res["passed"] is True
    assert res["final_state_probabilities"]["1"] == 1.0


def test_bell_state_entanglement():
    res = QuantumPrinciplesValidator.validate_entanglement_bell_state()
    assert res["passed"] is True
    assert res["state_probabilities"]["01"] == 0.0
    assert res["state_probabilities"]["10"] == 0.0


def test_unitary_evolution():
    res = QuantumPrinciplesValidator.validate_unitary_evolution()
    assert res["passed"] is True
    assert res["frobenius_error"] < 1e-6


def test_no_cloning_theorem():
    res = QuantumPrinciplesValidator.validate_no_cloning_theorem()
    assert res["passed"] is True
    assert res["theorem_upheld"] is True
