"""
Unit tests for Quantum Statevector Circuit Simulator
"""
import math
import pytest
import numpy as np
from src.core.simulator import QuantumCircuit


def test_ground_state_initialization():
    qc = QuantumCircuit(2)
    assert qc.num_qubits == 2
    assert qc.dim == 4
    assert qc.statevector[0] == 1.0
    assert np.all(qc.statevector[1:] == 0.0)


def test_hadamard_superposition():
    qc = QuantumCircuit(1)
    qc.h(0)
    probs = qc.get_probabilities()
    assert abs(probs[0] - 0.5) < 1e-6
    assert abs(probs[1] - 0.5) < 1e-6


def test_cnot_bell_state():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cnot(0, 1)
    probs = qc.get_probabilities()
    assert abs(probs[0] - 0.5) < 1e-6
    assert abs(probs[3] - 0.5) < 1e-6
    assert probs[1] < 1e-6
    assert probs[2] < 1e-6


def test_rotation_gates():
    qc = QuantumCircuit(1)
    qc.rx(math.pi, 0)
    probs = qc.get_probabilities()
    assert abs(probs[1] - 1.0) < 1e-6  # Rx(pi)|0> = -i|1>


def test_measurement_sampling():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cnot(0, 1)
    counts = qc.sample_measurements(shots=1000, seed=42)
    assert "00" in counts
    assert "11" in counts
    assert "01" not in counts
    assert "10" not in counts
    assert counts["00"] + counts["11"] == 1000
