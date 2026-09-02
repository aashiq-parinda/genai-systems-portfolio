"""
Unit tests for Quantum Noise Engine & Decoherence Models
"""
import pytest
import numpy as np
from src.core.noise_engine import QuantumNoiseEngine, NoiseParameters
from src.core.simulator import QuantumCircuit


def test_depolarizing_channel():
    qc = QuantumCircuit(1)
    rho = qc.get_density_matrix()
    engine = QuantumNoiseEngine(NoiseParameters(depolarizing_prob=0.1))
    noisy_rho = engine.apply_depolarizing_channel(rho)
    assert np.isclose(np.trace(noisy_rho), 1.0)
    assert not np.array_equal(rho, noisy_rho)


def test_bit_flip_channel():
    qc = QuantumCircuit(1)
    rho = qc.get_density_matrix()
    engine = QuantumNoiseEngine(NoiseParameters(bit_flip_prob=0.2))
    noisy_rho = engine.apply_bit_flip_channel(rho)
    assert np.isclose(np.trace(noisy_rho), 1.0)
    assert noisy_rho[1, 1] > 0.0  # Population transferred from |0> to |1>


def test_readout_noise():
    engine = QuantumNoiseEngine(NoiseParameters(readout_error_0_to_1=0.05, readout_error_1_to_0=0.05))
    ideal_counts = {"00": 1000}
    noisy_counts = engine.apply_readout_noise(ideal_counts)
    assert sum(noisy_counts.values()) == 1000
    assert "00" in noisy_counts
