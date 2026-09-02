"""
Unit tests for Quantum Error Mitigation (ZNE & Readout Inversion)
"""
import pytest
from src.core.error_mitigation import QuantumErrorMitigator


def test_zero_noise_extrapolation_linear():
    scales = [1.0, 2.0, 3.0]
    # Linear decay: E(c) = 1.0 - 0.05 * c  --> at c=0, E(0) = 1.0
    expectations = [0.95, 0.90, 0.85]
    res = QuantumErrorMitigator.zero_noise_extrapolation(scales, expectations, order=1)
    assert abs(res["zne_mitigated_expectation"] - 1.0) < 1e-4
    assert res["zne_mitigated_expectation"] > res["raw_unmitigated_expectation"]


def test_readout_error_mitigation():
    raw_counts = {"0": 900, "1": 100}
    res = QuantumErrorMitigator.readout_error_mitigation(raw_counts)
    assert "mitigated_counts" in res
    assert sum(res["mitigated_counts"].values()) == 1000
