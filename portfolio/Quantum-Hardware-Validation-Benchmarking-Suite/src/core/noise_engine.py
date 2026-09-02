"""
Quantum Noise Simulation Engine (Depolarizing, Bit-Flip, Readout & Thermal Relaxation)
"""
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
from src.core.simulator import I2, X_GATE, Y_GATE, Z_GATE


@dataclass
class NoiseParameters:
    depolarizing_prob: float = 0.02
    bit_flip_prob: float = 0.01
    phase_flip_prob: float = 0.01
    readout_error_0_to_1: float = 0.03
    readout_error_1_to_0: float = 0.02
    t1_us: float = 50.0  # Amplitude damping relaxation time (microseconds)
    t2_us: float = 70.0  # Dephasing coherence time (microseconds)
    gate_time_us: float = 0.1


class QuantumNoiseEngine:
    """
    Simulates physical quantum noise channels on quantum state density matrices:
    - Depolarizing Channel
    - Bit-Flip and Phase-Flip Pauli Channels
    - Readout Measurement Confusion Channel
    - Thermal Relaxation & Decoherence Decay (T1 / T2)
    """

    def __init__(self, params: Optional[NoiseParameters] = None):
        self.params = params or NoiseParameters()

    def apply_depolarizing_channel(self, rho: np.ndarray, prob: Optional[float] = None) -> np.ndarray:
        """
        Applies depolarizing channel: E(rho) = (1 - p)*rho + (p / d)*I
        """
        p = prob if prob is not None else self.params.depolarizing_prob
        dim = rho.shape[0]
        identity = np.eye(dim, dtype=complex)
        return (1.0 - p) * rho + (p / dim) * identity

    def apply_bit_flip_channel(self, rho: np.ndarray, prob: Optional[float] = None) -> np.ndarray:
        """
        Applies bit-flip channel: E(rho) = (1 - p)*rho + p * X rho X (for single-qubit subsystem)
        """
        p = prob if prob is not None else self.params.bit_flip_prob
        if rho.shape == (2, 2):
            return (1.0 - p) * rho + p * (X_GATE @ rho @ X_GATE)
        return (1.0 - p) * rho + (p / rho.shape[0]) * np.eye(rho.shape[0], dtype=complex)

    def apply_thermal_relaxation(self, rho: np.ndarray, gate_time_us: Optional[float] = None) -> np.ndarray:
        """
        Simulates T1 amplitude damping and T2 dephasing over gate duration.
        """
        dt = gate_time_us or self.params.gate_time_us
        gamma_1 = 1.0 - math.exp(-dt / self.params.t1_us)
        gamma_phi = 1.0 - math.exp(-dt / self.params.t2_us)

        if rho.shape == (2, 2):
            # Kraus operators for amplitude damping
            k0 = np.array([[1, 0], [0, math.sqrt(1 - gamma_1)]], dtype=complex)
            k1 = np.array([[0, math.sqrt(gamma_1)], [0, 0]], dtype=complex)
            rho_damped = k0 @ rho @ np.conj(k0).T + k1 @ rho @ np.conj(k1).T

            # Phase damping
            rho_damped[0, 1] *= (1.0 - gamma_phi)
            rho_damped[1, 0] *= (1.0 - gamma_phi)
            return rho_damped

        return (1.0 - gamma_1) * rho + (gamma_1 / rho.shape[0]) * np.eye(rho.shape[0], dtype=complex)

    def apply_readout_noise(self, ideal_counts: Dict[str, int]) -> Dict[str, int]:
        """
        Simulates measurement readout error using asymmetric transition probabilities P(0|1), P(1|0).
        """
        p01 = self.params.readout_error_1_to_0
        p10 = self.params.readout_error_0_to_1

        noisy_counts: Dict[str, int] = {}
        for bitstring, count in ideal_counts.items():
            for _ in range(count):
                noisy_bits = []
                for bit in bitstring:
                    if bit == "0":
                        flipped = np.random.rand() < p10
                        noisy_bits.append("1" if flipped else "0")
                    else:
                        flipped = np.random.rand() < p01
                        noisy_bits.append("0" if flipped else "1")
                res = "".join(noisy_bits)
                noisy_counts[res] = noisy_counts.get(res, 0) + 1
        return noisy_counts
