"""
Exact Matrix Statevector Quantum Circuit Simulator
"""
import cmath
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Standard Single-Qubit Gates
I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)
H_GATE = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0)
S_GATE = np.array([[1, 0], [0, 1j]], dtype=complex)
T_GATE = np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4.0)]], dtype=complex)


def rx_gate(theta: float) -> np.ndarray:
    return np.array([
        [math.cos(theta / 2.0), -1j * math.sin(theta / 2.0)],
        [-1j * math.sin(theta / 2.0), math.cos(theta / 2.0)]
    ], dtype=complex)


def ry_gate(theta: float) -> np.ndarray:
    return np.array([
        [math.cos(theta / 2.0), -math.sin(theta / 2.0)],
        [math.sin(theta / 2.0), math.cos(theta / 2.0)]
    ], dtype=complex)


def rz_gate(theta: float) -> np.ndarray:
    return np.array([
        [cmath.exp(-1j * theta / 2.0), 0],
        [0, cmath.exp(1j * theta / 2.0)]
    ], dtype=complex)


class QuantumCircuit:
    """
    High-Performance Statevector Quantum Circuit Engine with Multi-Qubit Tensor Product Expansion.
    """

    def __init__(self, num_qubits: int):
        if num_qubits < 1 or num_qubits > 16:
            raise ValueError("Number of qubits must be between 1 and 16.")
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self.statevector = np.zeros(self.dim, dtype=complex)
        self.statevector[0] = 1.0  # Initialize in |0...0> ground state
        self.operations: List[Dict[str, Any]] = []

    def h(self, qubit: int) -> "QuantumCircuit":
        """Applies Hadamard gate."""
        return self._apply_single_qubit_gate(H_GATE, qubit, "H")

    def x(self, qubit: int) -> "QuantumCircuit":
        """Applies Pauli-X gate."""
        return self._apply_single_qubit_gate(X_GATE, qubit, "X")

    def y(self, qubit: int) -> "QuantumCircuit":
        """Applies Pauli-Y gate."""
        return self._apply_single_qubit_gate(Y_GATE, qubit, "Y")

    def z(self, qubit: int) -> "QuantumCircuit":
        """Applies Pauli-Z gate."""
        return self._apply_single_qubit_gate(Z_GATE, qubit, "Z")

    def s(self, qubit: int) -> "QuantumCircuit":
        """Applies Phase S gate."""
        return self._apply_single_qubit_gate(S_GATE, qubit, "S")

    def t(self, qubit: int) -> "QuantumCircuit":
        """Applies Phase T gate."""
        return self._apply_single_qubit_gate(T_GATE, qubit, "T")

    def rx(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Applies Rx rotation gate."""
        return self._apply_single_qubit_gate(rx_gate(theta), qubit, f"Rx({theta:.2f})")

    def ry(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Applies Ry rotation gate."""
        return self._apply_single_qubit_gate(ry_gate(theta), qubit, f"Ry({theta:.2f})")

    def rz(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Applies Rz rotation gate."""
        return self._apply_single_qubit_gate(rz_gate(theta), qubit, f"Rz({theta:.2f})")

    def _apply_single_qubit_gate(self, gate_matrix: np.ndarray, qubit: int, name: str) -> "QuantumCircuit":
        if qubit < 0 or qubit >= self.num_qubits:
            raise IndexError(f"Qubit index {qubit} out of range (0-{self.num_qubits - 1}).")

        # Kronecker tensor product expansion: I x ... x Gate x ... x I
        operator = np.array([[1.0]], dtype=complex)
        for i in range(self.num_qubits):
            if i == qubit:
                operator = np.kron(operator, gate_matrix)
            else:
                operator = np.kron(operator, I2)

        self.statevector = operator @ self.statevector
        self.operations.append({"gate": name, "qubits": [qubit]})
        return self

    def cnot(self, control: int, target: int) -> "QuantumCircuit":
        """Applies Controlled-NOT (CX) entangling gate."""
        if control == target or control < 0 or control >= self.num_qubits or target < 0 or target >= self.num_qubits:
            raise ValueError(f"Invalid control ({control}) or target ({target}) qubit index.")

        # Construct full unitary for CNOT: |0><0| x I + |1><1| x X on specified qubits
        p0 = np.array([[1, 0], [0, 0]], dtype=complex)
        p1 = np.array([[0, 0], [0, 1]], dtype=complex)

        op0 = np.array([[1.0]], dtype=complex)
        op1 = np.array([[1.0]], dtype=complex)

        for i in range(self.num_qubits):
            if i == control:
                op0 = np.kron(op0, p0)
                op1 = np.kron(op1, p1)
            elif i == target:
                op0 = np.kron(op0, I2)
                op1 = np.kron(op1, X_GATE)
            else:
                op0 = np.kron(op0, I2)
                op1 = np.kron(op1, I2)

        cnot_matrix = op0 + op1
        self.statevector = cnot_matrix @ self.statevector
        self.operations.append({"gate": "CNOT", "qubits": [control, target]})
        return self

    def cz(self, control: int, target: int) -> "QuantumCircuit":
        """Applies Controlled-Z gate."""
        self.h(target)
        self.cnot(control, target)
        self.h(target)
        return self

    def get_probabilities(self) -> np.ndarray:
        """Returns array of probability values |c_i|^2."""
        return np.abs(self.statevector) ** 2

    def sample_measurements(self, shots: int = 1000, seed: Optional[int] = None) -> Dict[str, int]:
        """Samples discrete measurement outcomes according to Born rule probabilities."""
        if seed is not None:
            np.random.seed(seed)
        probs = self.get_probabilities()
        # Ensure exact probability sum of 1.0
        probs = probs / np.sum(probs)
        outcomes = np.random.choice(self.dim, size=shots, p=probs)

        counts: Dict[str, int] = {}
        for outcome in outcomes:
            bitstring = format(outcome, f"0{self.num_qubits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def get_density_matrix(self) -> np.ndarray:
        """Returns density matrix rho = |psi><psi|."""
        return np.outer(self.statevector, np.conj(self.statevector))
