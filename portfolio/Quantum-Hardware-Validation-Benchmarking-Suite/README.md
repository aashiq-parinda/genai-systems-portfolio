# Quantum Hardware Validation & Benchmarking Suite

[![CI Pipeline](https://github.com/aashiq-parinda/Quantum-Hardware-Validation-Benchmarking-Suite/actions/workflows/ci.yml/badge.svg)](https://github.com/aashiq-parinda/Quantum-Hardware-Validation-Benchmarking-Suite/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![NumPy & SciPy](https://img.shields.io/badge/Scientific%20Computing-NumPy%20%26%20SciPy-blue.svg)](https://scipy.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-grade scientific computing and quantum hardware evaluation suite** for validating quantum processors against fundamental quantum-mechanical principles, profiling noise channels, evaluating state fidelities, and implementing **Quantum Error Mitigation (Zero Noise Extrapolation & Readout Calibration)**.

---

## 📺 Video Walkthrough & Scientific Demo

[![Watch the Video Walkthrough](https://img.youtube.com/vi/YOUR_YOUTUBE_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=YOUR_YOUTUBE_VIDEO_ID)

> 💡 *Click the thumbnail above to watch the quantum circuit simulator walkthrough, noise channel profiling, and Zero Noise Extrapolation (ZNE) error mitigation demo on YouTube.*

---

## 🎯 Purpose & Scientific Context

Quantum computing platforms (IBM Quantum, IonQ, Quantinuum, Rigetti) require rigorous, reproducible verification before executing mission-critical algorithms (VQE, QAOA, QPE). 

This suite provides a hardware-agnostic benchmark framework:
1. **Mathematical Axiom Verification**: Deterministic verification of the Born rule, normalization, phase interference, Bell state entanglement, unitary conservation ($U^\dagger U = I$), and the no-cloning theorem.
2. **Comprehensive Noise Modeling**: Density matrix simulation of depolarizing, bit-flip, phase-flip, readout confusion matrices, and $T_1/T_2$ thermal relaxation channels.
3. **Quantum Error Mitigation (QEM)**: Zero Noise Extrapolation (ZNE) and readout confusion matrix inversion.
4. **DiVincenzo Scorecard**: Systematic evaluation against the 7 DiVincenzo criteria for physical quantum computers.

---

## 🏛️ System Architecture

```
                                  [ Quantum Circuit Specification ]
                                                 │
                                                 ▼
                              ┌─────────────────────────────────────┐
                              │ 1. Statevector Simulation Engine    │
                              │ - Universal Gate Set (H, S, T, CX)  │
                              │ - N-Qubit Tensor Product Expansion  │
                              └──────────────────┬──────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
         ┌──────────────────────────────┐                  ┌──────────────────────────────┐
         │ 2. Axiom Validation Engine   │                  │ 3. Noise Simulation Engine   │
         │ - Born Rule & Normalization  │                  │ - Depolarizing Channel       │
         │ - Superposition Interference │                  │ - Bit/Phase Flip Channels    │
         │ - Bell State Entanglement    │                  │ - T1 / T2 Thermal Relaxation │
         │ - No-Cloning Theorem Proof   │                  │ - Readout Confusion Matrix   │
         └──────────────┬───────────────┘                  └──────────────┬───────────────┘
                        │                                                 │
                        ▼                                                 ▼
         ┌──────────────────────────────┐                  ┌──────────────────────────────┐
         │ 4. DiVincenzo 7-Criteria     │                  │ 5. Error Mitigation (QEM)    │
         │ - Coherence Time Tracking    │                  │ - Zero Noise Extrapolation   │
         │ - Universal Gate Fidelity    │                  │ - Matrix Inversion Filter    │
         └──────────────┬───────────────┘                  └──────────────┬───────────────┘
                        │                                                 │
                        └────────────────────────┬────────────────────────┘
                                                 ▼
                              [ Empirical Benchmark Receipts & Report ]
```

---

## 📊 Verified Empirical Receipts

Empirical results generated via `src.eval.benchmark`:

### 🔬 1. Quantum Axioms Verification
| Axiomatic Principle | Verification Status | Mathematical Verification Method |
| :--- | :--- | :--- |
| **Wavefunction Normalization** | ✅ **PASS** | $\sum_i \|c_i\|^2 = 1.0 \pm 10^{-6}$ |
| **Born Rule Convergence** | ✅ **PASS** | Frequency matches probability amplitudes within binomial error |
| **Quantum Phase Interference** | ✅ **PASS** | $H \cdot Z \cdot H \|0\rangle \to \|1\rangle$ destructive cancellation |
| **Bell State Entanglement** | ✅ **PASS** | $\|\Phi^+\rangle = \frac{\|00\rangle + \|11\rangle}{\sqrt{2}}$, zero cross-terms |
| **Unitary Conservation** | ✅ **PASS** | $\|U^\dagger U - I\|_F < 10^{-6}$ |
| **No-Cloning Theorem** | ✅ **PASS** | Non-isometry proof $\langle \psi\|\phi\rangle^2 \neq \langle \psi\|\phi\rangle$ |

### ⚡ 2. Circuit State Fidelity & Noise Benchmarks
| Circuit Benchmark | Ideal State Fidelity | Noisy Fidelity (Depolarizing + Thermal) | Trace Distance | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Bell State $\|\Phi^+\rangle$** | `1.0000` | `0.9835` | `0.0165` | `0.015 ms` |
| **3-Qubit GHZ State** | `1.0000` | `0.9737` | `0.0263` | Sub-millisecond |

### 🛡️ 3. Quantum Error Mitigation (Zero Noise Extrapolation)
| Parameter | Unmitigated Noisy Expectation | ZNE-Mitigated Expectation | Error Recovery Delta |
| :--- | :--- | :--- | :--- |
| **Expectation $\langle Z \rangle$** | `0.9412` | **`1.0000`** | **+0.0588 (+6.25% recovery)** |

---

## 🛠️ Repository Structure

```text
├── .github/workflows/
│   └── ci.yml                   # Automated multi-version Python CI/CD pipeline
├── src/
│   ├── core/                    # Core scientific quantum computing modules
│   │   ├── __init__.py
│   │   ├── simulator.py         # Statevector circuit simulator & gate operations
│   │   ├── principles_validator.py # Mathematical verification of quantum axioms
│   │   ├── noise_engine.py      # Depolarizing, bit-flip, and T1/T2 noise channels
│   │   ├── error_mitigation.py  # ZNE & Readout confusion matrix inversion
│   │   └── fidelity_benchmarker.py # Uhlmann fidelity & DiVincenzo scorecard
│   ├── api/                     # Production FastAPI REST application
│   │   ├── __init__.py
│   │   └── app.py               # REST simulation, validation, and scorecard endpoints
│   └── eval/                    # Standalone reproducible benchmarking harness
│       ├── __init__.py
│       └── benchmark.py         # Automated validation runner and report generator
├── tests/
│   ├── test_simulator.py        # Gate algebra and statevector evolution tests
│   ├── test_principles.py       # Axiom verification tests (Born rule, no-cloning)
│   ├── test_noise.py            # Noise channel and decoherence tests
│   ├── test_mitigation.py       # ZNE and readout error mitigation tests
│   └── test_api.py              # FastAPI endpoint integration tests
├── results/
│   ├── benchmark_metrics.json   # Verified benchmark receipts
│   └── quantum_validation_report.md # Empirical validation report
├── quantum_validation_suite.ipynb # Interactive simulation notebook
├── requirements.txt             # Pinned production dependencies
└── README.md
```

---

## ⚡ Quickstart

### 1. Installation

```bash
git clone https://github.com/aashiq-parinda/Quantum-Hardware-Validation-Benchmarking-Suite.git
cd Quantum-Hardware-Validation-Benchmarking-Suite
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
pytest tests/ -v
```

### 3. Run Benchmark & Validation Harness

```bash
python -m src.eval.benchmark --output results/
```

### 4. Launch FastAPI REST Server

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation available at `http://localhost:8000/docs`.

---

## 📡 API Usage Examples

### Simulate Bell State Circuit (`POST /v1/quantum/simulate`)
```bash
curl -X POST "http://localhost:8000/v1/quantum/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "num_qubits": 2,
    "gates": [
      {"gate": "H", "qubit": 0},
      {"gate": "CNOT", "control": 0, "target": 1}
    ],
    "shots": 1000
  }'
```

**Response**:
```json
{
  "num_qubits": 2,
  "probabilities": [0.5, 0.0, 0.0, 0.5],
  "counts": {
    "00": 492,
    "11": 508
  },
  "statevector_norm": 1.0,
  "execution_time_ms": 0.082
}
```

### Apply Zero Noise Extrapolation (`POST /v1/quantum/mitigate-error`)
```bash
curl -X POST "http://localhost:8000/v1/quantum/mitigate-error" \
  -H "Content-Type: application/json" \
  -d '{
    "noise_scales": [1.0, 2.0, 3.0],
    "measured_expectations": [0.9412, 0.8824, 0.8236]
  }'
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.
