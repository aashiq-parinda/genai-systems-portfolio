"""
Production FastAPI Server for Quantum Circuit Simulation, Noise Profiling & Hardware Validation
"""
import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

from src.core.simulator import QuantumCircuit
from src.core.principles_validator import QuantumPrinciplesValidator
from src.core.noise_engine import QuantumNoiseEngine, NoiseParameters
from src.core.error_mitigation import QuantumErrorMitigator
from src.core.fidelity_benchmarker import FidelityBenchmarker

app = FastAPI(
    title="Quantum Hardware Validation & Benchmarking API",
    description="Production-grade Quantum Simulator, Noise Modeling, and Principles Validation Engine",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Schemas
# ==========================================

class CircuitExecutionRequest(BaseModel):
    num_qubits: int = Field(2, ge=1, le=10)
    gates: List[Dict[str, Any]] = Field(..., description="List of gate operations e.g. [{'gate': 'H', 'qubit': 0}]")
    shots: Optional[int] = Field(1000, ge=10, le=100000)


class CircuitExecutionResponse(BaseModel):
    num_qubits: int
    probabilities: List[float]
    counts: Dict[str, int]
    statevector_norm: float
    execution_time_ms: float


class ZNEMitigationRequest(BaseModel):
    noise_scales: List[float] = Field([1.0, 2.0, 3.0])
    measured_expectations: List[float] = Field([0.85, 0.72, 0.61])


class NoiseBenchmarkRequest(BaseModel):
    circuit_type: str = Field("bell", description="'bell' or 'ghz'")
    depolarizing_prob: Optional[float] = Field(0.02, ge=0.0, le=0.5)


# ==========================================
# Endpoints
# ==========================================

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "quantum-validation-suite",
        "version": "2.0.0",
        "timestamp": time.time(),
    }


@app.post(
    "/v1/quantum/simulate",
    response_model=CircuitExecutionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Circuit Simulation"]
)
def simulate_circuit(request: CircuitExecutionRequest):
    """
    Executes statevector simulation on arbitrary multi-qubit gate sequences.
    """
    try:
        start_time = time.perf_counter()
        qc = QuantumCircuit(request.num_qubits)

        for op in request.gates:
            gate = op.get("gate", "").upper()
            if gate == "H":
                qc.h(op["qubit"])
            elif gate == "X":
                qc.x(op["qubit"])
            elif gate == "Y":
                qc.y(op["qubit"])
            elif gate == "Z":
                qc.z(op["qubit"])
            elif gate == "S":
                qc.s(op["qubit"])
            elif gate == "T":
                qc.t(op["qubit"])
            elif gate == "CNOT":
                qc.cnot(op["control"], op["target"])
            elif gate == "CZ":
                qc.cz(op["control"], op["target"])

        probs = [float(p) for p in qc.get_probabilities()]
        counts = qc.sample_measurements(shots=request.shots or 1000)
        norm = float(np.sum(probs))
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return CircuitExecutionResponse(
            num_qubits=request.num_qubits,
            probabilities=probs,
            counts=counts,
            statevector_norm=round(norm, 6),
            execution_time_ms=round(elapsed_ms, 3),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Quantum circuit simulation error: {str(e)}"
        )


@app.post("/v1/quantum/validate-principles", tags=["Quantum Principles"])
def validate_quantum_principles():
    """
    Executes automated mathematical tests for Born rule, normalization, entanglement, and no-cloning.
    """
    return QuantumPrinciplesValidator.run_all_validations()


@app.post("/v1/quantum/benchmark-noise", tags=["Noise & Fidelity"])
def benchmark_noise(request: NoiseBenchmarkRequest):
    """
    Benchmarks circuit state fidelity under configured depolarizing and thermal relaxation noise channels.
    """
    noise_params = NoiseParameters(depolarizing_prob=request.depolarizing_prob or 0.02)
    engine = QuantumNoiseEngine(noise_params)

    if request.circuit_type.lower() == "ghz":
        return FidelityBenchmarker.benchmark_ghz_state(num_qubits=3, noise_engine=engine)
    return FidelityBenchmarker.benchmark_bell_state(noise_engine=engine)


@app.post("/v1/quantum/mitigate-error", tags=["Error Mitigation"])
def apply_error_mitigation(request: ZNEMitigationRequest):
    """
    Applies Zero Noise Extrapolation (ZNE) to recover noiseless expectation values.
    """
    return QuantumErrorMitigator.zero_noise_extrapolation(
        noise_scale_factors=request.noise_scales,
        expectation_values=request.measured_expectations,
    )


@app.get("/v1/quantum/divincenzo-scorecard", tags=["Hardware Scorecard"])
def get_divincenzo_scorecard():
    """
    Returns DiVincenzo 7-criteria quantum hardware readiness scorecard.
    """
    return FidelityBenchmarker.evaluate_divincenzo_scorecard()
