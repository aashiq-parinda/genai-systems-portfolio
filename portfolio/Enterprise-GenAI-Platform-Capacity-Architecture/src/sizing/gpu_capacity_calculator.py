"""GPU Capacity & Sizing Calculator for Enterprise LLM Inference.

Implements first-principles mathematical derivations for:
- Model Weights VRAM footprint across precisions (FP16, BF16, FP8, INT4)
- Per-user and multi-tenant KV-Cache memory consumption
- Continuous batching throughput & memory headroom
- Tensor Parallelism (TP) and Pipeline Parallelism (PP) cluster topology
- Multi-tier GPU node sizing for 10K -> 1M concurrency SLAs with N+1 redundancy
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


@dataclass
class ModelSpec:
    """Model architecture parameters."""
    name: str
    num_parameters_billions: float
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int  # For Grouped Query Attention (GQA)
    hidden_dimension: int
    max_context_length: int = 8192
    vocab_size: int = 128000

    @property
    def head_dim(self) -> int:
        return self.hidden_dimension // self.num_attention_heads


@dataclass
class GPUHardwareSpec:
    """Hardware specifications for target accelerators."""
    name: str
    vram_gb: float
    memory_bandwidth_gb_s: float
    fp16_tflops: float
    fp8_tflops: float
    interconnect_gb_s: float  # NVLink / PCIe bidirectional bandwidth
    tdp_watts: int
    typical_hourly_cost_usd: float


# Standard Model Reference Profiles
MODEL_PROFILES: Dict[str, ModelSpec] = {
    "Claude-X-Frontier-70B": ModelSpec(
        name="Claude-X-Frontier-70B",
        num_parameters_billions=70.0,
        num_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,  # GQA ratio 8:1
        hidden_dimension=8192,
        max_context_length=8192,
    ),
    "Enterprise-SLM-8B": ModelSpec(
        name="Enterprise-SLM-8B",
        num_parameters_billions=8.0,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_dimension=4096,
        max_context_length=8192,
    ),
    "Frontier-Reasoning-405B": ModelSpec(
        name="Frontier-Reasoning-405B",
        num_parameters_billions=405.0,
        num_layers=126,
        num_attention_heads=128,
        num_key_value_heads=16,
        hidden_dimension=16384,
        max_context_length=16384,
    )
}

# Standard Hardware Reference Profiles
GPU_PROFILES: Dict[str, GPUHardwareSpec] = {
    "NVIDIA-H100-SXM-80GB": GPUHardwareSpec(
        name="NVIDIA-H100-SXM-80GB",
        vram_gb=80.0,
        memory_bandwidth_gb_s=3350.0,
        fp16_tflops=989.0,
        fp8_tflops=1979.0,
        interconnect_gb_s=900.0,
        tdp_watts=700,
        typical_hourly_cost_usd=3.50,
    ),
    "NVIDIA-A100-SXM-80GB": GPUHardwareSpec(
        name="NVIDIA-A100-SXM-80GB",
        vram_gb=80.0,
        memory_bandwidth_gb_s=2039.0,
        fp16_tflops=312.0,
        fp8_tflops=624.0,
        interconnect_gb_s=600.0,
        tdp_watts=400,
        typical_hourly_cost_usd=2.20,
    ),
    "NVIDIA-L40S-48GB": GPUHardwareSpec(
        name="NVIDIA-L40S-48GB",
        vram_gb=48.0,
        memory_bandwidth_gb_s=864.0,
        fp16_tflops=366.0,
        fp8_tflops=733.0,
        interconnect_gb_s=64.0,  # PCIe Gen4
        tdp_watts=350,
        typical_hourly_cost_usd=1.25,
    )
}

PRECISION_BYTES: Dict[str, float] = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}


@dataclass
class SizingResult:
    """Structured calculation result for GPU capacity sizing."""
    model_name: str
    gpu_name: str
    precision: str
    target_concurrency: int
    average_context_tokens: int
    model_weights_vram_gb: float
    kv_cache_per_token_bytes: float
    kv_cache_per_request_mb: float
    total_kv_cache_gb: float
    activations_and_overhead_vram_gb: float
    total_cluster_vram_needed_gb: float
    min_tensor_parallelism: int
    vram_per_replica_gb: float
    gpus_per_replica: int
    active_replicas_needed: int
    n_plus_one_replicas: int
    total_gpus_required: int
    total_nodes_8x: int
    peak_aggregate_tokens_per_sec: float
    estimated_monthly_compute_cost_inr_cr: float


class GPUCapacityCalculator:
    """First-principles hardware capacity & sizing engine."""

    def __init__(self, inr_per_usd: float = 84.0):
        self.inr_per_usd = inr_per_usd

    def calculate_weights_vram(self, model: ModelSpec, precision: str = "fp8") -> float:
        """Calculate raw weights memory footprint in GB (with 1.05x allocator padding)."""
        bytes_per_param = PRECISION_BYTES.get(precision.lower(), 1.0)
        raw_gb = (model.num_parameters_billions * 1e9 * bytes_per_param) / (1024 ** 3)
        return round(raw_gb * 1.05, 2)

    def calculate_kv_cache_per_token_bytes(
        self, model: ModelSpec, kv_precision: str = "fp8"
    ) -> float:
        """Calculate KV-cache bytes required per token per request.
        
        Formula for Grouped Query Attention (GQA):
        KV_bytes = 2 (keys + values) * num_layers * num_kv_heads * head_dim * bytes_per_elem
        """
        bytes_per_elem = PRECISION_BYTES.get(kv_precision.lower(), 1.0)
        return 2 * model.num_layers * model.num_key_value_heads * model.head_dim * bytes_per_elem

    def calculate_kv_cache_per_request_mb(
        self, model: ModelSpec, context_length: int, kv_precision: str = "fp8"
    ) -> float:
        """Calculate KV-cache required for a single request of given context length in MB."""
        per_token_bytes = self.calculate_kv_cache_per_token_bytes(model, kv_precision)
        total_bytes = per_token_bytes * context_length
        return round(total_bytes / (1024 ** 2), 2)

    def calculate_min_tensor_parallelism(
        self, model_weights_vram_gb: float, gpu: GPUHardwareSpec
    ) -> int:
        """Determine minimum power-of-two Tensor Parallelism (TP) degree to fit weights."""
        usable_vram_per_gpu = gpu.vram_gb * 0.85  # Reserve 15% for cuda context & temporary tensors
        required_gpus = math.ceil(model_weights_vram_gb / usable_vram_per_gpu)
        # Round up to next power of 2 (1, 2, 4, 8)
        tp = 1
        while tp < required_gpus:
            tp *= 2
        return min(tp, 8)  # Single node TP max is typically 8

    def size_cluster(
        self,
        model: ModelSpec,
        gpu: GPUHardwareSpec,
        target_concurrency: int,
        precision: str = "fp8",
        kv_precision: str = "fp8",
        average_context_tokens: int = 4096,
        tokens_per_sec_per_stream: float = 35.0,
        n_plus_one: bool = True
    ) -> SizingResult:
        """Calculate full cluster topology, replica count, and hardware requirements."""
        weights_vram = self.calculate_weights_vram(model, precision)
        kv_per_token_bytes = self.calculate_kv_cache_per_token_bytes(model, kv_precision)
        kv_per_req_mb = self.calculate_kv_cache_per_request_mb(model, average_context_tokens, kv_precision)
        
        # Total KV cache for concurrent streams
        total_kv_gb = (kv_per_req_mb * target_concurrency) / 1024.0
        
        # Activations and CUDA runtime context overhead (~10% of weights + static 4GB per GPU)
        activations_overhead_gb = weights_vram * 0.10 + 4.0
        
        total_cluster_vram = weights_vram + total_kv_gb + activations_overhead_gb
        
        # Determine TP for a single replica
        tp = self.calculate_min_tensor_parallelism(weights_vram, gpu)
        gpus_per_replica = tp
        replica_total_vram = gpus_per_replica * gpu.vram_gb
        
        # KV Cache budget available per replica after weights
        vram_left_for_kv_per_replica = (replica_total_vram * 0.90) - weights_vram
        if vram_left_for_kv_per_replica <= 0:
            # Scale TP up if possible
            gpus_per_replica = 8
            replica_total_vram = gpus_per_replica * gpu.vram_gb
            vram_left_for_kv_per_replica = (replica_total_vram * 0.90) - weights_vram

        # Concurrent requests supported per single replica
        concurrency_per_replica = max(1, int((vram_left_for_kv_per_replica * 1024.0) / kv_per_req_mb))
        
        active_replicas = math.ceil(target_concurrency / concurrency_per_replica)
        total_replicas = active_replicas + (1 if n_plus_one else 0)
        total_gpus = total_replicas * gpus_per_replica
        total_nodes = math.ceil(total_gpus / 8)

        peak_tokens_sec = target_concurrency * tokens_per_sec_per_stream
        
        # Monthly compute cost in INR (730 hours / month)
        monthly_cost_usd = total_gpus * gpu.typical_hourly_cost_usd * 730
        monthly_cost_inr_cr = (monthly_cost_usd * self.inr_per_usd) / 1e7

        return SizingResult(
            model_name=model.name,
            gpu_name=gpu.name,
            precision=precision,
            target_concurrency=target_concurrency,
            average_context_tokens=average_context_tokens,
            model_weights_vram_gb=weights_vram,
            kv_cache_per_token_bytes=kv_per_token_bytes,
            kv_cache_per_request_mb=kv_per_req_mb,
            total_kv_cache_gb=round(total_kv_gb, 2),
            activations_and_overhead_vram_gb=round(activations_overhead_gb, 2),
            total_cluster_vram_needed_gb=round(total_cluster_vram, 2),
            min_tensor_parallelism=tp,
            vram_per_replica_gb=round(replica_total_vram, 2),
            gpus_per_replica=gpus_per_replica,
            active_replicas_needed=active_replicas,
            n_plus_one_replicas=total_replicas,
            total_gpus_required=total_gpus,
            total_nodes_8x=total_nodes,
            peak_aggregate_tokens_per_sec=round(peak_tokens_sec, 2),
            estimated_monthly_compute_cost_inr_cr=round(monthly_cost_inr_cr, 3),
        )
