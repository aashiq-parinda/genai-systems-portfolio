"""Unit tests validating first-principles hardware capacity calculations."""

import pytest
import math

from src.sizing.gpu_capacity_calculator import (
    GPUCapacityCalculator,
    MODEL_PROFILES,
    GPU_PROFILES,
    ModelSpec,
)
from src.sizing.latency_simulator import LatencySimulator


def test_weights_vram_calculation():
    calc = GPUCapacityCalculator()
    model_70b = MODEL_PROFILES["Claude-X-Frontier-70B"]
    
    # FP8 precision (1 byte per parameter)
    vram_fp8 = calc.calculate_weights_vram(model_70b, precision="fp8")
    assert 65.0 < vram_fp8 < 72.0, f"Expected ~68.5 GB, got {vram_fp8}"

    # FP16 precision (2 bytes per parameter)
    vram_fp16 = calc.calculate_weights_vram(model_70b, precision="fp16")
    assert 130.0 < vram_fp16 < 142.0, f"Expected ~137 GB, got {vram_fp16}"


def test_kv_cache_per_token_and_request():
    calc = GPUCapacityCalculator()
    model_70b = MODEL_PROFILES["Claude-X-Frontier-70B"]

    # 2 * 80 layers * 8 kv_heads * 128 head_dim * 1 byte = 163,840 bytes (160 KB)
    bytes_per_token = calc.calculate_kv_cache_per_token_bytes(model_70b, kv_precision="fp8")
    assert bytes_per_token == 163840

    # 4096 context length = 163840 * 4096 / (1024^2) = 640.0 MB
    mb_per_request = calc.calculate_kv_cache_per_request_mb(model_70b, context_length=4096, kv_precision="fp8")
    assert mb_per_request == 640.0


def test_tensor_parallelism_determination():
    calc = GPUCapacityCalculator()
    gpu_h100 = GPU_PROFILES["NVIDIA-H100-SXM-80GB"]
    
    # 70B in FP8 (~68.5 GB) requires TP=2 to fit weights + KV cache safely
    model_70b = MODEL_PROFILES["Claude-X-Frontier-70B"]
    weights_vram = calc.calculate_weights_vram(model_70b, precision="fp8")
    tp = calc.calculate_min_tensor_parallelism(weights_vram, gpu_h100)
    assert tp == 2

    # 8B in FP8 (~7.8 GB) fits on TP=1
    model_8b = MODEL_PROFILES["Enterprise-SLM-8B"]
    weights_8b = calc.calculate_weights_vram(model_8b, precision="fp8")
    tp_8b = calc.calculate_min_tensor_parallelism(weights_8b, gpu_h100)
    assert tp_8b == 1


def test_cluster_sizing_10k_concurrency():
    calc = GPUCapacityCalculator()
    model_70b = MODEL_PROFILES["Claude-X-Frontier-70B"]
    gpu_h100 = GPU_PROFILES["NVIDIA-H100-SXM-80GB"]

    result = calc.size_cluster(
        model=model_70b,
        gpu=gpu_h100,
        target_concurrency=10000,
        precision="fp8",
        kv_precision="fp8",
        average_context_tokens=4096,
        n_plus_one=True
    )

    assert result.target_concurrency == 10000
    assert result.gpus_per_replica == 2
    assert result.total_gpus_required > 100
    assert result.total_nodes_8x >= 15
    assert result.peak_aggregate_tokens_per_sec == 350000.0


def test_latency_simulator_sla():
    sim = LatencySimulator()
    profile = sim.simulate_profile(
        concurrency=10000,
        active_gpu_replicas=80,
        prompt_tokens=2048,
        generation_tokens=512
    )

    assert profile.ttft_p50_ms < 600.0
    assert profile.ttft_p95_ms < 1000.0  # Must meet 1s SLA
    assert profile.e2e_p95_s < 5.0       # Must meet 5s SLA
    assert profile.sla_violation_rate == 0.0
