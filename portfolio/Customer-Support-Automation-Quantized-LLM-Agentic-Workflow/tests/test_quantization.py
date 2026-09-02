"""
Unit tests for Quantization Config & Analytical VRAM Estimator
"""
import pytest
from src.core.quantization import QuantizationConfig, VRAMEstimator, QuantizationType


def test_quantization_config_defaults():
    config = QuantizationConfig()
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True
    data = config.to_dict()
    assert data["quant_type"] == "bnb_4bit_nf4"


def test_vram_estimator_7b_savings():
    metrics = VRAMEstimator.estimate_vram(
        param_count_billions=7.0,
        quant_type=QuantizationType.BITSANDBYTES_4BIT_NF4,
        context_window=2048,
        batch_size=1,
    )
    assert metrics["parameter_memory_gb"] > 0
    assert metrics["total_vram_gb"] < metrics["baseline_fp16_vram_gb"]
    assert metrics["vram_reduction_pct"] > 60.0
    assert metrics["total_vram_gb"] < 6.0  # 7B 4-bit fits in ~4.5-5.5 GB


def test_vram_estimator_context_scaling():
    short_ctx = VRAMEstimator.estimate_vram(param_count_billions=7.0, context_window=1024)
    long_ctx = VRAMEstimator.estimate_vram(param_count_billions=7.0, context_window=8192)
    assert long_ctx["kv_cache_memory_gb"] > short_ctx["kv_cache_memory_gb"]
    assert long_ctx["total_vram_gb"] > short_ctx["total_vram_gb"]
