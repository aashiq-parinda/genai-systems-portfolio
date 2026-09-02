"""
Quantization Configuration, VRAM Estimator, and Inference Optimization Engine
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional


class QuantizationType(str, Enum):
    NONE = "fp16"
    BITSANDBYTES_4BIT_NF4 = "bnb_4bit_nf4"
    BITSANDBYTES_4BIT_FP4 = "bnb_4bit_fp4"
    BITSANDBYTES_8BIT = "bnb_8bit"
    GPTQ_4BIT = "gptq_4bit"
    AWQ_4BIT = "awq_4bit"


@dataclass
class QuantizationConfig:
    """Configuration schema for quantized model deployment."""
    quant_type: QuantizationType = QuantizationType.BITSANDBYTES_4BIT_NF4
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_memory_mb: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quant_type": self.quant_type.value,
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "device_map": self.device_map,
            "max_memory_mb": self.max_memory_mb,
        }


class VRAMEstimator:
    """
    Analytical memory estimator for LLM deployments.
    Calculates parameter memory, KV cache, activation memory, and VRAM reduction ratios.
    """

    BYTES_PER_PARAM = {
        QuantizationType.NONE: 2.0,  # 16-bit float
        QuantizationType.BITSANDBYTES_8BIT: 1.0,  # 8-bit
        QuantizationType.BITSANDBYTES_4BIT_NF4: 0.55,  # 4-bit + scale metadata
        QuantizationType.BITSANDBYTES_4BIT_FP4: 0.55,
        QuantizationType.GPTQ_4BIT: 0.52,
        QuantizationType.AWQ_4BIT: 0.52,
    }

    @classmethod
    def estimate_vram(
        cls,
        param_count_billions: float = 7.0,
        quant_type: QuantizationType = QuantizationType.BITSANDBYTES_4BIT_NF4,
        context_window: int = 2048,
        batch_size: int = 1,
        num_layers: int = 32,
        hidden_size: int = 4096,
        num_heads: int = 32,
    ) -> Dict[str, float]:
        """
        Estimates total VRAM requirements in Gigabytes (GB).
        """
        bytes_per_param = cls.BYTES_PER_PARAM.get(quant_type, 2.0)
        param_memory_gb = (param_count_billions * 1e9 * bytes_per_param) / (1024**3)

        # KV cache = 2 * num_layers * hidden_size * context_window * batch_size * 2 bytes (FP16)
        kv_cache_bytes = 2 * num_layers * hidden_size * context_window * batch_size * 2
        kv_cache_gb = kv_cache_bytes / (1024**3)

        # Activation memory overhead approx ~20% of KV cache + 0.3GB runtime buffer
        activation_memory_gb = (kv_cache_gb * 0.2) + 0.3
        total_vram_gb = param_memory_gb + kv_cache_gb + activation_memory_gb

        # Baseline FP16 total for savings calculation
        baseline_param_gb = (param_count_billions * 1e9 * 2.0) / (1024**3)
        baseline_total_gb = baseline_param_gb + kv_cache_gb + activation_memory_gb
        vram_reduction_pct = ((baseline_total_gb - total_vram_gb) / baseline_total_gb) * 100.0

        return {
            "parameter_memory_gb": round(param_memory_gb, 2),
            "kv_cache_memory_gb": round(kv_cache_gb, 2),
            "activation_memory_gb": round(activation_memory_gb, 2),
            "total_vram_gb": round(total_vram_gb, 2),
            "baseline_fp16_vram_gb": round(baseline_total_gb, 2),
            "vram_reduction_pct": round(vram_reduction_pct, 2),
        }
