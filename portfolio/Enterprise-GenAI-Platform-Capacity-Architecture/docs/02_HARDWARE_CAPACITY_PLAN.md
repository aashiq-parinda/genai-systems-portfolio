# 02. Hardware Capacity & Sizing Plan (10K $\rightarrow$ 1M Concurrency)

## Executive Sizing Summary

Designing private inference infrastructure for a multi-division conglomerate requires first-principles capacity planning rather than arbitrary GPU leasing. This document establishes the mathematical derivations governing VRAM, KV-cache, concurrency, and hardware cluster topologies.

---

## 🧮 1. Mathematical Derivations

### A. Model Weights Memory Footprint ($M_{\text{weights}}$)
For a model with $P$ parameters stored at precision $B$ bytes per parameter (with a $5\%$ CUDA memory allocator and framework overhead buffer):

$$M_{\text{weights}} = \frac{P \times 10^9 \times B}{1024^3} \times 1.05 \quad (\text{GB})$$

* For **70B at FP8 ($B = 1.0$)**:
  $$M_{\text{weights}} = \frac{70 \times 10^9 \times 1.0}{1024^3} \times 1.05 \approx 68.45\text{ GB}$$
* For **70B at FP16 ($B = 2.0$)**:
  $$M_{\text{weights}} \approx 136.90\text{ GB}$$

---

### B. KV-Cache Memory per Concurrent Request ($M_{\text{KV}}$)
For a Transformer model utilizing **Grouped-Query Attention (GQA)** with $L$ layers, $N_{\text{KV}}$ key-value heads, head dimension $D_{\text{head}}$, and context window length $T$:

$$M_{\text{KV-Token}} = 2 \times L \times N_{\text{KV}} \times D_{\text{head}} \times B_{\text{KV}} \quad (\text{bytes/token})$$

For Claude-X 70B ($L=80, N_{\text{KV}}=8, D_{\text{head}}=128, B_{\text{KV}}=1\text{ byte for FP8}$):
$$M_{\text{KV-Token}} = 2 \times 80 \times 8 \times 128 \times 1 = 163,840\text{ bytes} \approx 160\text{ KB/token}$$

For an average active enterprise context length of $T = 4,096\text{ tokens}$:
$$M_{\text{KV-Request}} = \frac{160\text{ KB} \times 4096}{1024} = 640.0\text{ MB per active request}$$

---

### C. Total Cluster VRAM Requirement for Concurrency $C$
$$M_{\text{Total}} = M_{\text{weights}} + \left( C \times \frac{M_{\text{KV-Request}}}{1024} \right) + M_{\text{activations}} \quad (\text{GB})$$

For $C = 10,000$ concurrent streams:
$$M_{\text{KV-Total}} = 10,000 \times 0.640\text{ GB} = 6,400\text{ GB (6.4 TB VRAM)}$$

---

## ⚡ 2. Tensor & Pipeline Parallelism Topology

Because a single 70B model in FP8 ($68.5\text{ GB}$) exceeds the safe working boundary of a single 80GB GPU once activations and KV-cache are allocated, we partition across GPUs:

1. **Tensor Parallelism (TP = 2 / TP = 4)**:
   * Partitions attention heads and MLP matrices across intra-node NVLink ($900\text{ GB/s}$ bidirectional bandwidth).
   * TP=2 on $2 \times \text{H100-80GB}$ yields $160\text{ GB}$ total VRAM per replica:
     - Model weights: $68.5\text{ GB}$
     - Activations / Overhead: $11.5\text{ GB}$
     - Remaining for dynamic KV-Cache: $80.0\text{ GB}$ ($80,000\text{ MB}$)
     - Concurrent streams supported per replica: $\frac{80,000\text{ MB}}{640\text{ MB}} = 125\text{ streams}$

2. **Continuous Batching (vLLM / TensorRT-LLM PagedAttention)**:
   * Dynamic iteration-level scheduling eliminates bubble waste from variable prompt/generation lengths.
   * Virtual memory paging for KV cache eliminates $60–80\%$ memory fragmentation.

---

## 📊 3. Sizing Matrix Across Concurrency Milestones

| Concurrency Tier | Active Streams ($C$) | Replicas Needed (TP=2) | Total H100 80GB GPUs | 8x HGX Server Nodes | Peak Aggregate Tokens/Sec | Est. Monthly Compute (₹ Cr) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Initial Pilot** | 1,000 | 8 | 16 | 2 | 35,000 | ₹0.34 Cr |
| **Production Launch** | **10,000** | **80 + 1 (N+1)** | **162** | **21** | **350,000** | **₹3.48 Cr** |
| **Enterprise Scale** | 50,000 | 400 + 2 | 804 | 101 | 1,750,000 | ₹17.27 Cr |
| **Conglomerate-Wide** | 100,000 | 800 + 4 | 1,608 | 201 | 3,500,000 | ₹34.54 Cr |
| **Target 1M Scale** | 1,000,000 | 8,000 + 16 | 16,032 | 2,004 | 35,000,000 | ₹344.40 Cr |

> **FinOps Optimization**: By routing 70% of traffic to the 8B SLM tier (which requires only 1x L40S or 1x A100 per 400 streams), the blended production GPU count for 10K concurrency drops from **162 H100s $\rightarrow$ 68 mixed H100/L40S GPUs**, reducing compute expenditure by over **58%**.
