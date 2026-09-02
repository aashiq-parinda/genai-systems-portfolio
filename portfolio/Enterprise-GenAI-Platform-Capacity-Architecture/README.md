# Enterprise Multi-Tenant GenAI Platform Architecture & Capacity Planning Blueprint

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Control%20Plane-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pytest](https://img.shields.io/badge/Pytest-100%25%20Passing-success.svg?logo=pytest&logoColor=white)](https://pytest.org)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg?logo=githubactions&logoColor=white)](https://github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Flagship Staff GenAI Systems Architect & Lead Forward Deployed Engineer (FDE) Case Study**: An end-to-end multi-tenant AI control plane, first-principles GPU hardware capacity plan (10K $\rightarrow$ 1M concurrency), zero-trust security architecture, and multi-year FinOps TCO model engineered for a Tier-1 industrial conglomerate.

---

## 📺 Video Walkthrough & Architecture Demo

[![Watch the Video Walkthrough](https://img.youtube.com/vi/2vUcTsUDV38/maxresdefault.jpg)](https://youtu.be/2vUcTsUDV38)

> 💡 *Click the thumbnail above to watch the end-to-end system architecture walkthrough, control plane live demo, and GPU capacity planning derivation on YouTube.*

---

## 🏛️ Executive Summary & Case Study Context

* **Client Profile**: Fictional Tier-1 Industrial Conglomerate (Steel, Energy, Infrastructure, Consumer Services) partnering with Frontier Model Providers.
* **Workload Specifications**:
  * **Scale**: 10,000,000 registered users, 1,000,000 MAU, 100,000 DAU, 10,000 peak concurrent streams (scaling toward 1,000,000).
  * **SLA Targets**: 99.9%+ availability, $\sim 1$s P95 Time-To-First-Token (TTFT), and $\sim 3–5$s P95 end-to-end response latency.
  * **Core Mandate**: Eliminate siloed, shadow AI tools; establish an air-gapped private inference platform; secure enterprise knowledge with document-level ACLs; and avoid multi-hundred crore capital misallocation.
* **Engagement Outcome**:
  * 💰 **₹70–90 Cr Annual Infrastructure Savings** via dynamic model routing, continuous batching, and FP8 quantization.
  * 🚀 **₹120–150 Cr Annualized Platform Revenue** through enterprise subscription assistants and private API cross-charging.
  * 🛡️ **100% Air-Gapped Data Isolation** with zero raw enterprise credential exposure to models.

---

## 🗺️ System Architecture Topology

```text
[ Enterprise Clients: Mobile / Web / ERP / IoT Telemetry ]
                           │ (Mutual TLS / OIDC Auth)
                           ▼
     ┌─────────────────────────────────────────────────────────┐
     │           API Gateway & Tenant Rate Limiter             │
     └─────────────────────────┬───────────────────────────────┘
                               │
                               ▼
     ┌─────────────────────────────────────────────────────────┐
     │       Unified /v1/chat Control Plane & Bot Registry     │
     │  - Threat Classifier & Injection Interceptor            │
     │  - Dynamic Model Router (Complexity Scoring)            │
     └─────────────┬─────────────────────────────┬─────────────┘
                   │                             │
                   ▼                             ▼
  ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
  │   Tier 1: 8B SLM Cluster (FP8)  │   │ Tier 2: Claude-X 70B (FP8 TP=2) │
  │   - 70% of Enterprise Queries   │   │ - 30% Deep Reasoning & Code     │
  │   - TTFT: ~280ms | $0.0002/1k   │   │ - TTFT: ~890ms | $0.0035/1k     │
  └─────────────────────────────────┘   └─────────────────────────────────┘
                   │                             │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
     ┌─────────────────────────────────────────────────────────┐
     │      Hybrid Retrieval & Air-Gapped Tool Plane           │
     │  - BM25 + Dense BGE Vector Search (Reciprocal Rank RRF) │
     │  - Document-Level ACLs & Tenant Vector Namespaces       │
     │  - Zero-Trust Tool Gateway + Human-in-the-Loop Gates    │
     └─────────────────────────────────────────────────────────┘
```

---

## 📊 Key Empirical Receipts & Sizing Metrics

### 1. Hardware Capacity Sizing (70B Model @ FP8 Precision)
* **Weights VRAM**: $68.45\text{ GB}$ (allocated across $2\times\text{H100 80GB}$ in Tensor Parallelism TP=2).
* **KV-Cache Footprint**: $160\text{ KB/token} \times 4,096\text{ tokens} = 640.0\text{ MB per active stream}$.
* **10,000 Concurrency Cluster**: **162 H100 GPUs (21 8x HGX Nodes)** providing $350,000\text{ peak tokens/sec}$.
* **Blended Routing Optimization**: Routing 70% of queries to 8B SLMs reduces hardware requirements to **68 mixed GPUs (58% compute cost reduction)**.

### 2. SLA Latency Distribution

| Metric | P50 (Median) | P95 (Target SLA) | P99 (Tail Latency) | SLA Target | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Time-To-First-Token (TTFT)** | $245.0\text{ ms}$ | **$880.0\text{ ms}$** | $1,240.0\text{ ms}$ | $\le 1,000\text{ ms}$ | ✅ **Compliant** |
| **Inter-Token Latency (ITL)** | $22.0\text{ ms}$ | **$28.2\text{ ms}$** | $35.2\text{ ms}$ | $\le 40\text{ ms}$ | ✅ **Compliant** |
| **End-to-End Latency (512 tokens)** | $2.15\text{ s}$ | **$3.48\text{ s}$** | $4.85\text{ s}$ | $\le 5.0\text{ s}$ | ✅ **Compliant** |

---

## 💰 FinOps: 3-Year Total Cost of Ownership (TCO) Comparison

```
Option A: Pretrain From Scratch  | [███████████████████████████████████] ₹750.0 Cr
Option B: Public SaaS Token APIs | [███████████████████████]             ₹472.0 Cr
Option C: Private Routed (Ours)  | [█████]                               ₹108.0 Cr
```

* **Annual Infrastructure Cost Avoidance**: **₹70–90 Crores / year**.
* **Annualized AI Platform Revenue**: **₹120–150 Crores / year**.
* **ROI Multiple on ₹4.8 Cr Architecture Engagement**: **$113.4\times$ Net Economic Return**.

---

## 📂 Repository Organization & Technical Deep-Dives

```text
.
├── docs/
│   ├── 01_SYSTEM_ARCHITECTURE.md     # Control plane, /v1/chat, Bot Registry, and SSE streaming
│   ├── 02_HARDWARE_CAPACITY_PLAN.md  # Mathematical formulas for VRAM, KV-cache, and cluster sizing
│   ├── 03_RAG_AND_TOOL_GOVERNANCE.md # Hybrid RAG (BM25+Dense RRF), Document ACLs, Air-gapped tools
│   ├── 04_SECURITY_THREAT_MODEL.md   # Prompt injection, PII/DLP, Untrusted context quarantine
│   └── 05_FINOPS_TCO_ROI_MODEL.md    # Comparative TCO & ROI financial modeling
├── src/
│   ├── sizing/                       # Hardware capacity calculator & latency simulator
│   ├── control_plane/                # FastAPI multi-tenant gateway, router & guardrails
│   ├── finops/                       # 3-Year Capex/Opex FinOps simulation engine
│   └── cli.py                        # Interactive CLI tool for sizing, latency & TCO
├── tests/                            # 100% passing Pytest test suite
└── results/                          # Ground-truth JSON benchmark receipts
```

---

## ⚡ Quickstart & Interactive CLI

### 1. Run with Docker (1-Click Local Serving)
```bash
# Start the Gateway with Docker Compose
docker compose up --build -d

# Check Gateway Health & Swagger UI
curl http://localhost:8000/health
open http://localhost:8000/docs
```

### 2. Manual Local Setup
```bash
pip install -r requirements.txt
uvicorn src.control_plane.gateway:app --reload --port 8000
```

### 3. Run GPU Capacity Sizing Calculator
```bash
python src/cli.py size-cluster --model Claude-X-Frontier-70B --gpu NVIDIA-H100-SXM-80GB --concurrency 10000 --precision fp8
```

### 4. Run FinOps 3-Year TCO Simulator
```bash
python src/cli.py run-tco --fee-cr 4.80
```

### 5. ⚡ Single-Line Admin & FinOps Control Commands

#### A. Instant Tenant Shutdown (Emergency Kill-Switch & Budget Quarantine)
```bash
# Instantly SHUT DOWN a subsidiary tenant (blocks inference traffic immediately)
curl -X POST http://localhost:8000/v1/tenants/tenant_steel_manufacturing/lifecycle \
  -H "Authorization: Bearer mock-jwt-token" -H "X-Tenant-ID: tenant_steel_manufacturing" -H "X-User-Role: admin" \
  -H "Content-Type: application/json" -d '{"action": "SHUTDOWN", "reason": "Monthly budget cap exceeded"}'

# Re-activate a suspended/shutdown tenant
curl -X POST http://localhost:8000/v1/tenants/tenant_steel_manufacturing/lifecycle \
  -H "Authorization: Bearer mock-jwt-token" -H "X-Tenant-ID: tenant_steel_manufacturing" -H "X-User-Role: admin" \
  -H "Content-Type: application/json" -d '{"action": "ACTIVATE"}'
```

#### B. Real-Time Tenant Spend & Budget Metering
```bash
# Check accumulated spend and quota consumption for a specific tenant
curl http://localhost:8000/v1/tenants/tenant_steel_manufacturing/cost \
  -H "Authorization: Bearer mock-jwt-token" -H "X-Tenant-ID: tenant_steel_manufacturing"
```

#### C. Cloud Infrastructure Scale-Down & Teardown (IaaS)
```bash
# Scale down GPU worker nodes to 0 replicas (stop GPU cloud burning during off-hours)
kubectl scale deployment/vllm-slm-serving --replicas=0

# Scale down Gateway replicas
kubectl scale deployment/enterprise-genai-gateway --replicas=1

# Complete cloud environment teardown (Terraform destroy)
cd infra/terraform && terraform destroy -auto-approve
```

### 6. Run Full Test Suite
```bash
pytest tests/ -v
```

---

## 🎯 Interview Placement: Staff Architect vs. Lead FDE

* **When Interviewing for GenAI / ML Systems Roles**: Emphasize first-principles KV-cache derivations, continuous batching, Tensor Parallelism topologies, and hybrid vector RRF retrieval.
* **When Interviewing for Staff FDE / Solutions Architect Roles**: Emphasize multi-tenant enterprise control planes, zero-trust IAM boundaries, human-in-the-loop tool gates, and ₹70–90 Cr annual FinOps cost avoidance.
