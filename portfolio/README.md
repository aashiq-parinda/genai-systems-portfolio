# Generative AI & LLM Systems Engineering Portfolio

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers%20%26%20PEFT-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20REST%20%26%20SSE-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg?logo=githubactions&logoColor=white)](https://github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Production-grade applied Generative AI, LLM infrastructure, and scientific benchmarking portfolio** engineered for mid-to-senior GenAI / Machine Learning Systems roles at top-tier technology companies (Google, Meta, Amazon, Apple, Netflix).

---

## 🎯 Engineering Focus & Standards

Unlike typical bootcamp notebooks or basic UI wrappers, every repository in this portfolio is engineered against **MAANG production standards**:

1. **Modular Python Architecture**: Cleanly separated `src/core/` (domain logic/DAGs), `src/api/` (FastAPI streaming endpoints), `src/eval/` (evaluation harness), and `tests/` (comprehensive Pytest coverage).
2. **Empirical Benchmark Receipts**: Verified quantitative receipts backed by executable evaluation scripts (`results/benchmark_metrics.json`) measuring latency percentiles (p50/p90/p95/p99), VRAM footprints, throughput, and financial cost-per-ticket.
3. **Deterministic Safety Guardrails**: Active defense against adversarial prompt injections (DAN mode, system prompt leaks) and risk routing for high-stakes financial/legal workflows.
4. **CI/CD Automation**: Multi-version Python GitHub Actions workflows enforcing linting, unit testing, and benchmark regressions.

---

## 📂 Featured Portfolio Repositories

| Repository / Project | Focus Area | Core Technologies | Key Empirical Receipts | Live Demos & Code |
| :--- | :--- | :--- | :--- | :--- |
| **[1. Enterprise GenAI Platform & Capacity Architecture](./Enterprise-GenAI-Platform-Capacity-Architecture)** | **Staff AI Architecture & Lead FDE Systems** | Multi-Tenant `/v1/chat` Gateway, Bot Registry, Dynamic SLM/Frontier Router, GPU Capacity Math (10K→1M Concurrency), FinOps TCO | 💰 **₹70–90 Cr/yr Infra Savings**<br>🚀 **₹120–150 Cr Platform Run-Rate**<br>⚡ **<1s P95 TTFT, <3.5s E2E** | [![Code](https://img.shields.io/badge/Architecture-Control%20Plane-009688)](./Enterprise-GenAI-Platform-Capacity-Architecture) [![Docs](https://img.shields.io/badge/Docs-Capacity%20Plan-blue)](./Enterprise-GenAI-Platform-Capacity-Architecture/docs) |
| **[2. Customer Support Quantized LLM Agent](./Customer-Support-Automation-Quantized-LLM-Agentic-Workflow)** | **LLM Optimization & Cost Engineering** | 4-bit NF4 Quantization, bitsandbytes, Hybrid Search (BM25 + Dense RRF), FastAPI SSE | ⚡ **96.67% Cost Reduction**<br>💾 **65% VRAM Reduction** (14.5GB → 5.1GB)<br>🛡️ **0.9412 Guardrail F1** | [![Space](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/ashrafksalim/customer-support-quantized-llm-agent) [![Colab](https://img.shields.io/badge/Colab-Interactive%20Notebook-orange)](https://colab.research.google.com/drive/1RIf9_bZAoqmB9rq6nNWciNmaSPZJSpOx?usp=sharing) |
| **[3. Contract Risk Review & Reasoning](./LLM-Fine-Tuning-Agentic-Reasoning-for-Contract-Risk-Review)** | **Domain Fine-Tuning & Legal AI** | LoRA / PEFT Adapters, DeBERTa-v3 / Qwen2.5, LEDGAR Dataset, ChromaDB RAG | 🎯 **High-precision Clause Risk Extraction**<br>🔍 **Precedent Case Grounding**<br>⚖️ **Fairness & Bias Audited** | [![Notebook](https://img.shields.io/badge/Code-LoRA%20Pipeline-green)](./LLM-Fine-Tuning-Agentic-Reasoning-for-Contract-Risk-Review) |
| **[4. Quantum Hardware Benchmarking Suite](./Quantum-Hardware-Validation-Benchmarking-Suite)** | **Scientific Computing & Systems Eval** | Quantum Circuit Simulation, Hardware Noise Modeling, Error Mitigation | 🔬 **Hardware-Agnostic Validation**<br>📊 **Fidelity & Error Benchmarking**<br>📈 **Algorithmic Profiling** | [![Notebook](https://img.shields.io/badge/Code-Benchmarking%20Suite-purple)](./Quantum-Hardware-Validation-Benchmarking-Suite) |

---

## 🏛️ Deep Dive: Project Highlights

### 1. Enterprise Multi-Tenant GenAI Platform & Capacity Architecture
* **Repository**: [`Enterprise-GenAI-Platform-Capacity-Architecture`](./Enterprise-GenAI-Platform-Capacity-Architecture)
* **Problem**: A Tier-1 industrial conglomerate with 10M registered users and 100K DAU faced runaway API expenses, shadow AI tools across subsidiaries, and data leakage risks, risking a ₹500+ Cr misallocation in scratch pretraining.
* **Architecture & Solution**:
  - **Unified `/v1/chat` Control Plane & Bot Registry**: Centralized metadata catalog managing 50+ enterprise bots with document-level ACLs and RBAC/ABAC isolation.
  - **Dynamic Model-Routing Engine**: Routes 70% of low-complexity queries to 8B SLMs and 30% complex analytical reasoning to Claude-X / 70B frontier models.
  - **First-Principles GPU Capacity Plan**: Mathematical derivation of VRAM, KV-cache, continuous batching, and Tensor Parallelism (TP=2) for 10K $\rightarrow$ 1M concurrency ($<1$s P95 TTFT, $<3.5$s P95 E2E).
  - **Zero-Trust Tool Plane**: Air-gapped ERP/CRM tool execution with transactional Human-in-the-Loop gates for state modifications.
* **Empirical Receipts**:
  - **₹70–90 Cr Annual Infrastructure Cost Avoidance** modeled via private inference + routing.
  - **₹120–150 Cr Annualized Platform Revenue** unlocked across enterprise subscriptions.
  - **113.4x ROI** delivered on architecture engagement fee.

```
[ Enterprise Request ] ──► [ WAF / IAM Gateway ] ──► [ Injection & PII Firewall ]
                                                            │
                                                            ▼
                                             [ Dynamic Model Router ]
                                              │                    │
                          (70% Low Complexity)▼                    ▼ (30% Deep Reasoning)
                              [ 8B SLM Cluster ]             [ Claude-X 70B Cluster (TP=2) ]
                                      │                              │
                                      └──────────────┬───────────────┘
                                                     ▼
                                      [ Hybrid RAG + Air-Gapped Tool Plane ]
```

---

### 2. Customer Support Automation — Quantized LLM + Agentic Workflow
* **Repository**: [`Customer-Support-Automation-Quantized-LLM-Agentic-Workflow`](./Customer-Support-Automation-Quantized-LLM-Agentic-Workflow)
* **Problem**: High inference costs and latency SLAs when deploying 7B+ LLMs for high-volume tier-1 customer tickets, coupled with safety risks on billing disputes and prompt injection.
* **Architecture & Solution**:
  - **4-Bit NormalFloat4 (NF4)** quantization reducing memory footprint from 14.54 GB to 5.09 GB, enabling deployment on low-cost T4/L4 GPUs.
  - **Multi-Tier Safety Guardrails**: Real-time threat catalog intercepting adversarial jailbreaks and legal/fraud keywords (`chargeback`, `subpoena`, `lawsuit`, `GDPR`).
  - **Enterprise Hybrid Search RAG**: Combining BM25 keyword matching with Dense Vector Cosine Similarity fused via Reciprocal Rank Fusion (RRF).
  - **FastAPI REST & SSE Streaming**: Server-Sent Events token streaming endpoint for ultra-low Time-To-First-Token (TTFT).
* **Empirical Receipts**:
  - Cost per 1,000 tickets dropped from **$0.8921 → $0.0297 (96.67% reduction)**.
  - Mean latency reduced by **7.68x** with **0.9412 Guardrail F1 score**.

```
[ Incoming Support Query ] ──► [ Stage 1: Safety & Injection Scan ] ──► (High Risk) ──► [ Human Escalation Queue ]
                                              │ (Safe)
                                              ▼
                               [ Stage 2: Intent Classification ]
                                              │
                                              ▼
                               [ Stage 3: Hybrid RAG (BM25 + RRF) ]
                                              │
                                              ▼
                               [ Stage 4: 4-Bit Quantized LLM ] ──► [ SSE Streamed Response ]
```

---

### 2. Contract Risk Review — LoRA Fine-Tuning & Multi-Agent Legal Reasoning
* **Repository**: [`LLM-Fine-Tuning-Agentic-Reasoning-for-Contract-Risk-Review`](./LLM-Fine-Tuning-Agentic-Reasoning-for-Contract-Risk-Review)
* **Problem**: Generic LLMs hallucinate and miss non-standard liability, indemnification, and governing law risks in enterprise legal contracts.
* **Architecture & Solution**:
  - **Hybrid Encoder-Decoder Architecture**: Uses parameter-efficient fine-tuning (LoRA on DeBERTa-v3/Qwen) for deterministic clause classification trained on the LEDGAR corpus.
  - **Precedent Knowledge Retrieval**: Vector database embedding retrieval grounding risk determinations against statutory definitions and contract playbooks.
  - **Fairness & Bias Auditing**: Rigorous evaluation across contract jurisdictions and party asymmetries.

---

### 3. Quantum Hardware Validation & Benchmarking Suite
* **Repository**: [`Quantum-Hardware-Validation-Benchmarking-Suite`](./Quantum-Hardware-Validation-Benchmarking-Suite)
* **Problem**: Evaluating noisy intermediate-scale quantum (NISQ) circuits and algorithmic fidelity under realistic hardware error models.
* **Architecture & Solution**:
  - **Hardware-Agnostic Benchmarking Engine**: Profiles quantum circuit depth, gate counts, and decoherence times ($T_1, T_2$).
  - **Noise Simulation & Error Mitigation**: Evaluates zero-noise extrapolation (ZNE) and readout error mitigation strategies.
  - **Empirical Visualizations**: Publishes automated fidelity decay and benchmarking curves.

---

## 🛠️ Common Architectural Patterns & Quality Standards

Each repository across this portfolio adheres to unified engineering standards:

```text
repository-name/
├── .github/workflows/
│   └── ci.yml               # Multi-version Python CI (lint, test, benchmark)
├── src/
│   ├── core/                # Core algorithms, pipelines, and state DAGs
│   ├── api/                 # Production FastAPI REST and streaming servers
│   └── eval/                # Standalone reproducible evaluation harnesses
├── tests/                   # 100% passing Pytest unit & integration suites
├── results/                 # Ground-truth JSON metrics and markdown receipts
├── requirements.txt         # Pinned production dependencies
└── README.md                # Comprehensive documentation with CI badges & tables
```

---

## ⚡ Global Setup & Running Tests

To run tests across any repository:

```bash
# Clone the portfolio
git clone https://github.com/ashrafksalim/customer-support-quantized-llm-agent.git
cd customer-support-quantized-llm-agent

# Install dependencies
pip install -r requirements.txt

# Run full test suite
pytest tests/ -v

# Execute reproducible benchmark harness
python -m src.eval.benchmark --iterations 5 --output results/
```

---

## 👤 Author & Contact

**Khan Ashraf Salim**  
*Machine Learning & Generative AI Systems Engineer*  
- **Email**: [ashrafk.salim1@gmail.com](mailto:ashrafk.salim1@gmail.com)  
- **GitHub**: [@ashrafksalim](https://github.com/ashrafksalim)  
- **Hugging Face**: [@ashrafksalim](https://huggingface.co/ashrafksalim)  

---

## 📜 License

All projects in this portfolio are distributed under the [MIT License](LICENSE).
