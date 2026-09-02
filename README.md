# Generative AI & LLM Systems Engineering Portfolio

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers%20%26%20PEFT-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20REST%20%26%20SSE-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg?logo=githubactions&logoColor=white)](https://github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Production-grade applied Generative AI, LLM infrastructure, and scientific benchmarking portfolio** engineered for mid-to-senior GenAI / Machine Learning Systems roles at top-tier technology companies (Google, Meta, Amazon, Apple, Netflix).

> 🎓 **MAANG Interview & Systems Masterclass Resources**:
> * 📘 **[MAANG GenAI & LLM Engineering Interview Guide](MAANG_GENAI_INTERVIEW_GUIDE.md)**: 18+ real-world interview scenarios organized by Role and Seniority (**Junior/L3, Mid/L4, Senior/L5, Staff/L6, Principal/L7**) with runnable code, GPU arithmetic & system blueprints!
> * 📚 **[GenAI & LLM Glossary Cheat Sheet](GENAI_GLOSSARY_CHEATSHEET.md)**: 35+ core terms spanning Level 0 Basics (Tokenization, AdamW, NumPy Cosine Sim) to Level 3 Infrastructure (FlashAttention-3, PagedAttention, GRPO, 3D Parallelism) & Famous Industry Tools!
> * 🧪 **[GenAI Core Algorithms & Evaluation Masterclass Notebook](./notebooks/GenAI_Core_Algorithms_And_Evaluation_Masterclass.ipynb)**: Complete from-scratch live-coding implementations of Cosine/L2, BM25, IVFFlat, HNSW, In-Memory VectorDB with ACLs, BPE, GQA with KV-Cache, RoPE, INT8 Quantization, LoRA, Semantic Model Router, Air-Gapped Tool Calling, and the RAG Triad / LLM evaluation suite!
> * ⚡ **[Nano-Transformer (Micro-LLM From Scratch) Notebook](./notebooks/Nano_Transformer_Micro_LLM_From_Scratch.ipynb)**: Minimal, standalone Causal Transformer (~58K parameters) built from first principles that trains on CPU in **< 15 seconds**, featuring live loss curves, Perplexity tracking, generation with KV-caching, and attention heatmaps!

---

## 📺 Systems Architecture & Live Demo Walkthrough

[![Watch the Systems Portfolio Walkthrough](https://img.youtube.com/vi/YOUR_YOUTUBE_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=YOUR_YOUTUBE_VIDEO_ID)

> 💡 *Click the thumbnail above to watch the video walkthrough demonstrating the Enterprise AI Platform control plane, Quantized Agentic workflow, Legal LoRA fine-tuning, and Quantum Simulator.*

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
| **[1. Enterprise GenAI Platform & Capacity Architecture](./portfolio/Enterprise-GenAI-Platform-Capacity-Architecture)** | **Staff AI Architecture & Lead FDE Systems** | Multi-Tenant `/v1/chat` Gateway, Bot Registry, Dynamic SLM/Frontier Router, GPU Capacity Math (10K→1M Concurrency), FinOps TCO | 💰 **₹70–90 Cr/yr Infra Savings**<br>🚀 **₹120–150 Cr Platform Run-Rate**<br>⚡ **<1s P95 TTFT, <3.5s E2E** | [![Code](https://img.shields.io/badge/Architecture-Control%20Plane-009688)](./portfolio/Enterprise-GenAI-Platform-Capacity-Architecture) [![Docs](https://img.shields.io/badge/Docs-Capacity%20Plan-blue)](./portfolio/Enterprise-GenAI-Platform-Capacity-Architecture/docs) |
| **[2. Customer Support Quantized LLM Agent](https://github.com/aashiq-parinda/Customer-Support-Automation-Quantized-LLM-Agentic-Workflow)** | **LLM Optimization & Cost Engineering** | 4-bit NF4 Quantization, bitsandbytes, Hybrid Search (BM25 + Dense RRF), FastAPI SSE | ⚡ **96.67% Cost Reduction**<br>💾 **65% VRAM Reduction** (14.5GB → 5.1GB)<br>🛡️ **0.9412 Guardrail F1** | [![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/aashiq-parinda/Customer-Support-Automation-Quantized-LLM-Agentic-Workflow) [![Space](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/ashrafksalim/customer-support-quantized-llm-agent) [![Colab](https://img.shields.io/badge/Colab-Notebook-orange)](https://colab.research.google.com/drive/1RIf9_bZAoqmB9rq6nNWciNmaSPZJSpOx?usp=sharing) |
| **[3. Contract Risk Review & Reasoning](https://github.com/aashiq-parinda/LLM-Fine-Tuning-Agentic-Reasoning-for-Contract-Risk-Review)** | **Domain Fine-Tuning & Legal AI** | LoRA / PEFT Adapters, DeBERTa-v3 / Qwen2.5, LEDGAR Dataset, ChromaDB RAG | 🎯 **High-precision Clause Risk Extraction**<br>🔍 **Precedent Case Grounding**<br>⚖️ **Fairness & Bias Audited (1.000 F1)** | [![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/aashiq-parinda/LLM-Fine-Tuning-Agentic-Reasoning-for-Contract-Risk-Review) [![YouTube](https://img.shields.io/badge/YouTube-Video%20Demo-red?logo=youtube)](https://youtu.be/-iob4zi4bfY) |
---

## 📓 Interactive Live-Coding & Systems Notebooks

| Interactive Notebook | Focus & Algorithms | Complexity & Stack | Key Live Capabilities |
| :--- | :--- | :--- | :--- |
| **[🧪 GenAI Core Algorithms & Evaluation Masterclass](./notebooks/GenAI_Core_Algorithms_And_Evaluation_Masterclass.ipynb)** | **RAG, Vector Search, Transformer Internals & Eval** | Pure Python + NumPy | • **Distance Foundations**: Vectorized Cosine Sim ($\epsilon$-stable), Euclidean L2, Angular Distance.<br>• **Search & Indexing**: BM25 Okapi, IVFFlat (K-Means), HNSW graph search, RRF.<br>• **VectorDB**: In-Memory VectorDB with multi-tenant ACL clearance filtering.<br>• **Transformer Mechanics**: BPE Tokenizer, Scaled Dot-Product MHA, GQA with KV-Cache, RoPE.<br>• **Optimization**: Symmetric INT8 Quantizer, LoRA Layer ($W_0 + \frac{\alpha}{r}BA$).<br>• **Agentic Infra**: Semantic Model Router (SLM vs Frontier), Air-Gapped Tool Calling & RBAC write gate.<br>• **Evaluation Suite**: RAG Triad (Faithfulness, Relevance), Perplexity, ROUGE-L, BLEU-4, Token F1. |
| **[⚡ Nano-Transformer (Micro-LLM From Scratch)](./notebooks/Nano_Transformer_Micro_LLM_From_Scratch.ipynb)** | **End-to-End Causal Language Model** | ~58K Params, Pure NumPy | • **Zero Black Boxes**: Tokenizer, Positional Embeddings, Causal Attention, LayerNorm, SwiGLU FFN, LM Head.<br>• **Instant CPU Training**: Complete training loop in **< 15 seconds** with live Loss & Perplexity tracking.<br>• **Generation**: Autoregressive decoding with Temperature & Top-$K$ filtering.<br>• **Interpretability**: Real-time Attention Heatmap visualization. |

---

## 🏛️ Deep Dive: Project Highlights

### 1. Enterprise Multi-Tenant GenAI Platform & Capacity Architecture
* **Repository**: [`Enterprise-GenAI-Platform-Capacity-Architecture`](./portfolio/Enterprise-GenAI-Platform-Capacity-Architecture)
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
* **Repository**: [`Customer-Support-Automation-Quantized-LLM-Agentic-Workflow`](https://github.com/aashiq-parinda/Customer-Support-Automation-Quantized-LLM-Agentic-Workflow)
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
* **Repository**: [`LLM-Fine-Tuning-Agentic-Reasoning-for-Contract-Risk-Review`](https://github.com/aashiq-parinda/LLM-Fine-Tuning-Agentic-Reasoning-for-Contract-Risk-Review)
* **Problem**: Generic LLMs hallucinate and miss non-standard liability, indemnification, and governing law risks in enterprise legal contracts.
* **Architecture & Solution**:
  - **Hybrid Encoder-Decoder Architecture**: Uses parameter-efficient fine-tuning (LoRA on DeBERTa-v3/Qwen) for deterministic clause classification trained on the LEDGAR corpus.
  - **Precedent Knowledge Retrieval**: Vector database embedding retrieval grounding risk determinations against statutory definitions and contract playbooks.
  - **Fairness & Bias Auditing**: Rigorous evaluation across contract jurisdictions and party asymmetries.

---

### 3. Quantum Hardware Validation & Benchmarking Suite
* **Repository**: [`Quantum-Hardware-Validation-Benchmarking-Suite`](https://github.com/aashiq-parinda/Quantum-Hardware-Validation-Benchmarking-Suite)
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

To clone and run any repository:

```bash
# Clone the repository of your choice
git clone https://github.com/aashiq-parinda/Customer-Support-Automation-Quantized-LLM-Agentic-Workflow.git
cd Customer-Support-Automation-Quantized-LLM-Agentic-Workflow

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
- **GitHub**: [@aashiq-parinda](https://github.com/aashiq-parinda)  
- **Hugging Face**: [@ashrafksalim](https://huggingface.co/ashrafksalim)  

---

## 📜 License

All projects in this portfolio are distributed under the [MIT License](LICENSE).
