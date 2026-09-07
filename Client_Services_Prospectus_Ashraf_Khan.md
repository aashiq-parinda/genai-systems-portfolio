# 🤖 Generative AI & LLM Systems Engineering
## Client Services Prospectus & Technical Rate Card

**Ashraf Khan**  
*Senior Machine Learning & Generative AI Systems Engineer*  
📍 Mumbai, India (Global Remote Delivery)  
✉️ **Email:** [ashrafk.salim1@gmail.com](mailto:ashrafk.salim1@gmail.com)  
📱 **Direct Phone / WhatsApp:** +91-8779559898  
🌐 **Code Portfolio:** [github.com/aashiq-parinda](https://github.com/aashiq-parinda) | 🤗 **HuggingFace:** [huggingface.co/ashrafksalim](https://huggingface.co/ashrafksalim)  
🔗 **LinkedIn:** [linkedin.com/in/ashrafksalim](https://linkedin.com/in/ashrafksalim)

---

## 🎯 Executive Summary: Why Work With Me?

Most "AI engineers" deliver unversioned Jupyter notebooks, basic API wrapper scripts, and hallucination-prone chatbots that fail under production traffic, adversarial inputs, and real-world cost constraints.

As a **Senior ML & Generative AI Systems Engineer with 5+ years of enterprise software engineering experience** and **3 published production-grade GenAI systems with empirically verified benchmark receipts**, I deliver what most cannot:

1. **Production-Grade & Empirically Verified:** Every system ships with reproducible `benchmark_metrics.json` receipts measuring latency percentiles (p50/p90/p95/p99), VRAM footprints, throughput, cost-per-ticket, and F1 scores — not demo-day screenshots.
2. **Real Cost Engineering:** Proven **96.67% inference cost reduction** (from $0.89 to $0.03 per 1,000 tickets) and **₹70–90 Cr annual infrastructure savings** through quantization, intelligent model routing, and capacity planning.
3. **Adversarial Safety & Guardrails:** Multi-tier defense systems blocking DAN-mode jailbreaks, system prompt leaks, and legal/financial keyword escalation with **0.9412 Guardrail F1**.
4. **Enterprise Architecture at Scale:** First-principles GPU capacity plans scaling from 10K to 1M concurrent users with sub-second P95 TTFT — backed by mathematical VRAM, KV-cache, and Tensor Parallelism derivations.
5. **Full IP Transfer & CI/CD:** All source code, Docker containers, evaluation harnesses, and documentation belong 100% to you, with GitHub Actions CI/CD automation.

---

## 💰 Engagement & Pricing Matrix (Dual Currency)

| Service Domain | Core Scope | US & Global (USD) | Indian Domestic (INR) | Typical Turnaround |
| :--- | :--- | :---: | :---: | :---: |
| **LLM Inference Optimization & Quantization** | 4-bit NF4/GPTQ quantization, VRAM profiling, T4/L4 GPU deployment | **$3,000** | **₹1,20,000** | 5–8 Days |
| **Enterprise RAG Pipeline (Hybrid Search)** | BM25 + Dense RRF fusion, ChromaDB/Pinecone, FastAPI SSE streaming | **$4,500** | **₹1,80,000** | 7–12 Days |
| **LoRA / QLoRA Fine-Tuning Pipeline** | Domain-specific PEFT adapter training, eval harness, HuggingFace PEFT | **$3,500** | **₹1,40,000** | 5–10 Days |
| **Agentic Workflow & Multi-Agent Systems** | DAG orchestration, tool-calling, LangGraph, HITL safety gates | **$4,000** | **₹1,60,000** | 7–12 Days |
| **Safety & Adversarial Guardrail System** | Prompt injection defense, PII/legal routing, red-team evaluation | **$2,500** | **₹1,00,000** | 3–5 Days |
| **FastAPI Streaming Inference Server** | SSE/WebSocket token streaming, async GPU serving, load testing | **$2,000** | **₹80,000** | 3–5 Days |
| **Enterprise GenAI Architecture & Capacity Plan** | Multi-tenant gateway, GPU math, FinOps TCO, 10K→1M scaling blueprint | **$8,000** | **₹3,20,000** | 10–15 Days |
| **Custom ML Evaluation & Benchmarking Harness** | F1/BLEU/ROUGE/Perplexity, latency percentiles, CI/CD integration | **$2,000** | **₹80,000** | 3–5 Days |
| **Production GPU Deployment & MLOps** | Docker, ECS/GKE orchestration, model serving, A/B routing, monitoring | **$5,000** | **₹2,00,000** | 7–14 Days |

---

### Retainer & Advisory Options

- **Hourly Advisory & Troubleshooting:** **$120 USD / hr** | **₹5,000 INR / hr**
- **Part-Time Monthly Retainer (15 hours / week):** **$6,000 USD / month** | **₹2,50,000 INR / month**  
  *(Ideal for startups and enterprises needing a dedicated senior GenAI engineer without the overhead of a full-time hire.)*

---

## 🔬 Service Catalog Breakdown

### 1. LLM Inference Optimization & Cost Engineering
- **4-Bit NormalFloat4 (NF4) Quantization:** Reducing model VRAM footprints by 60–70% (e.g., 14.54 GB → 5.09 GB) using `bitsandbytes` NF4/GPTQ quantization, enabling deployment on low-cost T4/L4 GPUs instead of expensive A100s.
- **VRAM Profiling & Capacity Right-Sizing:** First-principles calculation of model weights, KV-cache, activation memory, and CUDA overhead to determine the minimum viable GPU tier for your concurrency requirements.
- **Latency Optimization:** Token-level streaming via Server-Sent Events (SSE), continuous batching configuration, and KV-cache management to achieve sub-second Time-To-First-Token (TTFT) at production scale.
- **Cost-Per-Request Engineering:** End-to-end financial modeling comparing API providers (OpenAI, Anthropic, Google) vs. self-hosted quantized inference, with empirical cost-per-1,000-request breakdowns.

### 2. Retrieval-Augmented Generation (RAG) & Knowledge Systems
- **Enterprise Hybrid Search RAG:** Combining sparse keyword retrieval (BM25 / Okapi) with dense vector cosine similarity, fused via Reciprocal Rank Fusion (RRF) for maximum recall with minimal hallucination.
- **Vector Database Engineering:** ChromaDB, Pinecone, Weaviate, or pgvector setup with optimized chunking strategies, embedding model selection (OpenAI `text-embedding-3-large`, `bge-large`, `e5-mistral`), and multi-tenant namespace isolation.
- **Agentic RAG with Self-Reflection:** Building self-corrective RAG loops that detect low-confidence retrievals, reformulate queries, and re-retrieve before generating — eliminating silent hallucination failures.
- **Domain Knowledge Ingestion:** Processing and indexing enterprise documents (PDF, DOCX, HTML, Confluence, Notion) with metadata-aware chunking, access control lists (ACLs), and incremental re-indexing pipelines.

### 3. Fine-Tuning & Domain Adaptation (LoRA / QLoRA / PEFT)
- **Parameter-Efficient Fine-Tuning (PEFT):** LoRA and QLoRA adapter training on domain-specific datasets using HuggingFace `transformers`, `trl`, and `peft` libraries — achieving domain expertise without full-model retraining costs.
- **Supported Model Architectures:** Llama 3.x, Mistral/Mixtral, Qwen2.5, DeBERTa-v3, Phi-3, and other encoder/decoder architectures.
- **Training Data Curation & Formatting:** Converting raw enterprise data (contracts, support tickets, medical records, financial filings) into instruction-tuning datasets with proper chat templates, system prompts, and evaluation splits.
- **Rigorous Evaluation & Bias Auditing:** Automated evaluation harnesses measuring task-specific metrics (classification F1, extraction precision, generation quality) with fairness audits across demographic and domain slices.

### 4. Agentic AI & Multi-Agent Orchestration
- **Multi-Stage DAG Pipelines:** Designing directed acyclic graph (DAG) workflows where each stage performs a distinct function — safety scanning, intent classification, retrieval, generation, and post-processing — with conditional routing and error recovery.
- **Tool-Calling & Function Execution:** Structured tool interfaces with Pydantic v2 schemas, air-gapped execution sandboxes, and transactional Human-in-the-Loop (HITL) gates for state-modifying operations (ERP writes, database mutations, financial transactions).
- **LangGraph & State Machines:** Building stateful, cyclic agent graphs using LangGraph with persistent checkpointing, supervisor topologies, and specialized worker agents (Researcher, Architect, Developer, Reviewer, QA).
- **Model Context Protocol (MCP):** Implementing local MCP JSON-RPC servers for standardized tool discovery and execution across heterogeneous AI agent frameworks.

### 5. Safety, Guardrails & Adversarial Defense
- **Multi-Tier Threat Catalogs:** Real-time interception of adversarial prompt injections (DAN mode, role-play jailbreaks, system prompt extraction), with configurable keyword-based escalation for legal/financial/HIPAA-sensitive content.
- **PII Detection & Redaction:** Automated personally identifiable information scanning and redaction in both user inputs and model outputs, ensuring compliance with GDPR, CCPA, and HIPAA regulations.
- **Red-Team Evaluation Suites:** Systematic adversarial testing frameworks that probe guardrail coverage across known attack vectors, with F1/precision/recall scoring and regression tracking.
- **Human Escalation Routing:** Intelligent escalation DAGs that route high-risk queries (billing disputes, legal threats, medical advice) to human agents with full conversation context preservation.

### 6. Enterprise Architecture, MLOps & GPU Capacity Planning
- **Multi-Tenant GenAI Platform Design:** Unified `/v1/chat` control plane with bot registry, dynamic SLM/Frontier model routing (70/30 split), RBAC/ABAC isolation, and document-level access control.
- **First-Principles GPU Capacity Plans:** Mathematical derivation of VRAM requirements, KV-cache sizing, continuous batching throughput, and Tensor Parallelism (TP=2/4/8) configurations for scaling from 10K to 1M+ concurrent users.
- **FinOps & TCO Modeling:** Total Cost of Ownership analysis comparing cloud GPU providers (AWS, GCP, Azure, Lambda Labs, CoreWeave), reserved vs. on-demand pricing, and infrastructure cost avoidance projections.
- **Production MLOps:** Docker containerization, Kubernetes/ECS orchestration, model versioning, A/B deployment strategies, Prometheus/Grafana monitoring, and automated rollback pipelines.

---

## 📦 What You Receive in Every Delivery

1. **Source Code & Scripts:** Clean, modular, linted (`ruff`), type-annotated Python source code following `src/core/`, `src/api/`, `src/eval/`, `tests/` architecture.
2. **Containerized Environment:** Dockerfile and `docker-compose.yml` or pinned `requirements.txt` ensuring 100% reproducibility across environments.
3. **Empirical Benchmark Receipts:** Verified `benchmark_metrics.json` with latency percentiles, VRAM profiling, throughput metrics, and cost calculations — not just "it works" screenshots.
4. **CI/CD Pipeline:** GitHub Actions workflow (`.github/workflows/ci.yml`) running multi-version Python linting, unit tests, and benchmark regressions on every commit.
5. **Comprehensive Documentation:** README with architecture diagrams, API reference, setup instructions, and inline code documentation.
6. **Handover Walkthrough Call:** 30–45 minute screen-share session to review architecture, demonstrate the system, discuss trade-offs, and ensure smooth knowledge transfer.

---

## 🛡️ Security, Privacy & Data Governance

- **Strict Non-Disclosure Agreement (NDA):** Signed prior to receiving any proprietary data, models, or project details.
- **Zero-Breach Healthcare Posture:** As an experienced HIPAA data custodian (former healthcare platform engineer at Fitwell Technologies), your data is processed exclusively on encrypted, isolated environments and never used to train public commercial AI models.
- **Intellectual Property Guarantee:** 100% of generated source code, trained adapters, evaluation data, and derived intellectual property belong to you upon project sign-off.
- **Secure Compute:** All development performed on encrypted workstations with ephemeral cloud instances (AWS/GCP) destroyed after project completion.

---

## 🚀 How to Get Started

```
  STEP 1                 STEP 2                 STEP 3                 STEP 4
┌──────────────┐       ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│ Email Scope  │──────►│ 30-Min Call  │──────►│ Proposal/SOW │──────►│ Execution &  │
│ & Use Case   │       │ Alignment    │       │ Fixed Price  │       │ Delivery     │
└──────────────┘       └──────────────┘       └──────────────┘       └──────────────┘
```

1. **Email your project overview:** Send your use case, dataset/model requirements, and desired outcomes to **ashrafk.salim1@gmail.com**.
2. **Introductory Scoping Call (30 mins):** We review technical requirements, infrastructure constraints, and agree on deliverables and milestones.
3. **Formal Statement of Work (SOW):** You receive a clear, fixed-price proposal detailing deliverables, timeline, and terms within 24 hours.
4. **Execution & Delivery:** Real-time milestone updates, clean execution, empirical benchmark receipts, and complete delivery with handover call.

---

*Ready to ship production-grade AI systems? Let's connect:*  
📧 **Email:** [ashrafk.salim1@gmail.com](mailto:ashrafk.salim1@gmail.com) | 📱 **Phone/WhatsApp:** +91-8779559898  
🌐 **GitHub:** [github.com/aashiq-parinda](https://github.com/aashiq-parinda) | 🤗 **HuggingFace:** [huggingface.co/ashrafksalim](https://huggingface.co/ashrafksalim)
