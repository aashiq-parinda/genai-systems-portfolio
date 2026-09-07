# 🤖 Generative AI & LLM Systems Engineering Freelance Services & Rate Card

**Ashraf Khan** | *Senior Machine Learning & Generative AI Systems Engineer*  
📍 Mumbai, India | 🌐 [Portfolio](https://github.com/aashiq-parinda) | 🤗 [HuggingFace](https://huggingface.co/ashrafksalim) | ✉️ [ashrafk.salim1@gmail.com](mailto:ashrafk.salim1@gmail.com) | 📱 +91-8779559898

---

## 💡 Executive Value Proposition

Unlike typical "AI engineers" who deliver unversioned Jupyter notebooks, basic LangChain wrappers, or demo-day chatbots that collapse under production traffic and adversarial inputs, I bring **5+ years of enterprise software engineering, HIPAA-compliant cloud architecture, and production-grade GenAI systems** with verified empirical receipts.

Every deliverable is:
- ✅ **Empirically Benchmarked:** Ships with `benchmark_metrics.json` measuring latency percentiles (p50/p90/p95/p99), VRAM footprints, throughput, and cost-per-request.
- ✅ **Production-Hardened:** Modular `src/core/`, `src/api/`, `src/eval/`, `tests/` architecture with 100% passing Pytest suites and GitHub Actions CI/CD.
- ✅ **Adversarially Tested:** Multi-tier guardrail systems with measured F1 scores against prompt injection, jailbreak, and data exfiltration attacks.
- ✅ **Cost-Engineered:** Proven 96.67% inference cost reduction and ₹70–90 Cr annual infrastructure savings through quantization and intelligent routing.
- ✅ **Confidential & Compliant:** Strict NDA execution, isolated compute environments, and HIPAA/PHI compliance expertise.

---

## 💰 Engagement & Pricing Models Summary

| Engagement Model | US & Global Rate (USD) | Indian Domestic Rate (INR) | Best Suited For |
| :--- | :--- | :--- | :--- |
| **Hourly Consulting / Code Review** | **$100 – $150 / hr** | **₹4,000 – ₹6,000 / hr** | Ad-hoc debugging, architecture review, model selection advisory |
| **Part-Time Retainer (15 hrs/week)** | **$5,500 – $7,000 / month** | **₹2,20,000 – ₹2,80,000 / month** | Ongoing AI engineering support, startup fractional ML engineer |
| **Fixed-Scope Milestone Project** | **$2,000 – $10,000 / project** | **₹80,000 – ₹4,00,000 / project** | Defined end-to-end deliverables with fixed turnaround |

---

## 📦 Detailed Service Catalog & Fixed-Price Tiers

### 1. LLM Inference Optimization & Cost Engineering

#### Service 1.1: 4-Bit Quantization & VRAM Optimization
- **Scope:** Model quantization using NormalFloat4 (NF4) or GPTQ via `bitsandbytes`; VRAM profiling of model weights, KV-cache, and activation memory; GPU tier right-sizing (T4/L4/A10G); comparative benchmarking of quantized vs. full-precision inference quality (perplexity, task-specific F1).
- **Deliverables:**
  - Quantized model checkpoint with verified inference quality metrics.
  - VRAM profiling report with before/after memory footprint comparison.
  - Cost-per-1,000-request financial analysis (self-hosted vs. API providers).
  - Deployment-ready Docker container with optimized inference configuration.
- **Turnaround:** 5–8 Business Days.
- **Pricing:** **$3,000 USD** | **₹1,20,000 INR**

#### Service 1.2: FastAPI Streaming Inference Server
- **Scope:** Production-grade FastAPI async server with Server-Sent Events (SSE) token streaming; configurable generation parameters (temperature, top-k, top-p, repetition penalty); health check endpoints; concurrent request handling with GPU memory management; load testing with Locust/k6.
- **Deliverables:**
  - FastAPI application with SSE streaming endpoint and OpenAPI documentation.
  - Docker container with GPU passthrough configuration.
  - Load test results with throughput and latency percentile reports.
  - Deployment guide for AWS ECS / GCP Cloud Run / bare-metal GPU.
- **Turnaround:** 3–5 Business Days.
- **Pricing:** **$2,000 USD** | **₹80,000 INR**

#### Service 1.3: End-to-End Inference Cost Engineering & Provider Comparison
- **Scope:** Comprehensive financial modeling comparing OpenAI, Anthropic, Google Vertex, AWS Bedrock, and self-hosted quantized inference; token-level cost accounting across input/output token ratios; break-even analysis for self-hosting vs. API at various traffic volumes; monthly/annual TCO projections.
- **Deliverables:**
  - Interactive cost comparison spreadsheet with scenario modeling.
  - Recommendation report with break-even traffic thresholds.
  - Implementation roadmap for the recommended inference strategy.
- **Turnaround:** 3–5 Business Days.
- **Pricing:** **$2,500 USD** | **₹1,00,000 INR**

---

### 2. Retrieval-Augmented Generation (RAG) & Knowledge Systems

#### Service 2.1: Enterprise Hybrid Search RAG Pipeline
- **Scope:** End-to-end RAG system combining BM25 sparse keyword retrieval with dense vector cosine similarity, fused via Reciprocal Rank Fusion (RRF); document ingestion pipeline (PDF, DOCX, HTML, Confluence); chunking strategy optimization (recursive character, semantic, sentence-window); embedding model selection and benchmarking.
- **Deliverables:**
  - Production-ready RAG pipeline with FastAPI endpoint.
  - ChromaDB / Pinecone / pgvector vector store with optimized index configuration.
  - Retrieval quality evaluation report (MRR, NDCG, Recall@K).
  - Document ingestion scripts with metadata extraction.
- **Turnaround:** 7–12 Business Days.
- **Pricing:** **$4,500 USD** | **₹1,80,000 INR**

#### Service 2.2: Agentic RAG with Self-Reflection & Query Reformulation
- **Scope:** Building self-corrective RAG loops that detect low-confidence retrievals via relevance scoring, automatically reformulate queries using chain-of-thought reasoning, and re-retrieve before final generation; grounded response verification against source documents; citation extraction and attribution.
- **Deliverables:**
  - Self-reflective RAG agent with configurable confidence thresholds.
  - RAG Triad evaluation metrics (Faithfulness, Answer Relevance, Context Relevance).
  - Hallucination detection and grounding verification module.
  - Comprehensive test suite with adversarial retrieval failure cases.
- **Turnaround:** 7–10 Business Days.
- **Pricing:** **$3,800 USD** | **₹1,50,000 INR**

#### Service 2.3: Multi-Tenant Knowledge Base with ACL Isolation
- **Scope:** Designing and implementing namespace-isolated vector stores with document-level access control lists (ACLs); RBAC/ABAC authorization middleware; tenant-scoped ingestion and retrieval; audit logging for compliance.
- **Deliverables:**
  - Multi-tenant vector store with namespace isolation.
  - RBAC middleware with JWT/OAuth2 integration.
  - Tenant administration API and CLI.
  - Security audit report with access control verification.
- **Turnaround:** 10–14 Business Days.
- **Pricing:** **$5,500 USD** | **₹2,20,000 INR**

---

### 3. Fine-Tuning & Domain Adaptation (LoRA / QLoRA / PEFT)

#### Service 3.1: LoRA / QLoRA Domain-Specific Fine-Tuning Pipeline
- **Scope:** Parameter-efficient fine-tuning using LoRA/QLoRA adapters on domain-specific datasets; supported architectures include Llama 3.x, Mistral, Qwen2.5, DeBERTa-v3, and Phi-3; training data formatting with proper chat templates; hyperparameter optimization (rank, alpha, target modules); evaluation harness with held-out test sets.
- **Deliverables:**
  - Trained LoRA adapter weights (HuggingFace-compatible `.safetensors`).
  - Training curves (loss, learning rate schedule) and evaluation metrics.
  - Reproducible training script with `Trainer` / `SFTTrainer` configuration.
  - Model card with performance benchmarks and recommended inference settings.
- **Turnaround:** 5–10 Business Days.
- **Pricing:** **$3,500 USD** | **₹1,40,000 INR**

#### Service 3.2: Training Data Curation & Instruction-Tuning Dataset Engineering
- **Scope:** Converting raw enterprise data (contracts, support tickets, medical records, financial filings, internal documentation) into high-quality instruction-tuning datasets; deduplication, quality filtering, and synthetic data augmentation; proper train/validation/test splitting with stratification.
- **Deliverables:**
  - Curated instruction-tuning dataset in JSONL/Parquet format.
  - Data quality report with distribution analysis and edge case documentation.
  - Data processing pipeline scripts for incremental dataset updates.
- **Turnaround:** 5–7 Business Days.
- **Pricing:** **$2,500 USD** | **₹1,00,000 INR**

#### Service 3.3: Fairness, Bias & Safety Evaluation Audit
- **Scope:** Rigorous evaluation of fine-tuned models across demographic slices, domain categories, and edge cases; measuring performance parity, toxicity scores, and refusal rates; generating compliance-ready audit reports.
- **Deliverables:**
  - Fairness evaluation report with per-slice metrics breakdown.
  - Toxicity and safety benchmark results.
  - Remediation recommendations with actionable implementation steps.
- **Turnaround:** 3–5 Business Days.
- **Pricing:** **$2,000 USD** | **₹80,000 INR**

---

### 4. Agentic AI & Multi-Agent Orchestration

#### Service 4.1: Multi-Stage Agentic DAG Pipeline
- **Scope:** Designing and implementing directed acyclic graph (DAG) workflows where each stage performs a distinct function — safety scanning, intent classification, retrieval, generation, and post-processing — with conditional routing, error recovery, and observability tracing.
- **Deliverables:**
  - Production-ready agentic pipeline with configurable stage graph.
  - Pydantic v2 tool schemas with structured input/output validation.
  - Observability integration (OpenTelemetry spans, structured logging).
  - Comprehensive Pytest suite covering happy paths and failure modes.
- **Turnaround:** 7–12 Business Days.
- **Pricing:** **$4,000 USD** | **₹1,60,000 INR**

#### Service 4.2: LangGraph Stateful Multi-Agent System
- **Scope:** Building stateful, cyclic agent graphs using LangGraph with persistent checkpointing; supervisor topologies coordinating specialized worker agents (Researcher, Architect, Developer, Reviewer, QA); Human-in-the-Loop (HITL) production deployment gates and circuit breakers.
- **Deliverables:**
  - LangGraph StateGraph implementation with visualization.
  - Agent topology configuration (supervisor, hierarchical, or peer-to-peer).
  - HITL safety gate middleware for state-modifying operations.
  - Demo application with real-time agent execution trace UI.
- **Turnaround:** 10–14 Business Days.
- **Pricing:** **$5,500 USD** | **₹2,20,000 INR**

#### Service 4.3: Tool-Calling & MCP Integration
- **Scope:** Implementing structured tool interfaces with air-gapped execution sandboxes; Model Context Protocol (MCP) JSON-RPC server for standardized tool discovery; transactional HITL gates for ERP/CRM/database mutations; RBAC write-gate authorization.
- **Deliverables:**
  - MCP-compliant tool server with auto-discovery.
  - Air-gapped tool execution sandbox with resource limits.
  - RBAC authorization middleware for tool access control.
  - Integration test suite with mock external services.
- **Turnaround:** 5–8 Business Days.
- **Pricing:** **$3,000 USD** | **₹1,20,000 INR**

---

### 5. Safety, Guardrails & Adversarial Defense

#### Service 5.1: Multi-Tier Guardrail System
- **Scope:** Designing and deploying real-time threat interception catalogs for prompt injection (DAN mode, role-play jailbreaks, system prompt extraction), legal/financial keyword escalation (`chargeback`, `subpoena`, `GDPR`, `lawsuit`), and PII leakage prevention; configurable severity tiers with human escalation routing.
- **Deliverables:**
  - Guardrail middleware with configurable threat catalogs.
  - Human escalation queue with full conversation context.
  - Guardrail F1/precision/recall evaluation report.
  - Red-team test suite with 50+ adversarial attack vectors.
- **Turnaround:** 3–5 Business Days.
- **Pricing:** **$2,500 USD** | **₹1,00,000 INR**

#### Service 5.2: Comprehensive Red-Team Evaluation
- **Scope:** Systematic adversarial testing across known attack categories (prompt injection, jailbreak, data exfiltration, indirect prompt injection via RAG, multi-turn manipulation); coverage scoring; regression tracking across model/guardrail updates.
- **Deliverables:**
  - Red-team evaluation report with categorized findings.
  - Attack vector catalog with severity ratings.
  - Automated regression test suite for CI/CD integration.
  - Remediation recommendations with priority ranking.
- **Turnaround:** 5–7 Business Days.
- **Pricing:** **$3,500 USD** | **₹1,40,000 INR**

---

### 6. Enterprise Architecture, MLOps & GPU Capacity Planning

#### Service 6.1: Enterprise GenAI Platform Architecture & Capacity Blueprint
- **Scope:** End-to-end architecture design for multi-tenant GenAI platforms; unified `/v1/chat` control plane with bot registry; dynamic model routing engine (SLM/Frontier 70/30 split); first-principles GPU capacity planning (VRAM, KV-cache, continuous batching, Tensor Parallelism) for 10K → 1M concurrent users; zero-trust tool plane with air-gapped execution.
- **Deliverables:**
  - Complete architecture document with system diagrams and data flow.
  - GPU capacity spreadsheet with VRAM/throughput/latency calculations.
  - FinOps TCO model with 3-year cost projections and ROI analysis.
  - Implementation roadmap with phased milestones.
- **Turnaround:** 10–15 Business Days.
- **Pricing:** **$8,000 USD** | **₹3,20,000 INR**

#### Service 6.2: Production GPU Deployment & MLOps Pipeline
- **Scope:** Containerizing ML inference services with Docker; orchestrating on Kubernetes (EKS/GKE) or AWS ECS/Fargate; model versioning with A/B deployment strategies; Prometheus/Grafana monitoring dashboards; automated rollback and scaling policies.
- **Deliverables:**
  - Docker images with multi-stage build optimization.
  - Kubernetes manifests or ECS task definitions with auto-scaling.
  - CI/CD pipeline (GitHub Actions) for automated model deployment.
  - Monitoring dashboard with latency, throughput, error rate, and GPU utilization.
- **Turnaround:** 7–14 Business Days.
- **Pricing:** **$5,000 USD** | **₹2,00,000 INR**

#### Service 6.3: Custom ML Evaluation & Benchmarking Harness
- **Scope:** Building automated evaluation harnesses measuring task-specific metrics (F1, BLEU, ROUGE-L, Perplexity, Token F1), latency percentiles (p50/p90/p95/p99), VRAM tracking, and throughput; CI/CD integration for benchmark regression detection; structured JSON output for dashboarding.
- **Deliverables:**
  - Standalone evaluation harness (`src/eval/`) with CLI interface.
  - `benchmark_metrics.json` ground-truth output format.
  - GitHub Actions workflow for automated benchmark regression.
  - Grafana/Markdown dashboard templates for metric visualization.
- **Turnaround:** 3–5 Business Days.
- **Pricing:** **$2,000 USD** | **₹80,000 INR**

---

## 🛡️ Terms of Service, Security & Governance

1. **Non-Disclosure Agreement (NDA):** Signed prior to receiving any proprietary data, model weights, or project specifications.
2. **Data Custody & Security:** All computation is performed on encrypted, ephemeral instances (AWS / GCP) adhering to enterprise security standards. No client data is ever uploaded to unauthorized third-party AI models or training pipelines.
3. **HIPAA Compliance:** As a former healthcare platform engineer with verified zero-breach posture, I maintain full compliance with PHI data handling requirements.
4. **Intellectual Property (IP):** 100% of code, trained models/adapters, evaluation data, and deliverables belong to the client upon milestone payment completion.
5. **Payment Schedule:**
   - Fixed-price projects: 50% deposit upon contract initiation, 50% upon delivery and client validation.
   - Retainers: Invoiced at the beginning of each 30-day billing cycle.
   - Payment rails supported: Wise (USD/EUR/GBP), Stripe, Direct Wire Transfer, and Razorpay/UPI/NEFT (INR).

---

## 📬 How to Book a Project

1. **Send an inquiry:** Email `ashrafk.salim1@gmail.com` with your use case, model/infrastructure requirements, and timeline.
2. **Introductory Scoping Call (30 mins):** We review technical requirements, infrastructure constraints, and agree on deliverables and milestones.
3. **Formal Proposal & Contract:** Detailed Statement of Work (SOW) with fixed milestones, deliverables, and dates provided within 24 hours.

---

*Ready to ship production-grade AI systems? Let's connect:*  
📧 **Email:** [ashrafk.salim1@gmail.com](mailto:ashrafk.salim1@gmail.com) | 📱 **Phone/WhatsApp:** +91-8779559898  
🌐 **GitHub:** [github.com/aashiq-parinda](https://github.com/aashiq-parinda) | 🤗 **HuggingFace:** [huggingface.co/ashrafksalim](https://huggingface.co/ashrafksalim)
