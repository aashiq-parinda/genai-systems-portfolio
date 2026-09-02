# Ashraf Khan
**Machine Learning Engineer · Generative AI Systems Engineer · LLM Fine-Tuning & Inference Optimization**  
📍 Mumbai, Maharashtra, India | 📧 [ashrafksalim1@gmail.com](mailto:ashrafksalim1@gmail.com) | 📞 +91-8779559898  
🌐 **GitHub**: [github.com/aashiq-parinda](https://github.com/aashiq-parinda) | 🤗 **HuggingFace**: [huggingface.co/ashrafksalim](https://huggingface.co/ashrafksalim) | 💼 **LinkedIn**: [linkedin.com/in/ashrafksalim](https://linkedin.com/in/ashrafksalim)

---

## 🎯 Professional Executive Summary
Production-grade **Machine Learning & Generative AI Systems Engineer** with **5+ years of software engineering experience** and **3 published GenAI research-grade portfolio systems** with empirically verified benchmark receipts. Specializes in LLM inference optimization (4-bit NF4 quantization, LoRA/PEFT fine-tuning), agentic reasoning pipelines (multi-stage DAGs, hybrid RAG), adversarial safety guardrails, and FastAPI streaming deployments on GPU hardware (T4/L4). All systems are backed by reproducible evaluation harnesses, CI/CD automation, and measurable cost, latency, and financial impact metrics.

---

## 🤖 Generative AI & Machine Learning Portfolio Projects

### 1. Customer Support Automation — Quantized LLM + Agentic Workflow
**GitHub**: [Customer-Support-Automation-Quantized-LLM-Agentic-Workflow](https://github.com/aashiq-parinda/Customer-Support-Automation-Quantized-LLM-Agentic-Workflow) | **HuggingFace Demo**: [Live Space](https://huggingface.co/spaces/ashrafksalim/customer-support-quantized-llm-agent) | **Colab**: [Interactive Notebook](https://colab.research.google.com/drive/1RIf9_bZAoqmB9rq6nNWciNmaSPZJSpOx?usp=sharing)

* **Problem**: High inference costs and latency SLAs when deploying 7B+ LLMs for high-volume tier-1 customer support, coupled with adversarial prompt injection and billing/legal safety risks.
* **4-Bit NF4 Quantization** (`bitsandbytes`): VRAM reduced from **14.54 GB → 5.09 GB (65% reduction)**, enabling low-cost T4/L4 GPU deployment.
* **Multi-Tier Safety Guardrails**: Real-time threat catalog blocking DAN-mode jailbreaks, system prompt leaks, and legal/fraud keywords (`chargeback`, `subpoena`, `GDPR`).
* **Enterprise Hybrid RAG**: BM25 keyword search fused with Dense Vector Cosine Similarity via Reciprocal Rank Fusion (RRF) for grounded, hallucination-resistant responses.
* **FastAPI SSE Streaming**: Server-Sent Events token-streaming endpoint for ultra-low Time-To-First-Token (TTFT).
* **Empirical Receipts (Verified `benchmark_metrics.json`):**
  * Cost per 1,000 tickets: **$0.8921 → $0.0297 (96.67% cost reduction)**
  * Mean latency: **7.68x improvement** across p50/p90/p95/p99 percentiles
  * Safety Guardrail F1: **0.9412**

---

### 2. Contract Risk Review — LoRA Fine-Tuning & Multi-Agent Legal Reasoning
**GitHub**: [LLM-Fine-Tuning-Agentic-Reasoning-for-Contract-Risk-Review](https://github.com/aashiq-parinda/LLM-Fine-Tuning-Agentic-Reasoning-for-Contract-Risk-Review)

* **Problem**: Generic LLMs hallucinate and miss liability, indemnification, and governing law risks in enterprise legal contracts — high-stakes failures with direct financial consequences.
* **LoRA / PEFT Fine-Tuning** on DeBERTa-v3 and Qwen2.5, trained on the LEDGAR legal corpus for deterministic clause risk classification.
* **ChromaDB Precedent Retrieval**: Vector embedding retrieval grounding risk determinations against statutory definitions and contract playbooks.
* **Fairness & Bias Auditing**: Rigorous evaluation across contract jurisdictions and party asymmetries.
* **Empirical Receipts:**
  * Clause Risk Classification F1: **1.000** (balanced LEDGAR evaluation set)
  * Precision liability, indemnification, and governing law extraction with near-zero hallucination rate on clause boundaries

---

### 3. Quantum Hardware Validation & Benchmarking Suite
**GitHub**: [Quantum-Hardware-Validation-Benchmarking-Suite](https://github.com/aashiq-parinda/Quantum-Hardware-Validation-Benchmarking-Suite)

* **Problem**: No standardized empirical reporting framework for evaluating NISQ circuits and algorithmic fidelity across realistic hardware error models.
* **Hardware-Agnostic Benchmarking Engine**: Profiles quantum circuit depth, gate counts, and decoherence times (T1, T2).
* **Noise Simulation & Error Mitigation**: Evaluates Zero-Noise Extrapolation (ZNE) and readout error mitigation strategies.
* **Automated Empirical Reporting**: Publishes fidelity decay curves, error benchmarking results, and algorithmic profiling dashboards.
* 100% passing Pytest unit & integration suites with reproducible `benchmark_metrics.json` receipts.

---

## 💼 Professional Experience

### **Senior Software & Systems Engineer** | Logicloop *(Mumbai, India | Jan 2026 – Present)*
* Built MAXX AI Voice Assistant with custom wake-word activation and multi-advisor conversational flows via Millis.ai (LLM integration).
* Led 3 developers; enforced CI/CD automation and Level 2 VAPT security standards across enterprise financial flows.

### **Full-Stack Developer & Data Custodian** | Fitwell Technologies Inc. *(Remote – VA, USA | Jun 2025 – Jan 2026)*
* Integrated LLM-powered features, Amazon Connect voice AI workflows, and HIPAA-compliant data pipelines on AWS private cloud.
* Deployed self-hosted Supabase PostgreSQL cloud backend on EC2 for PHI data isolation; maintained zero data breach posture.

### **Lead Software Engineer** | QuantGen Private Limited *(India | Nov 2024 – Jun 2025)*
* Delivered AI-integrated health platform (FitWell) and enterprise 24/7 operations platform (Kokomo) with AWS ECS/App Runner orchestration.

### **Software Engineer** | Logicloop *(Mumbai, India | May 2024 – Oct 2024)*
* Built Flutter frontend for HDFC Life InstaQuote, connecting to insurance pricing ML backend APIs.

### **Software Engineer** | SoftProdigy *(Remote – USA | Nov 2023 – Apr 2024)*
* Developed loan lending application workflows and implemented gRPC client API libraries.

### **Senior Software Development Engineer** | AKcess Labs Pvt. Ltd. *(Remote – London, UK | Jan 2023 – Nov 2023)*
* Built Flutter cross-platform applications with Flask backends for hyperlocal commerce platform.

### **Software Developer** | We3.Tech *(Maharashtra, India | Aug 2021 – Jan 2023)*
* Delivered eKYC, Digio eSign integrations, and banking API modules for Motilal Oswal MO Investor (SEBI/RBI compliant cryptographic security).

---

## 💻 Technical Skills & Core Competencies

* **LLM Optimization & Inference**: 4-bit NF4 Quantization, `bitsandbytes`, GGUF, Torch `float16`, VRAM profiling, T4/L4 GPU deployment.
* **Fine-Tuning & PEFT**: LoRA, QLoRA, PEFT Adapters, DeBERTa-v3, Qwen2.5, LEDGAR Corpus, Hugging Face `transformers`, `trl`.
* **Agentic Pipelines & RAG**: Multi-stage DAGs, ChromaDB, BM25 + Dense RRF Hybrid Search, Reciprocal Rank Fusion, LangChain, Prompt Engineering.
* **Safety & Adversarial Defense**: Multi-tier guardrail catalogs, DAN/jailbreak interception, GDPR/legal keyword routing, human escalation DAGs.
* **APIs & Serving**: FastAPI (Async), SSE token streaming, REST, gRPC, Protobuf, WebSockets.
* **ML Frameworks**: PyTorch, TensorFlow, scikit-learn, OpenCV, Hugging Face Ecosystem.
* **Evaluation & Benchmarking**: p50/p90/p95/p99 latency percentiles, F1/Precision/Recall, VRAM tracking, Pytest CI/CD.
* **Cloud & DevOps**: AWS (EC2, S3, ECS, Fargate, App Runner), GCP, Docker, GitHub Actions CI/CD.
* **Programming**: Python (Expert), Flutter/Dart, React Native, Node.js, SQL/PostgreSQL.

---

## 🎓 Education & Certifications

### Education
* **B.Sc. in Computer Science** | Rizvi College, Mumbai University *(2017 – 2020)* — CGPA: 9.33 / 10.00
* **Vocational Computer Science / HSC (12th)** | M.H. Saboo Siddik Polytechnic, Mumbai *(2015 – 2017)* — 75.80%

### Certifications
* **AI & Emerging Tech**: AI & Quantum Computing Mastery, Machine Learning with Python, Complete Quantum Computing Course.
* **Healthcare Compliance**: Healthcare API Compliance & HIPAA PHI Data Governance.
* **FinTech Standards**: FinTech Regulatory Standards & PCI-DSS Payment Processing.
