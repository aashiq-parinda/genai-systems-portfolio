# 🚀 The Ultimate MAANG Generative AI & LLM Engineering Interview Bible

> 🎯 **Target Roles**: GenAI Application Engineer | Applied AI Scientist | GenAI Infrastructure & MLOps Engineer | Lead / Principal AI Architect  
> 🏷️ **Seniority Levels**: 🥉 Mid-Level (L4/E4) | 🥈 Senior (L5/E5) | 🥇 Staff / Principal (L6+/E6+)  
> 🏢 **Target Companies**: Meta | Google (DeepMind/GCP) | Amazon (AWS/Bedrock) | Apple | Netflix | OpenAI  
> 📌 **Repository**: [Cracking The GenAI Portfolio](README.md) | [GenAI Glossary Cheat Sheet](GENAI_GLOSSARY_CHEATSHEET.md)  

---

## ⚡ 10-Second Cheat Sheet for Low-Attention Span Readers

| Interview Role | ⚡ 1-Sentence Secret to Pass | 🎯 Key Formula / Concept |
| :--- | :--- | :--- |
| 🌱 **AI/ML Fundamentals** | Explain Byte-Pair Encoding (BPE) vocabulary merging and how Softmax temperature $T$ flattens/sharpens logit probabilities. | $$P(y_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$ |
| 🤖 **GenAI App Engineer** | Don't just suggest `LangChain`. Build concurrent hybrid RAG (BM25 + Dense RRF) and stream tokens via SSE to keep TTFT < 200ms. | $$RRF = \sum \frac{1}{60 + \text{rank}}$$ |
| 🔬 **Applied AI Scientist** | Derive LoRA ($W_0 + \frac{\alpha}{r}BA$) and DPO implicit loss from scratch; explain why DPO drops the PPO reward model & critic. | $$\mathcal{L}\_{\text{DPO}} = -\log \sigma \left(\beta \log \frac{\pi\_{\theta}}{\pi\_{\text{ref}}}\right)$$ |
| ⚡ **GenAI Infra / MLOps** | Explain how PagedAttention cuts KV cache waste from 60% → 4% and why FP8 Tensor Cores give 2x GEMM speed over FP16 on H100s. | $$\text{VRAM} = \frac{P \times b}{\text{Quant}} + \text{KV Cache}$$ |
| 🏛️ **Lead / Staff Architect** | Design multi-tier model cascades: route 60% queries to local 2B SLMs ($0.00005/req) to keep total cost < $0.001/query. | $$\text{Cost} = \sum c(m_i) P(m_i)$$ |
| 🛡️ **AI Safety & Security** | Mitigate indirect prompt injection in RAG via structural context tags (`<context>`), pre-filtered vector RBAC ACL masks, and Presidio PII DLP. | $$\text{Filter: } \text{tenant}\_{\text{id}} = T\_{\text{req}}$$ |

---

## 📌 Table of Contents
- [🎯 Master Strategy: How to Crack 100% of MAANG GenAI Interviews](#-master-strategy-how-to-crack-100-of-maang-genai-interviews)
  - [1️⃣ The 6-Step MAANG GenAI System Design Blueprint](#1️⃣-the-6-step-maang-genai-system-design-blueprint)
  - [2️⃣ Mathematical Quick Reference & VRAM Cheat Sheet](#2️⃣-mathematical-quick-reference--vram-cheat-sheet)
- [📂 Role 0: Core AI & Machine Learning Fundamentals Round](#-role-0-core-ai--machine-learning-fundamentals-round)
  - [🥉 Mid (L4/E4): Tokenization Mechanics, Temperature Sampling & Softmax Logits (Google / Meta)](#-mid-l4e4-tokenization-mechanics-temperature-sampling--softmax-logits-google--meta)
- [📂 Role 1: GenAI / LLM Application Engineer (Full-Stack & Applied AI)](#-role-1-genai--llm-application-engineer-full-stack--applied-ai)
  - [🥉 Mid (L4/E4): Designing Low-TTFT Technical Doc RAG (Google)](#-mid-l4e4-designing-low-ttft-technical-doc-rag-google)
  - [🥉 Mid (L4/E4): Multi-Tier Guardrailing & Hallucination Defense (Amazon)](#-mid-l4e4-multi-tier-guardrailing--hallucination-defense-amazon)
  - [🥈 Senior (L5/E5): Multi-Agent Software Development & Tool Calling DAGs (Meta)](#-senior-l5e5-multi-agent-software-development--tool-calling-dags-meta)
  - [🥈 Senior (L5/E5): Hybrid Recommendation & LLM Semantic Reranking (Netflix)](#-senior-l5e5-hybrid-recommendation--llm-semantic-reranking-netflix)
  - [🥇 Staff / Principal (L6+): Global 100M QPS Multi-Tenant Knowledge Engine (Apple / Meta)](#-staff--principal-l6-global-100m-qps-multi-tenant-knowledge-engine-apple--meta)
- [📂 Role 2: GenAI Research & Fine-Tuning Engineer / Applied Scientist](#-role-2-genai-research--fine-tuning-engineer--applied-scientist)
  - [🥉 Mid (L4/E4): LoRA vs. QLoRA vs. Full Fine-Tuning Math (Meta)](#-mid-l4e4-lora-vs-qlora-vs-full-fine-tuning-math-meta)
  - [🥉 Mid (L4/E4): Direct Preference Optimization (DPO) vs. PPO Derivation (Google DeepMind)](#-mid-l4e4-direct-preference-optimization-dpo-vs-ppo-derivation-google-deepmind)
  - [🥈 Senior (L5/E5): Fine-Tuning a 32B Domain LLM with Synthetic Data (Amazon)](#-senior-l5e5-fine-tuning-a-32b-domain-llm-with-synthetic-data-amazon)
  - [🥇 Staff / Principal (L6+): Post-Training at Scale: GRPO, PRM & Reasoning Scaling Laws (OpenAI / Google)](#-staff--principal-l6-post-training-at-scale-grpo-prm--reasoning-scaling-laws-openai--google)
- [📂 Role 3: GenAI Infrastructure, ML Platform & MLOps Engineer](#-role-3-genai-infrastructure-ml-platform--mlops-engineer)
  - [🥉 Mid (L4/E4): Continuous Batching & PagedAttention KV Cache Mechanics (Meta)](#-mid-l4e4-continuous-batching--pagedattention-kv-cache-mechanics-meta)
  - [🥉 Mid (L4/E4): Quantization Mathematics: AWQ vs. GPTQ vs. FP8 vs. NF4 (AWS Bedrock)](#-mid-l4e4-quantization-mathematics-awq-vs-gptq-vs-fp8-vs-nf4-aws-bedrock)
  - [🥈 Senior (L5/E5): Speculative Decoding & Medusa Multi-Head Drafters (Netflix / Google)](#-senior-l5e5-speculative-decoding--medusa-multi-head-drafters-netflix--google)
  - [🥇 Staff / Principal (L6+): Distributed 3D Parallelism & NVLink Multi-Node Cluster Scaling (GCP / AWS)](#-staff--principal-l6-distributed-3d-parallelism--nvlink-multi-node-cluster-scaling-gcp--aws)
- [📂 Role 4: Lead / Principal / Staff GenAI Architect](#-role-4-lead--principal--staff-genai-architect)
  - [🥇 Staff (L6): Automated Enterprise Vulnerability & Security Patch Agent (Meta / Apple)](#-staff-l6-automated-enterprise-vulnerability--security-patch-agent-meta--apple)
  - [🏆 Principal (L7): Enterprise Model Gateway, Dynamic Cascading Router & Cache (Netflix / Google)](#-principal-l7-enterprise-model-gateway-dynamic-cascading-router--cache-netflix--google)
- [📂 Role 5: AI Safety, Security, Guardrails & Enterprise Compliance](#-role-5-ai-safety-security-guardrails--enterprise-compliance)
  - [🥇 Staff / Principal (L6+): Preventing Indirect Prompt Injection, Vector RBAC & SOC2/HIPAA Compliance (AWS / GCP)](#-staff--principal-l6-preventing-indirect-prompt-injection-vector-rbac--soc2hipaa-compliance-aws--gcp)
- [📊 Final Pre-Interview Mastery Checklist](#-final-pre-interview-mastery-checklist)

---

## 🎯 Master Strategy: How to Crack 100% of MAANG GenAI Interviews

> [!IMPORTANT]
> 🧠 **The MAANG Interview Filter**: 90% of candidates fail because they give high-level answers using black-box libraries (`LangChain`, `LlamaIndex`). MAANG interviewers hire engineers who understand **systems-level mechanics**: GPU memory bandwidth, KV cache page tables, loss function derivations, and tail latency P99 SLAs!

---

### 1️⃣ The 6-Step MAANG GenAI System Design Blueprint

When given any GenAI system design prompt at Meta, Google, or Amazon, follow this 6-step blueprint:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1️⃣  REQUIREMENTS & SLAS ──► TTFT < 200ms | ITL < 20ms | QPS | Budget      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2️⃣  GPU SIZING MATH     ──► Param Count (P) | Precision | VRAM & FLOPs      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3️⃣  DATA & RAG PIPELINE ──► BM25 + Dense RRF | Chunking | Cross-Encoder   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4️⃣  SERVING INFRA      ──► vLLM PagedAttention | Speculative Decoding       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5️⃣  GUARDRAILS & EVAL   ──► DeBERTa Jailbreak Scan | Ragas Hallucination Check│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6️⃣  FAILURE MODES      ──► GPU Memory OOM | Cache Threshing | Fallbacks     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 2️⃣ Mathematical Quick Reference & VRAM Cheat Sheet

Keep these formulas memorized for rapid whiteboarding arithmetic:

#### 🧮 A. Model Weight VRAM Calculation
$$\text{Memory}_{\text{weights}} = \frac{P \times b}{\text{Quantization Factor}}$$
* $P$ = Parameters (in billions), $b$ = Precision bytes (FP16 = 2, FP32 = 4, INT8 = 1, INT4 = 0.5).
* 💡 *Quick Example*: **70B parameter model in FP16** requires $70 \times 2 = \mathbf{140\text{ GB}}$ VRAM. In **4-bit (NF4)**, it requires $70 \times 0.5 = \mathbf{35\text{ GB}}$ VRAM!

#### 🧮 B. KV Cache Memory Footprint per Sequence
$$\text{Memory}_{\text{KV}} = 2 \times L \times H_{\text{KV}} \times D_{\text{head}} \times S \times B \times b$$
* $L$ = Layers, $H_{\text{KV}}$ = Key-Value Heads, $D_{\text{head}}$ = Head Dimension ($d_{\text{model}} / H_{\text{query}}$), $S$ = Sequence Length, $B$ = Batch Size, $b$ = Precision bytes.

> [!TIP]
> ⚡ **Grouped-Query Attention (GQA) Magic**: In Llama-3-70B, $H_{\text{query}} = 64$ but $H_{\text{KV}} = 8$ (8:1 ratio). This reduces KV cache memory consumption by **8x**, enabling much larger batch sizes ($B$)!

#### 🧮 C. Total Training Memory (Adam Optimizer)
$$\text{Memory}_{\text{training}} \approx 16 \times P \text{ bytes}$$
* **Model Weights (FP16)**: $2P$ bytes  
* **Gradients (FP16)**: $2P$ bytes  
* **Adam Optimizer State (FP32)**: $12P$ bytes ($4P$ for FP32 master weights, $4P$ for 1st moment, $4P$ for 2nd moment).

---

## 📂 Role 0: Core AI & Machine Learning Fundamentals Round

---

### 🥉 Mid (L4/E4): Tokenization Mechanics, Temperature Sampling & Softmax Logits (Google / Meta)

> 🏢 **Company**: Google / Meta  
> 🧠 **Concepts**: Byte-Pair Encoding (BPE), Logit Softmax Transformation, Sampling Temperature ($T$), Sub-word Token Splitting.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> LLMs don't read words or letters — they process sub-word token IDs generated via **Byte-Pair Encoding (BPE)**. Temperature $T$ scales raw logits ($z_i / T$) before Softmax: $T \to 0$ forces greedy deterministic output ($\arg\max$), while $T > 1.0$ flattens probabilities to increase output variety.

#### 🎯 Question Statement
"Explain how Byte-Pair Encoding (BPE) tokenization operates. Why do modern LLMs struggle with character-counting tasks (e.g. counting the letter 'r' in 'strawberry'), and how does adjusting sampling temperature $T$ alter logit probability distributions?"

#### 📐 Softmax Temperature Formula
$$P(y_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

```text
Logit z_i ──► [ Scale by 1/T ] ──► [ Exp(z_i / T) ] ──► [ Normalize Sum ] ──► Prob P(y_i)
```

#### 💻 Python BPE & Temperature Demo

```python
import numpy as np

def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0.0:
        # Greedy Argmax (Temperature = 0)
        probs = np.zeros_like(logits)
        probs[np.argmax(logits)] = 1.0
        return probs
        
    # Scale logits by temperature
    scaled_logits = logits / temperature
    # Subtract max for numerical stability (prevents overflow)
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_logits / np.sum(exp_logits)

# Example: Logits for 3 candidate tokens
logits = np.array([2.0, 1.0, 0.1])
print("T=0.2 (Sharp/Deterministic):", softmax_with_temperature(logits, temperature=0.2))
print("T=1.0 (Standard):          ", softmax_with_temperature(logits, temperature=1.0))
print("T=2.0 (Flat/Creative):     ", softmax_with_temperature(logits, temperature=2.0))
```

> [!TIP]
> 🚀 **MAANG Interviewer Edge**:  
> Tell the interviewer: *"LLMs fail to count letters in 'strawberry' because BPE tokenizers group 'straw' and 'berry' into single token IDs `[42821, 14210]`. The attention layers operate on token vectors, never seeing the underlying character bytes unless explicitly prompted to spell out words character-by-character!"*

---

## 📂 Role 1: GenAI / LLM Application Engineer (Full-Stack & Applied AI)

---

### 🥉 Mid (L4/E4): Designing Low-TTFT Technical Doc RAG (Google)

> 🏢 **Company**: Google / GCP  
> 🧠 **Concepts**: Prefill vs. Decode Phase, Reciprocal Rank Fusion (RRF), Server-Sent Events (SSE), TTFT Latency.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> High TTFT ruins RAG UX. Fix it by running **BM25 keyword search + Dense Vector search concurrently via `asyncio.gather`**, merging ranks with **Reciprocal Rank Fusion (RRF)**, and streaming tokens instantly using **FastAPI Server-Sent Events (SSE)**.

#### 🏗️ Architecture Blueprint

```text
[ User Query ] ──► [ Gateway: FastAPI / SSE Stream ]
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
[ BM25 Keyword Search ]        [ Dense Vector Embeddings ]
   (Sparse Index)                 (Vertex AI / Qdrant)
         │                                 │
         └────────────────┬────────────────┘
                          ▼
             [ Reciprocal Rank Fusion (RRF) ]
                          │
                          ▼
            [ Cohere / BGE Cross-Encoder Reranker ] (Top 5 Chunks)
                          │
                          ▼
            [ Quantized Llama-3 8B / Gemma 2B ] (vLLM Engine)
                          │
                          ▼
             [ Server-Sent Events (SSE) Stream ]
```

#### 📐 Reciprocal Rank Fusion (RRF) Formula
Combining BM25 keyword matching with Dense Vector Cosine Similarity:

$$RRF\_Score(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
where $k = 60$, $M$ is search systems (BM25, Dense), and $r_m(d)$ is document rank.

#### 💻 Concurrent Hybrid RAG Code Implementation

```python
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import vllm

app = FastAPI()

async def hybrid_retrieval(query: str):
    # ⚡ Run BM25 and Vector Search CONCURRENTLY to eliminate network latency
    bm25_task = asyncio.create_task(bm25_search(query))
    vector_task = asyncio.create_task(vector_search(query))
    bm25_res, vector_res = await asyncio.gather(bm25_task, vector_task)
    
    # 🔀 Fusing RRF Scores
    rrf_scores = {}
    k = 60
    for rank, doc in enumerate(bm25_res):
        rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + 1.0 / (k + rank + 1)
    for rank, doc in enumerate(vector_res):
        rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + 1.0 / (k + rank + 1)
        
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:5]

@app.get("/api/v1/stream_qa")
async def stream_qa(query: str):
    top_chunks = await hybrid_retrieval(query)
    prompt = build_context_prompt(query, top_chunks)
    
    async def token_generator():
        # 🚀 vLLM AsyncEngine token streaming
        results_generator = vllm_engine.generate(prompt, sampling_params, request_id)
        async for output in results_generator:
            yield f"data: {output.outputs[0].text}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")
```

> [!TIP]
> 🚀 **MAANG Interviewer Edge**:  
> Tell the interviewer: *"We hit TTFT < 15ms for repeat queries by adding a **Redis Semantic Cache**. If incoming query cosine similarity to a cached entry > 0.96, we serve the cached response immediately, bypassing LLM inference entirely!"*

---

### 🥉 Mid (L4/E4): Multi-Tier Guardrailing & Hallucination Defense (Amazon)

> 🏢 **Company**: Amazon  
> 🧠 **Concepts**: Jailbreak Defense, Hallucination Auditing, System Prompt Leak Protection, Guardrail Latency Budget.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> Never run a 7B LLM to check if a prompt is malicious! Use a lightweight **DeBERTa-v3 ONNX model (5ms)** on input, and run **asynchronous NLI entailment scoring (15ms)** on output to block hallucinations without stopping the stream.

#### 🏗️ Multi-Tier Guardrail Flow

```text
                               ┌─────────────────────────┐
                               │   📥 Incoming Query     │
                               └────────────┬────────────┘
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │ 🛡️ STAGE 1: Light Input Scan (ONNX SLM)       │  <-- Latency: 4ms
                    └───────────────────────┬──────────────────────┘
                                            │
                                  [ Safe ]  │  [ Jailbreak / DAN ] ──► 🚨 Block (400)
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │ 🔍 STAGE 2: Intent Router & Context RAG     │  <-- Latency: 20ms
                    └───────────────────────┬──────────────────────┘
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │ ⚡ STAGE 3: Core LLM Generation              │  <-- SSE Token Stream
                    └───────────────────────┬──────────────────────┘
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │ ⚖️ STAGE 4: Output Grounding NLI Judge       │  <-- Latency: 15ms
                    └──────────────────────────────────────────────┘
```

#### 📊 Guardrail Latency Budget Table

| Guardrail Tier | Model / Tech | Latency Budget | Action on Failure |
| :--- | :--- | :--- | :--- |
| **Input Sanitization** | Regex + DeBERTa-v3 ONNX | **4 ms** | Hard Reject (HTTP 400) |
| **RAG Grounding Check** | Vector Overlap + BM25 Score | **10 ms** | Fallback to Pre-written Macro |
| **Output PII Masking** | Presidio / Regex Token Buffer | **2 ms** | Replace with `[REDACTED_PII]` |

---

### 🥈 Senior (L5/E5): Multi-Agent Software Development & Tool Calling DAGs (Meta)

> 🏢 **Company**: Meta  
> 🧠 **Concepts**: ReAct Pattern, Stateful DAG Execution, Loop Detection, Context Window Compression, Sandbox Isolation.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> Prevent code-generating agents from looping infinitely by calculating a **SHA256 hash of (CodeDiff + TestErrors)** per turn. If a hash repeats, kill the loop! Compress old terminal outputs into **Key Lessons Summaries** so context never overflows.

#### 🏗️ Agentic State Machine Architecture

```text
                       ┌────────────────────────────────┐
                       │   📥 GitHub Issue Trigger      │
                       └───────────────┬────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    🧠 PLANNER AGENT (Llama-3-70B)                           │
│              Generates Step DAG: [Parse, Code, Test, Refactor]              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    💻 CODER AGENT (CodeLlama/Qwen2.5)                       │
│               Modifies target files in transient workspace                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    🧪 TEST RUNNER AGENT (gVisor Sandbox)                    │
│               Executes `pytest` in isolated microVM container               │
└───────────────┬──────────────────────────────────────────────┬──────────────┘
                │                                              │
       ❌ [ Tests Fail ]                                💥 [ Tests Pass ]
                │                                              │
                ▼                                              ▼
┌───────────────────────────────┐              ┌───────────────────────────────┐
│  🔍 CRITIC DIAGNOSTIC AGENT   │              │   🚀 PULL REQUEST GENERATOR   │
│ Summarizes stack trace & diff │              └───────────────────────────────┘
└───────────────┬───────────────┘
                │
                ▼ (Max Iterations = 5)
   🔄 [ Loop Back to Coder Agent ]
```

#### 💻 Infinite Loop Detection Code

```python
class AgentExecutionDAG:
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.visited_hashes = set()

    def run_agent_loop(self, issue_desc: str):
        state = {"issue": issue_desc, "iteration": 0, "history": []}
        
        while state["iteration"] < self.max_iterations:
            state["iteration"] += 1
            code_diff = coder_agent.generate_diff(state)
            test_results = sandbox_executor.run_pytest()
            
            if test_results.passed:
                return pr_generator.create_pr(state)
                
            # 🛑 Deterministic Loop Prevention Hash
            state_hash = hash_state(code_diff, test_results.failure_summary)
            if state_hash in self.visited_hashes:
                raise LoopDetectedError("Agent repeated identical code modification cycle!")
            self.visited_hashes.add(state_hash)
            
            # 📦 Compress context for next iteration
            state["history"].append(critic_agent.summarize_failure(test_results))
            
        raise MaxIterationsExceededError("Agent reached max iterations without passing tests.")
```

---

### 🥈 Senior (L5/E5): Hybrid Recommendation & LLM Semantic Reranking (Netflix)

> 🏢 **Company**: Netflix  
> 🧠 **Concepts**: Two-Stage Recommendation Architecture, Collaborative Filtering + LLM Reranker, Logit Calibration.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> Never feed millions of catalog items directly into an LLM! Use a fast **Two-Tower Neural Net (20ms)** to retrieve 500 candidates, then use an **8B LLM prefill pass to score logits over the top 30 candidates (35ms)**, hitting P99 < 100ms!

#### 🏗️ Two-Stage Recommendation System

```text
[ User Query / Context ]
       │
       ├───────────────────────────────────────┐
       ▼                                       ▼
[ Candidate Retrieval Stage ]          [ Semantic Intent Extraction ]
   Two-Tower DNN / Vector DB               Fine-Tuned MiniLM / SLM
   Retrieves ~500 Candidates              Extracts Genre, Tone, Era
       │                                       │
       └───────────────────┬───────────────────┘
                           ▼
              [ Candidate Fusion & Scoring ]
                           │
                           ▼
          [ ⚡ LLM Reranker: 8B Quantized Model ]
           (Evaluates top 30 candidates via logits)
                           │
                           ▼
          [ 🎬 Final Ranked Movie Carousel ]
```

---

### 🥇 Staff / Principal (L6+): Global 100M QPS Multi-Tenant Knowledge Engine (Apple / Meta)

> 🏢 **Company**: Apple / Meta  
> 🧠 **Concepts**: Multi-Tenant Isolation, Tiered Routing, Cost-per-Query SLAs, Global Gateways.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> To guarantee **<$0.001 per query cost**, build a **Complexity Evaluator Router**: send 60% simple queries to fast on-device SLMs ($0.00005/req), 30% queries to self-hosted 8B models ($0.0003/req), and reserve expensive frontier models (GPT-4o/Llama-70B) for top 10% hard queries!

#### 🏗️ Enterprise Multi-Tenant Architecture

```text
[ Client Requests (50+ Internal Apps) ]
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   🌐 GLOBAL API GATEWAY (Envoy / Istio)                     │
│    Token Bucket Rate Limiting | Dynamic Tenant Auth | PII Masking Engine    │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   ⚡ SEMANTIC CACHE LAYER (Redis Enterprise)                │
│             Similarity Threshold > 0.95 ──► Instant Return (15ms)           │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │ (Cache Miss)
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   🧠 DYNAMIC MODEL ROUTING ENGINE                           │
│                                                                             │
│   Complexity Evaluator ──► Score < 0.3  ──► On-Device SLM (60% traffic)     │
│   (DeBERTa Router)     ──► Score 0.3-0.8 ──► Quantized 8B (30% traffic)    │
│                        ──► Score > 0.8  ──► Frontier 70B (10% traffic)     │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
┌──────────────┐ ┌───────────┐ ┌──────────────┐
│ On-Device SLM│ │ vLLM (8B) │ │ Multi-GPU Pod│
│ (Apple NPU)  │ │ (AWS Infer)│ │ (H100 NVLink)│
└──────────────┘ └───────────┘ └──────────────┘
```

#### 🧮 Weighted Cost Calculation Formula
$$\text{Cost}_{\text{avg}} = (0.60 \times \$0.00005) + (0.30 \times \$0.0003) + (0.10 \times \$0.004) = \mathbf{\$0.00052 \text{ / query}}$$
*(Well under the $0.001 enterprise budget threshold!)*

---

## 📂 Role 2: GenAI Research & Fine-Tuning Engineer / Applied Scientist

---

### 🥉 Mid (L4/E4): LoRA vs. QLoRA vs. Full Fine-Tuning Math (Meta)

> 🏢 **Company**: Meta  
> 🧠 **Concepts**: Low-Rank Adaptation (LoRA), QLoRA NormalFloat4 (NF4), Double Quantization, Memory Footprint.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> LoRA freezes base weights $W_0$ and trains low-rank matrices $B \cdot A$. **QLoRA compresses base weights to 4-bit NF4**, reducing 70B fine-tuning VRAM from **1,120 GB down to 41.4 GB**, enabling a 70B model to be trained on a **single 80GB A100 GPU**!

#### 📐 LoRA Formula
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B A) x$$
* Matrix $A \in \mathbb{R}^{r \times k}$ initialized via $\mathcal{N}(0, \sigma^2)$, Matrix $B \in \mathbb{R}^{d \times r}$ initialized to $0$.

```text
                     Full Weight W_0 (Frozen)
                         [ d  x  k ]
                              │
  Input x ────────────────────┼────────────────────► (+) ──► Output h
                              │
                     ┌────────┴────────┐
                     │ Low-Rank Adapt  │
                     │  Matrix A (r x k)
                     │        │        │
                     │        ▼        │
                     │  Matrix B (d x r)
                     └─────────────────┘
```

#### 📊 70B Fine-Tuning Memory Footprint Matrix

| Memory Component | Full Fine-Tuning (FP16) | LoRA (FP16 Adapt + FP16 Base) | QLoRA (NF4 Base + FP16 Adapt) |
| :--- | :--- | :--- | :--- |
| **Base Model Weights** | 140 GB (FP16) | 140 GB (FP16) | **35 GB (NF4)** |
| **Adapter Parameters ($r=16$)** | 0 GB | 0.8 GB | 0.8 GB |
| **Gradients** | 140 GB | 0.8 GB | 0.8 GB |
| **Optimizer States (Adam)** | 840 GB (FP32) | 4.8 GB (FP32) | 4.8 GB (FP32) |
| **💾 Total VRAM Required** | **~1,120 GB (14x H100s)** | **~146 GB (2x H100s)** | **~41.4 GB (1x A100 80GB! 🎉)** |

---

### 🥉 Mid (L4/E4): Direct Preference Optimization (DPO) vs. PPO Derivation (Google DeepMind)

> 🏢 **Company**: Google DeepMind  
> 🧠 **Concepts**: RLHF, Bradley-Terry Model, Closed-Form Implicit Reward, DPO Loss Function.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> DPO mathematically re-parameterizes the reward function in terms of policy weights $\pi_\theta(y \mid x) / \pi_{\text{ref}}(y \mid x)$. This **completely eliminates training a separate Reward Model & Critic network**, making alignment fast and stable!

#### 📐 DPO Loss Function Derivation
Starting from the Bradley-Terry preference model $P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$, substituting the implicit reward yields:

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

---

### 🥈 Senior (L5/E5): Fine-Tuning a 32B Domain LLM with Synthetic Data (Amazon)

> 🏢 **Company**: Amazon  
> 🧠 **Concepts**: Evol-Instruct, MinHash LSH Deduplication, Data Quality Filtering, Contamination Auditing.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> Don't train on raw synthetic data! Use **Evol-Instruct** to mutate simple prompts into complex reasoning questions, run **MinHash LSH (0.8 Jaccard threshold)** to remove near-duplicates, and filter out low-reward responses.

```text
[ Raw Domain Documents ] ──► [ Evol-Instruct Mutation ] ──► [ Synthetic LLM Generation ] ──► [ MinHash LSH Deduplication ] ──► [ Clean Fine-Tuning Dataset ]
```

---

### 🥇 Staff / Principal (L6+): Post-Training at Scale: GRPO, PRM & Reasoning Scaling Laws (OpenAI / Google)

> 🏢 **Company**: OpenAI / Google DeepMind  
> 🧠 **Concepts**: Group Relative Policy Optimization (GRPO), Process Reward Models (PRM), Test-Time Search Scaling.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> Models like **DeepSeek-R1** and **OpenAI o1** use **GRPO** (Group Relative Policy Optimization) instead of PPO. GRPO samples a group of $G$ outputs for a prompt and calculates relative normalized advantage across the group, **eliminating the Value (Critic) model completely**!

#### 📐 GRPO Advantage Calculation
$$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}$$

```text
                                [ Prompt q ]
                                     │
      ┌──────────────────────────────┼──────────────────────────────┐
      ▼                              ▼                              ▼
 [ Output o_1 ]                 [ Output o_2 ]                 [ Output o_G ]
  Reward: r_1                    Reward: r_2                    Reward: r_G
      │                              │                              │
      └──────────────────────────────┼──────────────────────────────┘
                                     ▼
                      [ Group Relative Advantage ]
                  A_i = (r_i - mean(r)) / std(r)
```

---

## 📂 Role 3: GenAI Infrastructure, ML Platform & MLOps Engineer

---

### 🥉 Mid (L4/E4): Continuous Batching & PagedAttention KV Cache Mechanics (Meta)

> 🏢 **Company**: Meta  
> 🧠 **Concepts**: Static vs. Continuous Batching, Virtual Memory Paging, Fragmented VRAM, vLLM Engine.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> Static batching wastes GPU compute on padding tokens. **Continuous batching** operates at the iteration level to inject new queries immediately. **PagedAttention** organizes KV cache into non-contiguous physical pages, cutting memory waste from **60% → <4%**!

```text
LOGICAL KV BLOCKS                BLOCK TABLE               PHYSICAL GPU MEMORY PAGES
[ Req 0: Block 0 ] ──────────► [ Block 0 ──► Page 7 ] ────► Physical Memory Page 7
[ Req 0: Block 1 ] ──────────► [ Block 1 ──► Page 3 ] ────► Physical Memory Page 3
[ Req 1: Block 0 ] ──────────► [ Block 0 ──► Page 12] ────► Physical Memory Page 12
```

---

### 🥉 Mid (L4/E4): Quantization Mathematics: AWQ vs. GPTQ vs. FP8 vs. NF4 (AWS Bedrock)

> 🏢 **Company**: AWS Bedrock / ML Platform  
> 🧠 **Concepts**: AWQ Channel Scaling, GPTQ Second-Order Optimization, Tensor Core FP8 Execution.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> **AWQ** protects the top 1% salient weight channels via per-channel scaling without hardware speed penalties. On **NVIDIA H100 GPUs**, native **FP8 (E4M3/E5M2)** delivers **2x GEMM throughput** over FP16 with zero perplexity loss!

#### 📐 AWQ Scaling Formula
$$s_x = \arg\min_s \left\| W X - \text{Quantize}(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1} X \right\|$$

---

### 🥈 Senior (L5/E5): Speculative Decoding & Medusa Multi-Head Drafters (Netflix / Google)

> 🏢 **Company**: Netflix / Google  
> 🧠 **Concepts**: Memory Bandwidth vs Compute Bound, Draft Model Rejection Sampling, Medusa Heads.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> LLM generation is memory bandwidth bound. Speculative Decoding uses a small **Draft Model (68M)** to rapidly draft 4 tokens, then uses the **Large Target Model (70B) to verify all 4 tokens concurrently in a single forward pass**, giving a **2x-4x speedup** with zero loss in output quality!

```text
[ Fast Draft Model (68M) ] ──► Drafts 4 Tokens ──► [ Target Model (70B) ] ──► Single-Pass Parallel Verification ──► 🚀 3x Speedup!
```

---

### 🥇 Staff / Principal (L6+): Distributed 3D Parallelism & NVLink Multi-Node Cluster Scaling (GCP / AWS)

> 🏢 **Company**: GCP / AWS / Meta  
> 🧠 **Concepts**: Tensor Parallelism (TP), Pipeline Parallelism (PP), Data Parallelism (ZeRO-3), NVLink vs. InfiniBand.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> To train a 405B model across 1,024 H100 GPUs: set **TP = 8** inside each 8-GPU node (communicating over **900 GB/s NVLink**), set **PP = 16** across nodes to split layers, and use **ZeRO-3 Data Parallelism (DP = 8)** over 400 Gbps InfiniBand switches!

```text
                             1,024 H100 GPU CLUSTER
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
  [ Tensor Parallel (TP=8) ]   [ Pipeline Parallel (PP=16) ]   [ Data Parallel (DP/ZeRO-3 = 8) ]
  Intra-Node (NVLink @ 900GB/s)   Inter-Node (InfiniBand/RoCE)    Inter-Node All-Reduce
```

---

## 📂 Role 4: Lead / Principal / Staff GenAI Architect

---

### 🥇 Staff (L6): Automated Enterprise Vulnerability & Security Patch Agent (Meta / Apple)

> 🏢 **Company**: Meta / Apple  
> 🧠 **Concepts**: Static AST Parser, Vulnerability Remediation Agent, gVisor Execution Sandbox, Security Approval Gate.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> Build an agentic security patcher: trigger on Semgrep scanner webhooks, parse AST to scope the vulnerability line range, generate minimal diffs using DeepSeek-Coder, run tests in an isolated **gVisor microVM sandbox**, and auto-merge if CVSS < 7.0!

```text
[ Vulnerability Webhook ] ──► [ AST Parser ] ──► [ Patch Agent ] ──► [ gVisor Test Sandbox ] ──► [ Auto-Merge / Security Escalation ]
```

---

### 🏆 Principal (L7): Enterprise Model Gateway, Dynamic Cascading Router & Cache (Netflix / Google)

> 🏢 **Company**: Netflix / Google / Amazon  
> 🧠 **Concepts**: Multi-Cloud Gateway, Envoy C++ Filters, Semantic Response Caching, SLA Circuit Breakers.

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> Architect a multi-cloud AI Gateway serving 1 Billion requests/day: run rate-limiting and PII redaction inside an **Envoy C++ proxy filter (<1.2ms latency)**, hit a **Redis Semantic Cache**, and dynamically fallback (vLLM → Azure OpenAI → AWS Bedrock) if upstream errors spike!

```text
[ Global Requests ] ──► [ Envoy C++ Proxy Gateway (<1.2ms) ] ──► [ Redis Semantic Cache ] ──► [ Dynamic Cascading Router ] ──► [ Upstream LLM Providers ]
```

---

## 📂 Role 5: AI Safety, Security, Guardrails & Enterprise Compliance

---

### 🥇 Staff / Principal (L6+): Preventing Indirect Prompt Injection, Vector RBAC & SOC2/HIPAA Compliance (AWS / GCP)

> 🏢 **Company**: AWS / GCP  
> 🧠 **Concepts**: Direct vs. Indirect Prompt Injection, Vector DB Pre-filtering RBAC/ABAC, Presidio PII DLP, Zero Data Retention (ZDR).

> [!NOTE]
> ⚡ **10-Second TL;DR**:  
> Indirect prompt injection is the #1 enterprise RAG security vulnerability. Block it by enforcing structural prompt tags (`<context>`), isolating untrusted text data from instructions, applying **pre-filtering payload masks (RBAC)** in vector databases, and using streaming token PII buffers!

#### 🎯 Question Statement
"Architect an enterprise AI security and compliance platform for a multi-tenant healthcare/financial RAG assistant. How do you defend against indirect prompt injection embedded inside uploaded PDF context, enforce tenant-level role-based access control (RBAC) in vector search, and guarantee SOC 2 Type II / HIPAA Zero-Data-Retention SLAs?"

#### 🏗️ Enterprise AI Security Blueprint

```text
[ User Query + OAuth JWT Token ]
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🛡️ STAGE 1: Edge API Gateway (Envoy + PII Redaction)                         │
│   Extracts Tenant ID & Clearance Level | Runs Presidio PII Masking           │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🔑 STAGE 2: Pre-Filtered Vector DB Search (Qdrant / Milvus)                 │
│   Enforces Pre-Filter Payload Mask: `tenant_id == T AND clearance <= U`     │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │ (Retrieved Chunks)
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🔒 STAGE 3: Structural Context Sanitization & Prompt Isolation              │
│   Wraps untrusted chunk data inside `<untrusted_context>` XML tags           │
│   Applies Indirect Prompt Injection ONNX Scanner                            │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ⚡ STAGE 4: Enterprise LLM Engine (Zero-Data-Retention SLA)                  │
│   KMS Envelope Encryption | Ephemeral Memory | SSE Output PII Stream Buffer │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 💻 Vector DB Pre-Filtering RBAC Code (Qdrant)

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models

qdrant_client = QdrantClient(host="localhost", port=6333)

def search_rag_with_rbac(query_vector: list, user_tenant_id: str, user_clearance_level: int):
    # 🔒 Enforce PRE-FILTERING (Filtering vectors BEFORE graph traversal)
    rbac_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=user_tenant_id)
            ),
            models.FieldCondition(
                key="security_clearance",
                range=models.Range(lte=user_clearance_level)
            )
        ]
    )
    
    # Search Qdrant vector database with strict tenant RBAC filter
    search_results = qdrant_client.search(
        collection_name="enterprise_docs",
        query_vector=query_vector,
        query_filter=rbac_filter,
        limit=5
    )
    return search_results
```

> [!TIP]
> 🚀 **MAANG Interviewer Edge**:  
> Tell the interviewer: *"Never use post-filtering (retrieving top 100 vectors then dropping unauthorized ones in Python code) because if top candidate vectors belong to another tenant, post-filtering severely degrades recall. Pre-filtering inside HNSW payload masks guarantees 100% recall over authorized documents!"*

---

## 📊 Final Pre-Interview Mastery Checklist

Before stepping into your MAANG interview room, check off every box:

- [ ] 🧮 **GPU Memory Math**: Can calculate exact FP16 weight VRAM, KV cache size, and Adam optimizer overhead on a whiteboard in 60 seconds.
- [ ] ⚡ **KV Cache Optimization**: Can explain Grouped-Query Attention (GQA) ratios and PagedAttention virtual memory block tables.
- [ ] 📐 **Fine-Tuning Formulations**: Can derive LoRA ($W_0 + \frac{\alpha}{r}BA$), QLoRA NF4 quantization, and DPO loss functions from memory.
- [ ] 🚀 **Inference Acceleration**: Can explain Speculative Decoding rejection sampling math, continuous batching, and FP8 Tensor Core speedups.
- [ ] 🧠 **Post-Training at Scale**: Can explain why GRPO eliminates the Value/Critic model in reasoning models like DeepSeek-R1 and OpenAI o1.
- [ ] 🛡️ **Security & Guardrails**: Can design end-to-end indirect prompt injection defenses, pre-filtered vector RBAC queries, and streaming PII redactors.

---
*This guide is part of the [Cracking The GenAI Portfolio](README.md) open-source repository.*
