# 🚀 The Ultimate MAANG Generative AI & LLM Engineering Interview Bible

> **Target Roles**: GenAI Application Engineer | Applied AI Scientist | GenAI Infrastructure & MLOps Engineer | Lead / Principal AI Architect  
> **Seniority Levels**: Mid-Level (L4/E4/IC4) | Senior (L5/E5/IC5) | Staff / Principal / Lead (L6+/E6+/IC6+)  
> **Target Companies**: Meta, Google (DeepMind/GCP), Amazon (AWS/Bedrock), Apple, Netflix, Microsoft / OpenAI  
> **Author & Repository**: [Cracking The GenAI Portfolio](README.md)  

---

## 📌 Table of Contents
1. [🎯 Master Strategy: How to Crack 100% of MAANG GenAI Interviews](#-master-strategy-how-to-crack-100-of-maang-genai-interviews)
   - [The 6-Step MAANG GenAI System Design Framework](#1-the-6-step-maang-genai-system-design-framework)
   - [Mathematical Quick Reference & Estimation Formula Cheat Sheet](#2-mathematical-quick-reference--estimation-formula-cheat-sheet)
2. [📂 Role 1: GenAI / LLM Application Engineer (Full-Stack & Applied AI)](#-role-1-genai--llm-application-engineer-full-stack--applied-ai)
   - [Mid-Level (L4/E4): Designing Low-TTFT Technical Doc RAG (Google)](#mid-level-l4e4-designing-low-ttft-technical-doc-rag-google)
   - [Mid-Level (L4/E4): Multi-Tier Guardrailing & Hallucination Defense (Amazon)](#mid-level-l4e4-multi-tier-guardrailing--hallucination-defense-amazon)
   - [Senior (L5/E5): Multi-Agent Software Development & Tool Calling DAGs (Meta)](#senior-l5e5-multi-agent-software-development--tool-calling-dags-meta)
   - [Senior (L5/E5): Hybrid Recommendation & LLM Semantic Reranking Engine (Netflix)](#senior-l5e5-hybrid-recommendation--llm-semantic-reranking-engine-netflix)
   - [Staff / Principal (L6+): Global 100M QPS Multi-Tenant Knowledge Engine (Apple / Meta)](#staff--principal-l6-global-100m-qps-multi-tenant-knowledge-engine-apple--meta)
3. [📂 Role 2: GenAI Research & Fine-Tuning Engineer / Applied Scientist](#-role-2-genai-research--fine-tuning-engineer--applied-scientist)
   - [Mid-Level (L4/E4): Mathematical Formulations of LoRA vs. QLoRA vs. Full Fine-Tuning (Meta)](#mid-level-l4e4-mathematical-formulations-of-lora-vs-qlora-vs-full-fine-tuning-meta)
   - [Mid-Level (L4/E4): Direct Preference Optimization (DPO) vs. PPO Derivation (Google DeepMind)](#mid-level-l4e4-direct-preference-optimization-dpo-vs-ppo-derivation-google-deepmind)
   - [Senior (L5/E5): Fine-Tuning a 32B Domain LLM with Synthetic Data Pipelines (Amazon)](#senior-l5e5-fine-tuning-a-32b-domain-llm-with-synthetic-data-pipelines-amazon)
   - [Staff / Principal (L6+): Post-Training at Scale: GRPO, PRM, and Reasoning Scaling Laws (OpenAI / Google)](#staff--principal-l6-post-training-at-scale-grpo-prm-and-reasoning-scaling-laws-openai--google)
4. [📂 Role 3: GenAI Infrastructure, ML Platform & MLOps Engineer](#-role-3-genai-infrastructure-ml-platform--mlops-engineer)
   - [Mid-Level (L4/E4): Continuous Batching & PagedAttention KV Cache Mechanics (Meta)](#mid-level-l4e4-continuous-batching--pagedattention-kv-cache-mechanics-meta)
   - [Mid-Level (L4/E4): Quantization Mathematics: AWQ vs. GPTQ vs. FP8 vs. NF4 (AWS Bedrock)](#mid-level-l4e4-quantization-mathematics-awq-vs-gptq-vs-fp8-vs-nf4-aws-bedrock)
   - [Senior (L5/E5): Speculative Decoding & Medusa Multi-Head Drafters (Netflix / Google)](#senior-l5e5-speculative-decoding--medusa-multi-head-drafters-netflix--google)
   - [Staff / Principal (L6+): Distributed 3D Parallelism & NVLink Multi-Node Cluster Scaling (GCP / AWS)](#staff--principal-l6-distributed-3d-parallelism--nvlink-multi-node-cluster-scaling-gcp--aws)
5. [📂 Role 4: Lead / Principal / Staff GenAI Architect](#-role-4-lead--principal--staff-genai-architect)
   - [Staff (L6): Automated Enterprise Vulnerability & Security Patch Agent System (Meta / Apple)](#staff-l6-automated-enterprise-vulnerability--security-patch-agent-system-meta--apple)
   - [Principal (L7): Enterprise Model Gateway, Dynamic Cascading Router & Semantic Cache (Netflix / Google)](#principal-l7-enterprise-model-gateway-dynamic-cascading-router--semantic-cache-netflix--google)
6. [📊 Final Pre-Interview Mastery Checklist](#-final-pre-interview-mastery-checklist)

---

## 🎯 Master Strategy: How to Crack 100% of MAANG GenAI Interviews

> [!IMPORTANT]
> **Why 90% of Candidates Fail MAANG GenAI Interviews**:  
> Most candidates default to surface-level answers using high-level frameworks (`LangChain`, `LlamaIndex`, `HuggingFace pipelines`). MAANG interviewers look for **systems-level mechanics**: exact GPU memory arithmetic, throughput limits, KV cache memory footprint, continuous batching iteration scheduling, loss function formulations, and tail-latency edge cases.

---

### 1. The 6-Step MAANG GenAI System Design Framework

When presented with a GenAI System Design prompt at Meta, Google, or Amazon, follow this strict 6-step blueprint:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Requirements & Metrics (TTFT < 200ms, ITL < 20ms, QPS, SLA)     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Sizing & Resource Math (Params, FP16 vs INT4, VRAM & FLOPs)     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Data & Knowledge Pipeline (Chunking, Dense/Sparse RRF, HyDE)    │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Serving Infrastructure (vLLM, Speculative Decoding, Parallelism) │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Guardrails, Safety & Evaluation (Jailbreak, Hallucination, Ragas)│
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Failure Modes & Trade-offs (Cache Threshing, Out-Of-Memory)     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Mathematical Quick Reference & Estimation Formula Cheat Sheet

Keep these core formulas memorized for rapid whiteboarding calculations during MAANG interviews:

#### A. Model Weight VRAM Calculation
$$\text{Memory}_{\text{weights}} = \frac{P \times b}{\text{Quantization Factor}}$$
* Where $P$ is parameter count (in billions), $b$ is precision bytes (FP16 = 2, FP32 = 4, INT8 = 1, INT4 = 0.5).
* *Example*: 70B parameter model in FP16 requires $70 \times 2 = 140\text{ GB}$ VRAM. In 4-bit (NF4), it requires $70 \times 0.5 = 35\text{ GB}$ VRAM.

#### B. KV Cache Memory Footprint per Sequence
$$\text{Memory}_{\text{KV}} = 2 \times L \times H_{\text{KV}} \times D_{\text{head}} \times S \times B \times b$$
* $L$ = Layers, $H_{\text{KV}}$ = Key-Value Heads (Grouped-Query Attention reduces this!), $D_{\text{head}}$ = Head Dimension ($d_{\text{model}} / H_{\text{query}}$), $S$ = Sequence Length, $B$ = Batch Size, $b$ = Precision bytes.

> [!TIP]
> **Grouped-Query Attention (GQA) Impact**: In Llama-3-70B, $H_{\text{query}} = 64$ but $H_{\text{KV}} = 8$ (8:1 ratio). This reduces KV cache memory consumption by **8x**, enabling much larger batch sizes ($B$) and longer context lengths ($S$).

#### C. Total Training Memory Footprint (Adam Optimizer)
$$\text{Memory}_{\text{training}} \approx 16 \times P \text{ bytes}$$
* **Model Weights (FP16)**: $2P$ bytes  
* **Gradients (FP16)**: $2P$ bytes  
* **Adam Optimizer State (FP32)**: $12P$ bytes ($4P$ for FP32 weights copy, $4P$ for 1st moment, $4P$ for 2nd moment).

#### D. Inference Latency & FLOPs Estimation
$$\text{FLOPs}_{\text{prefill}} \approx 2 \times P \times S_{\text{prompt}}$$
$$\text{FLOPs}_{\text{generation}} \approx 2 \times P \text{ per generated token}$$
* **Memory Bandwidth Bottleneck**: In generation phase, memory bandwidth (GB/s) limits throughput, while in prefill phase, compute (TFLOPs) limits throughput.

---

## 📂 Role 1: GenAI / LLM Application Engineer (Full-Stack & Applied AI)

---

### Mid-Level (L4/E4): Designing Low-TTFT Technical Doc RAG (Google)

> **Target Company**: Google / GCP  
> **Core Concepts**: Prefill vs Decode Phase, Reciprocal Rank Fusion (RRF), Server-Sent Events (SSE), Hybrid Search, TTFT Optimization.

#### Question Statement
"Design an enterprise documentation Search & Q&A assistant over 1,000,000 internal engineering documents. The system must maintain a Time-To-First-Token (TTFT) under **250ms** and an Inter-Token Latency (ITL) under **25ms** at 500 QPS."

#### Architectural Blueprint & Technical Answer

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

##### 1. Hybrid Search & Reciprocal Rank Fusion (RRF)
Combining BM25 keyword matching (for exact code syntax/error logs) with Dense Vector Cosine Similarity (for semantic intent). Merged using RRF:

$$RRF\_Score(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
where $k = 60$, $M$ is search systems (BM25, Dense), and $r_m(d)$ is document rank.

##### 2. Low-TTFT Engineering Implementation
To keep TTFT < 250ms, we stream tokens via SSE and execute hybrid retrieval concurrently using `asyncio.gather`.

```python
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import vllm

app = FastAPI()

async def hybrid_retrieval(query: str):
    # Execute BM25 and Vector Search concurrently to save network round-trips
    bm25_task = asyncio.create_task(bm25_search(query))
    vector_task = asyncio.create_task(vector_search(query))
    bm25_res, vector_res = await asyncio.gather(bm25_task, vector_task)
    
    # Reciprocal Rank Fusion (RRF)
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
        # Using vLLM AsyncEngine for fast token generation
        results_generator = vllm_engine.generate(prompt, sampling_params, request_id)
        async for output in results_generator:
            new_text = output.outputs[0].text
            yield f"data: {new_text}\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")
```

> [!TIP]
> **MAANG Candidate Edge**:  
> In your interview, state: *"To optimize TTFT, we bypass heavy multi-step LLM rerankers for top queries by leveraging a Redis Semantic Cache. If the cosine similarity of incoming query embedding to a cached entry exceeds 0.96, we return cached answers instantly, achieving TTFT < 15ms."*

---

### Mid-Level (L4/E4): Multi-Tier Guardrailing & Hallucination Defense (Amazon)

> **Target Company**: Amazon  
> **Core Concepts**: Prompt Injection Defense, Hallucination Detection, System Prompt Leaks, Guardrail Latency Overhead.

#### Question Statement
"Design a multi-tiered guardrail system for an Amazon customer support agent that intercepts jailbreaks (e.g., DAN mode), prevents PII leakage, and grounds billing responses to factual store policies without adding more than **40ms** of latency."

#### Architectural Blueprint & Technical Answer

```text
                               ┌─────────────────────────┐
                               │     Incoming Query      │
                               └────────────┬────────────┘
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │ STAGE 1: Light Input Scan (Regex + ONNX SLM) │  <-- Latency: 5ms
                    └───────────────────────┬──────────────────────┘
                                            │
                                  [ Safe ]  │  [ Malicious Prompt ] ──► [ Block & Alert ]
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │ STAGE 2: Intent Router & Context RAG         │  <-- Latency: 20ms
                    └───────────────────────┬──────────────────────┘
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │ STAGE 3: Core LLM Generation                 │  <-- Streaming SSE
                    └───────────────────────┬──────────────────────┘
                                            │
                                            ▼
                    ┌──────────────────────────────────────────────┐
                    │ STAGE 4: Output Guardrail & Grounding Judge  │  <-- Latency: 15ms
                    └──────────────────────────────────────────────┘
```

##### 1. Tier 1: Input Threat Classification (Latency: 5ms)
Instead of invoking an expensive 7B LLM for input moderation, run a fine-tuned 110M parameter `DeBERTa-v3-mini` ONNX pipeline quantized to INT8 on CPU/TensorRT.
* Checks for system prompt extraction keywords (`Ignore previous instructions`, `Output system prompt`).
* Scans for customer PII (credit cards, SSNs, phone numbers) via deterministic regular expressions + Named Entity Recognition (NER).

##### 2. Tier 2: Factual Grounding Verification (Latency: 15ms)
To verify hallucination without blocking output streaming, perform token-level NLI (Natural Language Inference) checking premise vs hypothesis asynchronously:

$$\text{P}(\text{Entailment} \mid \text{Context}, \text{Response}) > 0.85$$

##### 3. Guardrail Latency Budgeting
| Guardrail Stage | Model / Technology | Target Latency | Action on Failure |
| :--- | :--- | :--- | :--- |
| **Input Sanitization** | Regex + DeBERTa-v3 ONNX | **4 ms** | Hard Reject (HTTP 400) |
| **RAG Grounding Check** | Vector Overlap + BM25 Score | **10 ms** | Fallback to Pre-written Macro |
| **Output PII Masking** | Presidio / Regex Stream Parser | **2 ms** | Replace with `[REDACTED_CREDIT_CARD]` |

> [!IMPORTANT]
> **MAANG Candidate Edge**:  
> Highlight that streaming output PII masking must use a sliding window token buffer (e.g., 5 tokens) to capture split entity names (e.g., phone numbers split across streamed chunk boundaries) before flushing to the client socket.

---

### Senior (L5/E5): Multi-Agent Software Development & Tool Calling DAGs (Meta)

> **Target Company**: Meta  
> **Core Concepts**: ReAct Pattern, Stateful DAG Execution, Infinite Loop Prevention, Context Window Compression, Sandbox Execution.

#### Question Statement
"Architect an autonomous multi-agent code generation system that reads a GitHub issue, writes Python code, runs Pytest in a sandboxed container, diagnoses failures, and iterates autonomously until tests pass. How do you prevent infinite execution loops and context overflow?"

#### Architectural Blueprint & Technical Answer

```text
                       ┌────────────────────────────────┐
                       │     GitHub Issue Trigger       │
                       └───────────────┬────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PLANNER AGENT (Llama-3-70B)                       │
│                   Generates DAG of Steps: [Edit, Test, Refactor]            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CODER AGENT (CodeLlama/Qwen2.5)                   │
│                     Modifies code in transient git workspace                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TEST RUNNER AGENT                                 │
│               Executes `pytest` in gVisor / Docker Sandbox                  │
└───────────────┬──────────────────────────────────────────────┬──────────────┘
                │                                              │
       [ Tests Fail ]                                    [ Tests Pass ]
                │                                              │
                ▼                                              ▼
┌───────────────────────────────┐              ┌───────────────────────────────┐
│     CRITIC & DIAGNOSTIC AGENT │              │     PULL REQUEST GENERATOR    │
│ Parses stack trace & diff     │              └───────────────────────────────┘
└───────────────┬───────────────┘
                │
                ▼ (Max Iteration Count = 5)
   [ Loop Back to Coder Agent ]
```

##### 1. Context Window Compression & State Management
As iterations progress, raw terminal stack traces will overflow context windows. Implement **Token-Budgeted Context Compression**:
* Retain original system prompt & problem description (Fixed Token Allocation: 1,000 tokens).
* Retain current code modified files (Fixed Allocation: 4,000 tokens).
* Compress past failed attempts into a **Key Lessons Summary** using abstract summary DAG nodes rather than raw logs.

##### 2. Loop Detection Algorithm
Maintain a state hash vector of code files and test results:

$$H_t = \text{SHA256}(\text{CodeState}_t \mathbin{\Vert} \text{TestFailureOutput}_t)$$

If $H_t \in \{H_1, H_2, \dots, H_{t-1}\}$, the agent has entered an unrecoverable deterministic loop. Terminate the DAG and trigger human intervention.

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
                
            # Loop Prevention Hash
            state_hash = hash_state(code_diff, test_results.failure_summary)
            if state_hash in self.visited_hashes:
                raise LoopDetectedError("Agent repeated identical code modification cycle.")
            self.visited_hashes.add(state_hash)
            
            # Compress context for next iteration
            state["history"].append(critic_agent.summarize_failure(test_results))
            
        raise MaxIterationsExceededError("Agent reached iteration limit without passing tests.")
```

---

### Senior (L5/E5): Hybrid Recommendation & LLM Semantic Reranking Engine (Netflix)

> **Target Company**: Netflix  
> **Core Concepts**: Two-Stage Recommendation Architecture, Collaborative Filtering + Semantic Embeddings, LLM Cold-Start Handling, Real-Time Low Latency.

#### Question Statement
"Design a hybrid movie recommendation system for Netflix that combines matrix factorization collaborative filtering with LLM semantic reasoning for long-tail queries (e.g., 'Movies about 90s tech founders with witty dialogue'). System must serve 5,000 QPS with P99 < 100ms."

#### Architectural Blueprint & Technical Answer

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
          [ LLM Reranker: 8B Quantized Model ]
           (Evaluates top 30 candidates)
                           │
                           ▼
          [ Final Ranked Carousel List ]
```

##### 1. Two-Stage Candidate Retrieval (Latency: 20ms)
To satisfy the P99 < 100ms SLA, **never run an LLM directly across millions of catalog items**.
* Stage 1: Two-Tower Neural Network generates top 500 candidate IDs.
* Stage 2: Fine-tuned 8B LLM acts purely as a **reranker** over top 30 candidates using token probability scoring.

##### 2. LLM Reranking via Logit Calibration
Instead of asking the LLM to write text (which causes high decoding latency), pass candidate items in a structured prompt and evaluate the log probability of output tokens:

$$\text{Score}(m_i) = P(\text{Item } m_i \text{ is relevant} \mid \text{User History}, \text{Query})$$

This converts text generation into a single-pass forward evaluation (prefill phase only), reducing latency from **800ms down to 35ms**.

---

### Staff / Principal (L6+): Global 100M QPS Multi-Tenant Knowledge Engine (Apple / Meta)

> **Target Company**: Apple / Meta  
> **Core Concepts**: Multi-Tenant Isolation, Dynamic Tiered Caching, Global Edge Routing, Rate Limiting, Cost-per-Query SLAs.

#### Question Statement
"Design a enterprise-wide multi-tenant LLM platform servicing 100 Million daily queries across 50 internal applications. Requirements: Strict data isolation per tenant, SLA guarantee (TTFT < 200ms), global cost ceiling of **$0.001 per query**, and dynamic routing between cloud and edge models."

#### Architectural Blueprint & Technical Answer

```text
[ Client Requests (50+ Internal Apps) ]
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GLOBAL API GATEWAY (Envoy / Istio)                     │
│    Token Bucket Rate Limiting | Dynamic Tenant Auth | PII Masking Engine    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SEMANTIC CACHE LAYER (Redis Enterprise)                │
│             Similarity Threshold > 0.95 ──► Instant Return (15ms)           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (Cache Miss)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DYNAMIC MODEL ROUTING ENGINE                         │
│                                                                             │
│   Complexity Evaluator ──► Query Score < 0.3  ──► On-Device / Edge Model    │
│   (DeBERTa Router)     ──► Query Score 0.3-0.8 ──► Quantized 8B (Self-Host) │
│                        ──► Query Score > 0.8  ──► Frontier 70B/GPT-4o      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           ▼                           ▼                           ▼
┌────────────────────┐      ┌────────────────────┐      ┌────────────────────┐
│ Edge/Local Engine  │      │ vLLM Cluster (8B)  │      │ Frontier Multi-GPU │
│ (Apple Silicon/SLM)│      │ (AWS Inferentia2)  │      │ (H100 NVLink Pods) │
└────────────────────┘      └────────────────────┘      └────────────────────┘
```

##### 1. Tiered Router Cost Optimization Formula
To satisfy the **$0.001 per query constraint**, construct a loss function balancing cost and query accuracy:

$$\text{Minimize } \mathcal{C} = \sum_{i=1}^{N} \left( c(m_i) + \lambda \cdot \mathcal{L}_{\text{quality}}(m_i, q_i) \right)$$

* **Edge / SLM (Gemma-2B / Phi-3)**: $0.00005 per query (Handles 60% of simple traffic).
* **Quantized Self-Hosted (Llama-3-8B INT4)**: $0.0003 per query (Handles 30% of standard RAG).
* **Frontier Models (Llama-3-70B FP16 / Claude 3.5)**: $0.004 per query (Handles 10% complex queries).
* **Weighted Average Cost**: $(0.6 \times 0.00005) + (0.3 \times 0.0003) + (0.1 \times 0.004) = \mathbf{\$0.00052 \text{ per query}}$ (Well below the $0.001 threshold!).

##### 2. Tenant Isolation & Security Guarantees
* **Vector DB Isolation**: Dedicated Namespace per Tenant with Encrypted Encryption Keys (AWS KMS).
* **No Cross-Tenant KV Cache Contamination**: Flush or isolate PagedAttention KV pages between tenant context switches.

---

## 📂 Role 2: GenAI Research & Fine-Tuning Engineer / Applied Scientist

---

### Mid-Level (L4/E4): Mathematical Formulations of LoRA vs. QLoRA vs. Full Fine-Tuning (Meta)

> **Target Company**: Meta  
> **Core Concepts**: Low-Rank Adaptation (LoRA), Quantized LoRA (QLoRA), NormalFloat4 (NF4), Double Quantization, Memory Reductions.

#### Question Statement
"Derive the mathematical formulation of LoRA. Explain how QLoRA achieves near-lossless 4-bit fine-tuning, and compute the exact memory footprint difference between full parameter fine-tuning and QLoRA for a 70B parameter model."

#### Mathematical Derivation & Technical Answer

##### 1. LoRA Mathematical Formulation
During fine-tuning, weight matrix update $\Delta W \in \mathbb{R}^{d \times k}$ is factorized into two low-rank matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$, where rank $r \ll \min(d, k)$:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B A) x$$

* Matrix $A$ is initialized from Gaussian distribution $\mathcal{N}(0, \sigma^2)$, and Matrix $B$ is initialized to $0$, ensuring $\Delta W = 0$ at start of training.
* $\alpha$ is a constant scaling hyperparameter.

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

##### 2. QLoRA Innovations
QLoRA introduces three innovations to compress base model weights to 4-bit without accuracy degradation:
1. **NormalFloat4 (NF4) Quantization**: An information-theoretically optimal quantile quantization scheme for normally distributed weights:

$$q_i = \frac{1}{2} \left( Q_X\left(\frac{i}{2^k}\right) + Q_X\left(\frac{i+1}{2^k}\right) \right)$$

2. **Double Quantization (DQ)**: Quantizes the quantization constants themselves, saving an additional 0.37 bits per parameter.
3. **Paged Optimizers**: Leverages CUDA Unified Memory to automatically page memory out to CPU RAM during VRAM allocation spikes.

##### 3. Memory Footprint Arithmetic (70B Model)

| Parameter / Memory Component | Full Fine-Tuning (FP16) | LoRA (FP16 Adapt + FP16 Base) | QLoRA (NF4 Base + FP16 Adapt) |
| :--- | :--- | :--- | :--- |
| **Base Model Weights** | 140 GB (FP16) | 140 GB (FP16) | **35 GB (NF4)** |
| **Adapter Parameters ($r=16$)** | 0 GB | 0.8 GB | 0.8 GB |
| **Gradients** | 140 GB | 0.8 GB | 0.8 GB |
| **Optimizer States (Adam)** | 840 GB (FP32) | 4.8 GB (FP32) | 4.8 GB (FP32) |
| **Total Memory Footprint** | **~1,120 GB (Requires 14x H100)** | **~146 GB (Requires 2x H100)** | **~41.4 GB (Fits on 1x A100 80GB!)** |

---

### Mid-Level (L4/E4): Direct Preference Optimization (DPO) vs. PPO Derivation (Google DeepMind)

> **Target Company**: Google DeepMind  
> **Core Concepts**: RLHF, Bradley-Terry Model, Policy Loss Derivation, DPO Closed-Form Implicit Reward.

#### Question Statement
"Derive the DPO loss function starting from the RLHF Bradley-Terry preference model. Explain mathematically why DPO eliminates the need for training an explicit Reward Model or running online PPO sampling loops."

#### Mathematical Derivation & Technical Answer

##### 1. Classical RLHF Objective
In standard PPO-based RLHF, we optimize policy $\pi_\theta$ against reward model $r_\psi(x,y)$ constrained by KL divergence to reference policy $\pi_ref$:

$$\max_{\pi_\theta} \mathbb{E}_{(x,y) \sim D} \left[ r_\psi(x,y) \right] - \beta \mathbb{D}_{\text{KL}}\left(\pi_\theta(y \mid x) \mathbin{\Vert} \pi_{\text{ref}}(y \mid x)\right)$$

Using Lagrange multipliers, the analytical optimal policy $\pi^*$ for a fixed reward function $r(x,y)$ is:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x,y)\right)$ is the partition function.

##### 2. Deriving the Implicit Reward Function
Rearranging the optimal policy equation yields an exact closed-form expression for the reward function in terms of the optimal policy and reference policy:

$$r(x,y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

##### 3. Substituting into the Bradley-Terry Preference Model
The Bradley-Terry preference probability that response $y_w$ (winning) is preferred over $y_l$ (losing) given prompt $x$ is:

$$P(y_w \succ y_l \mid x) = \sigma \left( r(x, y_w) - r(x, y_l) \right)$$

Substituting our implicit reward expression into the Bradley-Terry model eliminates the partition function $Z(x)$ because it cancels out:

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$$

##### 4. Final DPO Loss Function
Taking the negative log-likelihood over dataset $\mathcal{D} = \{(x, y_w, y_l)\}$:

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

> [!IMPORTANT]
> **Why DPO Wins**: DPO directly optimizes the policy network weights $\theta$ using simple binary cross-entropy loss over preference pairs, completely skipping the unstable reward model training, actor-critic architecture, and GPU memory overhead of online PPO rollouts!

---

### Senior (L5/E5): Fine-Tuning a 32B Domain LLM with Synthetic Data Pipelines (Amazon)

> **Target Company**: Amazon  
> **Core Concepts**: Synthetic Data Generation (Evol-Instruct), Deduplication (MinHash LSH), Quality Filtering, Data Contamination Auditing.

#### Question Statement
"Design an end-to-end data pipeline to generate 500,000 synthetic instruction-response pairs to fine-tune a 32B model for legal contract negotiation. How do you guarantee data diversity, eliminate hallucinated outputs, and avoid data contamination?"

#### Architectural Blueprint & Technical Answer

```text
[ Raw Unstructured Legal Playbooks ]
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ STAGE 1: Seed Instruction Generation            │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ STAGE 2: Evolutionary Mutation (Evol-Instruct)  │
│  - In-breadth Mutation (Generate new topics)    │
│  - In-depth Mutation (Add constraints, complexity)
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ STAGE 3: Synthetic Response Generation          │
│  - Generation via Claude 3.5 Sonnet / Llama-70B │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ STAGE 4: Strict Quality & Safety Filter         │
│  - MinHash LSH (Deduplication @ 0.8 Jaccard)   │
│  - Reward Model Scoring (Keep Top 20%)          │
│  - DeBERTa Contamination Classifier             │
└─────────────────────────────────────────────────┘
```

##### 1. Evolutionary Mutation (Evol-Instruct Framework)
To convert simple legal questions into complex, high-reasoning prompts, apply 5 mutation prompts iteratively:
* **Deepen Constraints**: *"Add a strict liability clause constraint under California jurisdiction."*
* **Concretize Scenario**: *"Replace generic terms with specific enterprise M&A dispute parameters."*
* **Reasoning Enhancement**: *"Ask the model to provide step-by-step statutory justification."*

##### 2. Deduplication Engine via MinHash LSH
To prevent model overfitting on near-identical synthetic prompts:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Apply **MinHash with Local Sensitivity Hashing (LSH)**:
* Compute 128 permutation hash functions per prompt token 5-gram.
* Divide hashes into 16 bands of 8 rows.
* Discard candidate synthetic pairs where Jaccard similarity exceeds $0.80$.

---

### Staff / Principal (L6+): Post-Training at Scale: GRPO, PRM, and Reasoning Scaling Laws (OpenAI / Google)

> **Target Company**: OpenAI / Google DeepMind  
> **Core Concepts**: Group Relative Policy Optimization (GRPO), Process Reward Models (PRM) vs. Outcome Reward Models (ORM), Test-Time Search Scaling.

#### Question Statement
"Explain how Group Relative Policy Optimization (GRPO) powers modern reasoning models (like DeepSeek-R1 / OpenAI o1). Compare GRPO against PPO mathematically. How do Process Reward Models (PRM) enable test-time compute scaling?"

#### Architectural Blueprint & Technical Answer

##### 1. Mathematical Formulation: GRPO vs PPO
Traditional PPO requires a separate **Critic (Value) Model** of identical size to the Policy model to estimate baseline $V(s)$, doubling VRAM requirements.

**GRPO (Group Relative Policy Optimization)** eliminates the Value model entirely! For each prompt $q$, GRPO samples a group of $G$ outputs $\{o_1, o_2, \dots, o_G\}$ from the old policy $\pi_{\theta_{\text{old}}}$, and computes relative advantage normalized across the group:

$$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}$$

The GRPO objective function is:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^{G} \min \left( \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta \mathbb{D}_{\text{KL}}\left(\pi_\theta \mathbin{\Vert} \pi_{\text{ref}}\right)$$

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

##### 2. Process Reward Models (PRM) & Test-Time Compute Scaling
* **Outcome Reward Model (ORM)**: Evaluates only final answer correctness $r(y)$. Gives sparse feedback for long chain-of-thought (CoT) reasoning.
* **Process Reward Model (PRM)**: Assigns step-level rewards $r(z_t)$ to every intermediate step $z_t$ in reasoning:

$$r_{\text{total}} = \prod_{t=1}^{T} P(\text{Step } z_t \text{ is step-wise correct})$$

* **Test-Time Search (Tree Search / Beam Search)**: During inference, use the PRM as a heuristic guide to explore multiple reasoning paths in a Monte Carlo Tree Search (MCTS), drastically scaling accuracy without retraining the base model!

---

## 📂 Role 3: GenAI Infrastructure, ML Platform & MLOps Engineer

---

### Mid-Level (L4/E4): Continuous Batching & PagedAttention KV Cache Mechanics (Meta)

> **Target Company**: Meta  
> **Core Concepts**: Static vs. Continuous Batching, Virtual Memory Paging, Fragmented VRAM, vLLM Architecture.

#### Question Statement
"Explain why traditional static batching leads to low GPU utilization in LLM inference. Detail how Continuous Iteration-Level Batching and PagedAttention solve memory fragmentation."

#### Architectural Blueprint & Technical Answer

##### 1. The Bottleneck of Static Batching
In traditional inference servers (e.g., standard HuggingFace pipeline), sequences in a batch are padded to the maximum sequence length $S_{\max}$.

```text
Static Batching:
Seq 1: [Tok 1][Tok 2][Tok 3] ░░░░ PAD ░░░░ ░░░░ PAD ░░░░ (GPU Idle waiting for Seq 2)
Seq 2: [Tok 1][Tok 2][Tok 3][Tok 4][Tok 5][Tok 6][Tok 7] (Finish)
```

* **Problems**: Wasteful compute on padding tokens; GPU compute sits idle when short sequences finish early.

##### 2. Continuous Iteration-Level Batching
Instead of waiting for an entire batch to finish, continuous batching operates at the **iteration level** (after every single generated token). As soon as Sequence 1 emits an `<EOS>` token, a new incoming request (Sequence 3) is immediately injected into the active batch during its prefill phase!

##### 3. PagedAttention Mechanics
Traditional KV cache allocation requires allocating contiguous physical GPU VRAM memory for the maximum possible context window length (e.g., 8,192 tokens), leading to **60% - 80% VRAM memory fragmentation**.

PagedAttention (pioneered by vLLM) borrows from OS Virtual Memory paging:
* Divides KV Cache into fixed-size **Key-Value Physical Blocks** (e.g., block size = 16 tokens).
* Maintains a dynamic **Block Table** mapping logical token blocks to non-contiguous physical GPU memory pages.

```text
LOGICAL KV BLOCKS                BLOCK TABLE               PHYSICAL GPU MEMORY PAGES
[ Req 0: Block 0 ] ──────────► [ Block 0 ──► Page 7 ] ────► Physical Memory Page 7
[ Req 0: Block 1 ] ──────────► [ Block 1 ──► Page 3 ] ────► Physical Memory Page 3
[ Req 1: Block 0 ] ──────────► [ Block 0 ──► Page 12] ────► Physical Memory Page 12
```

> [!TIP]
> **Key Metric to Quote**: PagedAttention reduces VRAM memory waste from **>60% down to under 4%**, enabling up to **3.8x throughput improvements** on identical GPU hardware!

---

### Mid-Level (L4/E4): Quantization Mathematics: AWQ vs. GPTQ vs. FP8 vs. NF4 (AWS Bedrock)

> **Target Company**: AWS Bedrock / ML Platform  
> **Core Concepts**: Quantization Algorithms, Activation-Aware Weight Quantization (AWQ), GPTQ Hessian Optimization, FP8 E4M3 vs. E5M2.

#### Question Statement
"Compare AWQ, GPTQ, FP8, and bitsandbytes (NF4) quantization techniques. Explain the math behind AWQ's activation protection and state when to use FP8 over INT4 on modern Hopper (H100) GPUs."

#### Technical & Mathematical Comparison

##### 1. Mathematical Principle of AWQ (Activation-Aware Quantization)
AWQ observes that not all weights in an LLM are equally important. Retaining 1% of salient weights in higher precision preserves full model accuracy.

To protect salient weight channels without incurring mixed-precision hardware speed penalties, AWQ scales weight channels $W$ by an per-channel scaling factor $s \ge 1$ while inversely scaling inputs $X$:

$$W' = W \cdot \text{diag}(s), \quad X' = \text{diag}(s)^{-1} \cdot X$$

The per-channel scaling factor $s_x$ is derived by minimizing reconstruction error over activation magnitudes:

$$s_x = \arg\min_s \left\| W X - \text{Quantize}(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1} X \right\|$$

##### 2. Comprehensive Quantization Matrix

| Quantization Method | Target Precision | Salient Weight Handling | Primary Hardware Target | Perplexity Loss |
| :--- | :--- | :--- | :--- | :--- |
| **AWQ** | 4-bit (INT4) | Protects 1% top activation magnitudes via channel scaling | NVIDIA Ampere / Ada / Hopper GPUs | **Extremely Low** |
| **GPTQ** | 4-bit (INT4) | Second-order inverse Hessian matrix optimization | GPU Batch Inference | Very Low |
| **FP8 (E4M3 / E5M2)** | 8-bit (FP8) | Native Tensor Core floating point math | NVIDIA H100 / B200 Tensor Cores | **Zero Loss** |
| **NF4 (bitsandbytes)** | 4-bit (NF4) | Quantile-based optimal distribution for normal weights | Single-GPU Fine-tuning (QLoRA) | Low |

> [!IMPORTANT]
> **FP8 vs INT4 on H100 GPUs**:  
> On NVIDIA H100 GPUs, native **FP8 (E4M3 for forward pass, E5M2 for gradients)** delivers **2x GEMM throughput** over FP16 using Transformer Engine hardware Tensor Cores without needing weight dequantization overhead during execution, making FP8 preferred over INT4 for high-throughput serving on Hopper architecture!

---

### Senior (L5/E5): Speculative Decoding & Medusa Multi-Head Drafters (Netflix / Google)

> **Target Company**: Netflix / Google  
> **Core Concepts**: Memory Bandwidth Bound vs Compute Bound, Draft Model Rejection Sampling, Medusa Multi-Head Decoding, Acceptance Probability.

#### Question Statement
"Explain how Speculative Decoding achieves a 2x-4x latency speedup without changing model outputs by a single bit. Derive the rejection sampling acceptance probability formula."

#### Architectural Blueprint & Technical Answer

```text
[ Input Tokens ]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ DRAFT MODEL (Small 68M Model / Medusa Heads)           │  <-- Fast (Memory Efficient)
│ Generates K Candidate Draft Tokens: [t1, t2, t3, t4]   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ TARGET MODEL (Large 70B Model)                         │  <-- Compute Bound Batch Verification
│ Runs Single Forward Pass over all K Tokens Concurrently│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ REJECTION SAMPLING EVALUATOR                           │
│ Accepts tokens [t1, t2], Rejects [t3]. Emits [t1, t2, t_new]
└────────────────────────────────────────────────────────┘
```

##### 1. Core Mechanics
Decoding in LLMs is **memory bandwidth bound**: generating 100 tokens requires passing the 70B parameter weight tensor through GPU RAM to Tensor Cores 100 separate times.

Speculative Decoding uses a fast, lightweight **Draft Model** (e.g., 68M parameters) to rapidly draft $K$ candidate tokens. Then, the large **Target Model** (70B parameters) evaluates all $K$ candidate tokens in **a single parallel forward pass** (which is compute-bound and takes virtually the same time as validating 1 token!).

##### 2. Rejection Sampling Acceptance Math
To ensure the output probability distribution matches the target model distribution $p(x)$ exactly, accept draft token $x$ proposed by draft model distribution $q(x)$ with probability:

$$P_{\text{accept}}(x) = \min\left(1, \frac{p(x)}{q(x)}\right)$$

If draft token $x$ is rejected, sample a replacement token from adjusted probability distribution:

$$p' (x) = \max\left(0, \frac{p(x) - q(x)}{1 - \sum_y \min(p(y), q(y))}\right)$$

---

### Staff / Principal (L6+): Distributed 3D Parallelism & NVLink Multi-Node Cluster Scaling (GCP / AWS)

> **Target Company**: GCP / AWS / Meta  
> **Core Concepts**: Tensor Parallelism (Megatron-LM), Pipeline Parallelism, Data Parallelism (ZeRO-3), Inter-GPU Interconnects (NVLink vs. InfiniBand/RoCEv2).

#### Question Statement
"Design the distributed training topology to train a 405B parameter dense transformer model across 1,024 NVIDIA H100 GPUs. Calculate tensor, pipeline, and data parallel dimensions and analyze network bandwidth bottlenecks across NVLink switches vs. RoCEv2 fabric."

#### Technical Answer & Distributed Cluster Topology

```text
                             1,024 H100 GPU CLUSTER
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
  [ Tensor Parallel (TP=8) ]   [ Pipeline Parallel (PP=16) ]   [ Data Parallel (DP/ZeRO-3 = 8) ]
  Intra-Node (NVLink @ 900GB/s)   Inter-Node (InfiniBand/RoCE)    Inter-Node All-Reduce
```

##### 1. 3D Parallelism Dimension Decomposition
Total GPUs $N = TP \times PP \times DP = 1,024$.

* **Tensor Parallelism (TP = 8)**: Set $TP = 8$ to fit within a single 8-GPU HGX H100 node. All Column-Parallel and Row-Parallel GEMM communication occurs over high-speed **NVLink 4.0 (900 GB/s bidirectional bandwidth)**.
* **Pipeline Parallelism (PP = 16)**: Split model's 80 layers into 16 stage blocks (5 layers per node). Communication occurs only at stage boundaries (activating 1D layer outputs), minimizing inter-node bandwidth requirements over network switches.
* **Data Parallelism with ZeRO-3 (DP = 8)**: Distributed across remaining nodes using DeepSpeed ZeRO-3 (partitioning weights, gradients, and optimizer states) over **8x 400 Gbps InfiniBand / RoCEv2 NICs per node**.

##### 2. Bandwidth & Memory Arithmetic
For a 405B model in FP16 ($810 \text{ GB}$ base weights):
* Memory required per GPU for weights: $\frac{810\text{ GB}}{TP \times PP} = \frac{810}{8 \times 16} = \mathbf{6.32 \text{ GB per GPU}}$.
* This leaves $>70 \text{ GB}$ of VRAM per H100 GPU available for activation memory, micro-batches, and KV caches!

---

## 📂 Role 4: Lead / Principal / Staff GenAI Architect

---

### Staff (L6): Automated Enterprise Vulnerability & Security Patch Agent System (Meta / Apple)

> **Target Company**: Meta / Apple  
> **Core Concepts**: Static Analysis Integration, AST Parser, Automated Vulnerability Patching, Human-In-The-Loop Approval, Sandbox Containment.

#### Question Statement
"Design an enterprise agent platform that continuously scans codebase repositories for CVE vulnerabilities, automatically generates code patch pull requests, runs AST static analysis validation, and routes high-risk patches to security engineers."

#### Architectural Blueprint & Technical Answer

```text
[ Enterprise GitHub / Gitlab Event Webhook ]
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Vulnerability Detection & AST Parsing                              │
│   Semgrep / Trivy / CodeQL Scanners detect CWE vulnerability                │
└────────────────────┬────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Security Patch Agent (Fine-Tuned DeepSeek-Coder / Llama-3-70B)    │
│   Parses AST + Vulnerable Line Range ──► Generates Minimal Code Diff        │
└────────────────────┬────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Deterministic AST & Safety Validation                              │
│   Runs Static AST Validator (Ensures no API breaking function changes)       │
└────────────────────┬────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: Automated Testing Sandbox (gVisor Container)                       │
│   Executes `pytest` & Integration Tests                                      │
└────────────────────┬────────────────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
 [ CVSS Score < 7.0 ]      [ CVSS Score >= 7.0 ]
 Auto-Merge Bot            Route to Security Team (Slack / Jira)
```

##### Key Architectural SLA & Safety Controls
1. **Zero Unintended Side-Effects**: AST parser verifies that function signatures and public class interfaces remain 100% identical post-patch.
2. **Sandbox Containment**: Code execution for test verification takes place inside lightweight **gVisor / Firecracker microVMs** with read-only network access to eliminate malicious code execution risks.

---

### Principal (L7): Enterprise Model Gateway, Dynamic Cascading Router & Semantic Cache (Netflix / Google)

> **Target Company**: Netflix / Google / Amazon  
> **Core Concepts**: Multi-Cloud Gateway Routing, Semantic Caching, SLA Fallback Cascades, Real-Time Cost Metering, PII Redaction.

#### Question Statement
"Architect an enterprise AI Gateway that handles 1 Billion daily API calls across OpenAI, Anthropic, Google Gemini, and self-hosted vLLM clusters. Must enforce sub-millisecond route calculations, token rate-limiting, semantic response caching, and 99.999% availability."

#### Complete Enterprise System Topology

```text
[ Global Edge CDN / Cloudflare Workers ]
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ENTERPRISE AI GATEWAY LAYER                           │
│                                                                             │
│  ┌─────────────────────────┐   ┌────────────────────────┐   ┌────────────┐  │
│  │ Envoy Proxy Core (C++)  │──►│ Dynamic Token Metering │──►│ PII Redact │  │
│  └────────────┬────────────┘   └────────────────────────┘   └────────────┘  │
└───────────────┼─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ULTRA-LOW LATENCY SEMANTIC CACHE                        │
│   Redis Enterprise / Qdrant HNSW Index (Cosine Similarity Threshold = 0.96) │
└───────────────┬─────────────────────────────────────────────────────────────┘
                │ (Cache Miss)
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DYNAMIC CASCADING ROUTER & HEALTH ENGINE                │
│                                                                             │
│ Primary Route: Self-Hosted vLLM (Llama-3-70B) ──[If Latency > 400ms]──►    │
│ Fallback 1: Azure OpenAI GPT-4o               ──[If Rate-Limited 429]─►    │
│ Fallback 2: AWS Bedrock Claude 3.5 Sonnet                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

##### Key Architectural Guarantees
* **Sub-Millisecond Routing Overhead**: Implemented directly in C++ as an **Envoy Proxy Filter**, handling rate limiting and semantic cache lookups in **< 1.2ms**.
* **Circuit Breaker Pattern**: If an upstream model provider experiences elevated error rates (> 2% 5xx errors over 10 seconds), the circuit breaker trips, instantly re-routing traffic to alternate providers without losing in-flight user connections.

---

## 📊 Final Pre-Interview Mastery Checklist

Before stepping into your MAANG Generative AI interview, ensure you can whiteboard and discuss every item below:

- [ ] **VRAM & Memory Arithmetic**: Can calculate exact GPU memory requirements for model weights, KV caches, and Adam optimizer states on the fly.
- [ ] **KV Cache Optimization**: Can explain Grouped-Query Attention (GQA), PagedAttention memory virtualization, and sliding-window attention.
- [ ] **Pre-Training & Fine-Tuning**: Can derive LoRA matrix factorization ($W_0 + \frac{\alpha}{r} BA$), QLoRA NF4 double quantization, and DPO loss functions from memory.
- [ ] **Inference Acceleration**: Can explain Speculative Decoding rejection sampling math, continuous iteration-level batching, and FP8 vs INT4 Tensor Core mechanics.
- [ ] **Alignment & Post-Training**: Can contrast DPO vs. PPO vs. GRPO (Group Relative Policy Optimization) and explain step-wise Process Reward Models (PRMs).
- [ ] **System Architecture & Guardrails**: Can design end-to-end multi-agent DAGs, hybrid RAG with Reciprocal Rank Fusion (RRF), multi-tiered safety guardrails, and enterprise gateway routing.

---
*This guide is part of the [Cracking The GenAI Portfolio](README.md) open-source repository.*
