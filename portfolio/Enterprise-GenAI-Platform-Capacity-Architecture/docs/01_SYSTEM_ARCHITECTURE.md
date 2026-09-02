# 01. Enterprise Multi-Tenant AI Platform Architecture

## Executive Overview
This document specifies the target architecture for a centralized, multi-tenant Generative AI control and data plane engineered for an enterprise conglomerate operating across diverse subsidiaries (e.g., manufacturing, infrastructure, consumer services, and internal enterprise functions).

Rather than permitting fragmented, shadow AI deployments with siloed LLM wrappers, this platform provides a unified **`/v1/chat` Inference Gateway**, a dynamic **Bot Registry**, semantic **Model Routing**, and **Air-Gapped Tool Execution**.

---

## 🏛️ System Architecture Topology

```mermaid
graph TD
    subgraph Client Layer
        A1[Mobile & Web Apps]
        A2[Internal ERP / CRM Portals]
        A3[Automated Batch Pipelines]
    end

    subgraph Edge & API Gateway
        B1[Cloudflare / Envoy API Gateway]
        B2[OAuth2 / OIDC / Mutual TLS Auth]
        B3[Tenant Rate Limiting & WAF]
    end

    subgraph AI Control Plane
        C1[Unified /v1/chat Ingestion]
        C2[Bot Registry & Policy Engine]
        C3[Pre-Execution Guardrails & PII Scanners]
        C4[Dynamic Semantic Model Router]
    end

    subgraph Data & Retrieval Plane
        D1[Hybrid RAG Engine: Dense + BM25]
        D2[Reciprocal Rank Fusion RRF]
        D3[Document ACL & Tenant Namespace Filter]
        D4[Cross-Encoder Reranker]
    end

    subgraph Inference Compute Tier
        E1[SLM Cluster: 8B FP8 - Low Complexity]
        E2[Frontier Cluster: Claude-X 70B FP8 - Deep Reasoning]
        E3[Dedicated Embedding & OCR Cluster]
    end

    subgraph Secure Enterprise Tool Execution
        F1[Air-Gapped Tool Gateway]
        F2[ERP Telemetry & Inventory APIs]
        F3[Human-in-the-Loop Approval Queue]
    end

    A1 & A2 & A3 --> B1
    B1 --> B2 --> B3 --> C1
    C1 --> C2 --> C3 --> C4
    C4 -->|Context Retrieval| D1 --> D2 --> D3 --> D4 --> C4
    C4 -->|70% Workload| E1
    C4 -->|30% Reasoning| E2
    C4 -->|Live System Tools| F1
    F1 --> F2
    F1 -->|Dangerous Write Ops| F3
```

---

## 🎯 Key Architectural Subsystems

### 1. Unified `/v1/chat` Gateway & Bot Registry
* **Single Contract**: All enterprise client applications interact with standard OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/embeddings`).
* **Bot Registry (`/v1/bots`)**: Central metadata catalog containing:
  - System prompts and domain instructions
  - Associated vector knowledge bases (`kb_id`s)
  - Authorized tools (`allowed_tools`, `restricted_tools`)
  - Target model routing policy (`dynamic_routed`, `force_frontier`, `slm_only`)
  - RBAC role requirements (`required_role: plant_engineer`)

### 2. Tiered Model Routing (SLM vs Frontier)
Sending 100% of enterprise queries to frontier reasoning models causes runaway costs and latency spikes.
* **Tier 1 — Small Language Models (SLMs, 8B FP8)**:
  - Handles ~70% of enterprise queries (FAQ lookup, entity extraction, summarization, metadata tagging, JSON schema formatting).
  - TTFT: ~250–350ms, Cost: ~$0.0002 / 1K tokens.
* **Tier 2 — Frontier Models (Claude-X / 70B+ FP8)**:
  - Handles ~30% complex analytical workloads (root-cause diagnostics, multi-hop legal analysis, code generation, financial synthesis).
  - TTFT: ~800–1000ms, Cost: ~$0.0035 / 1K tokens.

### 3. Server-Sent Events (SSE) Streaming & Connection Resilience
* Real-time token streaming over HTTP/2 SSE connections.
* **Client Disconnect Handling**: Immediate termination of inference context via ASGI disconnect listeners to prevent GPU compute waste on abandoned sessions.
* **Semantic Response Caching**: Redis-backed cache for identical enterprise queries, bypassing LLM inference entirely for static corporate policies.

### 4. Failure Domain Isolation & High Availability
* **GPU Worker Health Probes**: Real-time monitoring of vLLM worker health. Unhealthy worker nodes are evicted within 5 seconds.
* **Fallback Degradation**: If the Frontier cluster experiences transient saturation or failover, requests automatically degrade to an optimized SLM with explicit warning metadata in the response headers.
* **$N+1$ Redundancy**: Minimum 1 spare serving replica provisioned per cluster group to guarantee 99.9%+ availability during maintenance and node drains.
