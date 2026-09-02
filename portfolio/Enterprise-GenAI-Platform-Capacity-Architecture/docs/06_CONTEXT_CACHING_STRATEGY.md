# 06. Context Caching Strategy — Prefix KV-Cache & Cross-Model Handoff

## Overview

Context caching is one of the highest-leverage latency and cost optimisations available
in production LLM serving. This document specifies the **three-layer caching hierarchy**
implemented in the Enterprise GenAI Platform, covering:

1. **Layer 1 — Semantic Response Cache** (Redis): Bypass inference entirely for repeated
   queries.
2. **Layer 2 — Prefix KV-Cache**: Reuse the transformer's key-value attention states for
   shared system prompt prefixes across concurrent enterprise requests.
3. **Layer 3 — Cross-Model Context Handoff**: Compress multi-turn conversation state when
   a request escalates from the SLM tier to the Frontier tier mid-session.

---

## Background: Why KV-Cache Prefix Caching Matters

Every LLM inference call involves computing attention key-value pairs for every token in
the input context. For enterprise bots, the **system prompt** (role definition, policy
instructions, knowledge base context) can be **500–2,000 tokens long** and is **identical
across every user request to the same bot**.

Without prefix caching, each request pays the full cost of processing those shared tokens:

```
Request A:  [SYSTEM_PROMPT_1800_TOKENS] + [USER_QUERY_50_TOKENS]  → 1850 tokens billed
Request B:  [SYSTEM_PROMPT_1800_TOKENS] + [USER_QUERY_65_TOKENS]  → 1865 tokens billed
Request C:  [SYSTEM_PROMPT_1800_TOKENS] + [USER_QUERY_40_TOKENS]  → 1840 tokens billed
```

With prefix caching, only the **incremental user query tokens** are computed after the
first request. Subsequent requests hit the cached KV states:

```
Request A:  [SYSTEM_PROMPT_1800_TOKENS] + [USER_QUERY_50_TOKENS]  → 1850 tokens (cold)
Request B:  [CACHE_HIT: 1800 tokens]   + [USER_QUERY_65_TOKENS]  →   65 tokens billed ✅
Request C:  [CACHE_HIT: 1800 tokens]   + [USER_QUERY_40_TOKENS]  →   40 tokens billed ✅
```

This is the mechanism behind **Anthropic's Prompt Caching** (90% discount on cached
tokens), **Google's Context Caching API** (75% cost reduction), and **vLLM's
`enable_prefix_caching=True`** flag for self-hosted deployments.

---

## Layer 1 — Semantic Response Cache (Existing)

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Semantic Response Cache (Redis)               │
│                                                         │
│  Key:   SHA-256(bot_id + normalised_query_embedding)    │
│  Value: Full serialised response JSON                   │
│  TTL:   Configurable per bot (default: 1 hour)         │
│                                                         │
│  Use case: Identical or near-identical FAQ queries,     │
│  corporate policy lookups, static reference data.       │
│  Cache Hit → Zero LLM inference cost.                   │
└─────────────────────────────────────────────────────────┘
```

**Hit Rate**: ~15–25% across enterprise workloads (high for HR/policy bots, low for
analytical bots).

---

## Layer 2 — Prefix KV-Cache (New)

```
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Prefix KV-Cache                               │
│                                                         │
│  Key:   SHA-256(model_name + system_prompt_text)        │
│  Value: KV state reference handle (e.g. vLLM block ID)  │
│  TTL:   10 minutes (sliding window, reset on hit)       │
│                                                         │
│  Use case: All requests sharing the same bot system     │
│  prompt. After first request, cached prefix KV states   │
│  are reused — only the user query tokens are computed.  │
└─────────────────────────────────────────────────────────┘
```

### Implementation Architecture

```
Incoming Request
      │
      ▼
┌─────────────────────────────┐
│ ContextCacheManager         │
│  .get_or_create_prefix_     │
│   entry(system_prompt)      │
└──────────┬──────────────────┘
           │
     ┌─────┴─────┐
     │ Cache Hit? │
     └─────┬─────┘
           │
     ┌─────▼────────────────┐      ┌─────────────────────────────────┐
     │  HIT                 │      │  MISS                           │
     │  Reuse KV block ref  │      │  Full prefill of system prompt  │
     │  Process user tokens │      │  Store new KV entry in cache    │
     │  only (delta tokens) │      │  Process full context           │
     └─────────────────────┘      └─────────────────────────────────┘
```

### Cost Impact Model (Anthropic Pricing Example)

| Metric | Without Prefix Cache | With Prefix Cache |
| :--- | :--- | :--- |
| System prompt tokens | 1,800 | 1,800 (first req only) |
| Per-request billed tokens | 1,850 avg | 65 avg (user query) |
| Cost per 1K input tokens | $0.003 | $0.0003 (cached tier) |
| Cost/request (steady state) | $0.00555 | $0.0000195 |
| **Savings at 10K req/day** | — | **~$54/day / $19,700/yr** per bot |

At 50 enterprise bots with active caching: **~$985K/yr potential savings** (varies by
system prompt length and traffic).

### Cache Invalidation Policy

Prefix cache entries are invalidated when:
- **TTL expires** (default: 10 minutes sliding window)
- **System prompt changes** (content hash mismatch → automatic eviction)
- **Model version upgrades** (model_name is part of the cache key)
- **Manual admin flush** via `DELETE /v1/cache/prefixes/{bot_id}`

---

## Layer 3 — Cross-Model Context Handoff (New)

When the Dynamic Model Router escalates a multi-turn conversation from the **SLM tier**
to the **Frontier tier** (e.g., the user's 4th message is a complex analytical request),
the Frontier model must process the full prior conversation history cold — unless a
handoff mechanism is in place.

```
Turn 1: [User: simple FAQ]     → Routed to SLM-8B ✅
Turn 2: [User: follow-up]      → Routed to SLM-8B ✅
Turn 3: [User: complex analysis] → Escalate to Frontier-70B
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │  Context Handoff Engine       │
                          │                               │
                          │  1. Retrieve SLM conversation │
                          │     history (turns 1–2)       │
                          │  2. Compress via SLM summary  │
                          │     (<100 tokens)             │
                          │  3. Prepend compressed summary│
                          │     to Frontier context       │
                          │  4. Route turn 3 to Frontier  │
                          └───────────────────────────────┘
```

### Handoff Compression Format

```json
{
  "handoff_summary": "User asked about leave balance (answered: 12 days remaining) and payslip download link (answered: portal.company.com/payslips). Now requesting complex tax liability analysis.",
  "handoff_tokens": 48,
  "original_turns_compressed": 2,
  "original_tokens_saved": 312
}
```

This avoids re-processing full history tokens through the expensive Frontier model while
maintaining conversational coherence.

---

## Observability & Cache Metrics

Every `/v1/chat/completions` response includes cache metadata in the response body:

```json
{
  "cache_metadata": {
    "layer1_semantic": "MISS",
    "layer2_prefix": "HIT",
    "prefix_tokens_saved": 1800,
    "estimated_cost_saved_usd": 0.000054,
    "layer3_handoff_active": false
  }
}
```

Platform-level aggregated metrics exposed via `/v1/cache/stats`:

```json
{
  "prefix_cache_hit_rate": 0.87,
  "total_tokens_saved_today": 18420000,
  "estimated_daily_cost_savings_usd": 184.20,
  "active_prefix_entries": 47,
  "evictions_last_hour": 3
}
```

---

## Integration with vLLM (Self-Hosted Inference)

For self-hosted SLM/Frontier clusters running vLLM, enable prefix caching with:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    enable_prefix_caching=True,   # ← Automatic prefix KV-Cache
    gpu_memory_utilization=0.90,
    tensor_parallel_size=2,
)
```

The `ContextCacheManager` in `src/control_plane/context_cache.py` mirrors this
behaviour in a **model-agnostic, API-layer implementation** that works with any
inference backend (vLLM, Anthropic API, Google Vertex AI, Azure OpenAI).

---

## References

- Anthropic Prompt Caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- Google Context Caching: https://ai.google.dev/gemini-api/docs/caching
- vLLM Prefix Caching: https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html
- PagedAttention (KV block management): Kwon et al. 2023, "Efficient Memory Management for
  Large Language Model Serving with PagedAttention"
