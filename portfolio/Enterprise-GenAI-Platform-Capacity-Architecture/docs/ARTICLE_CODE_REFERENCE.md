# 📖 Article Code Reference Map

> **For readers of:** *"SDEs: You Know API Design. But Can You Design an API for an LLM?"*
>
> This document maps every concept in the article to the **exact file and function** that implements it in this project. Read the article section, then open the linked file to see it running.

---

## How to Run This First

```bash
git clone https://github.com/aashiq-parinda/Enterprise-GenAI-Platform-Capacity-Architecture
cd Enterprise-GenAI-Platform-Capacity-Architecture

cp .env.example .env
docker compose up
```

Open **http://localhost:8001/docs** → the public API gateway Swagger UI.
Open **http://localhost:8000/docs** → the enterprise control plane Swagger UI.

The dev API key is printed to the terminal on first startup.

---

## Section-by-Section Code Map

---

### § 1 — OpenAI-Compatible API

> *"A simplified FastAPI endpoint..."*

| What | File | What to look at |
|---|---|---|
| The `/v1/chat/completions` route | [`src/gateway/api/routes/chat.py`](../src/gateway/api/routes/chat.py) | `chat_completions()` — the actual endpoint handler |
| Request/response Pydantic schemas | [`src/gateway/schemas/chat.py`](../src/gateway/schemas/chat.py) | `ChatCompletionRequest`, `ChatCompletionResponse`, `ChatStreamChunk` |
| OpenAI-compatible model list | [`src/gateway/api/routes/health.py`](../src/gateway/api/routes/health.py) | `list_models()` — `GET /v1/models` |

**Try it:**
```bash
curl http://localhost:8001/v1/models \
  -H "Authorization: Bearer $LLM_API_KEY"
```

---

### § 2 — Your API Key Should Never Reach the GPU Server

> *"FastAPI becomes your control plane. vLLM becomes your data plane."*

| What | File | What to look at |
|---|---|---|
| Auth middleware — intercepts every request | [`src/gateway/middleware/auth.py`](../src/gateway/middleware/auth.py) | `AuthMiddleware.dispatch()` — keys validated here, never forwarded |
| vLLM client — the only code that talks to vLLM | [`src/gateway/infrastructure/vllm_client.py`](../src/gateway/infrastructure/vllm_client.py) | `VLLMClient` — note: it only accepts internal URLs, never public |
| How the gateway is wired together | [`src/gateway/main.py`](../src/gateway/main.py) | Middleware registration order — auth wraps everything |

**The key insight in code:**
```python
# src/gateway/middleware/auth.py
# The raw key is validated HERE. It never flows to the inference layer.
result = await auth_service.validate_raw_key(raw_key)
```

---

### § 3 — Authentication ≠ Identity ≠ Quota

> *"Those are different problems."*

| Concept | File | What to look at |
|---|---|---|
| **Authentication** — "Who is this account?" | [`src/gateway/services/auth_service.py`](../src/gateway/services/auth_service.py) | `validate_raw_key()` — hash comparison, returns `(APIKey, User)` or `None` |
| **Identity/Risk** — "How suspicious is this?" | [`src/gateway/services/identity_service.py`](../src/gateway/services/identity_service.py) | `IdentityService.assess()` — computes `RiskAssessment` from coarse signals |
| **Quota** — "How much budget remains?" | [`src/gateway/services/quota_service.py`](../src/gateway/services/quota_service.py) | `check_daily_quota()`, `check_monthly_quota()` — separate from auth |
| The three services called in sequence | [`src/gateway/services/inference_service.py`](../src/gateway/services/inference_service.py) | `InferenceService.chat()` — see steps 1→5 in order |

**These three services are deliberately separate classes.** Notice how `InferenceService` calls them independently — auth is middleware, identity is a risk check, quota is a budget check.

---

### § 4 — Free-Tier Abuse Prevention / Identity Signals

> *"Build a risk and identity model from multiple signals."*

| What | File | What to look at |
|---|---|---|
| Risk score calculation | [`src/gateway/services/identity_service.py`](../src/gateway/services/identity_service.py) | `IdentityService.assess()` — 5 signals, weighted score, `low/medium/high` |
| Privacy-preserving IP bucketing | [`src/gateway/services/identity_service.py`](../src/gateway/services/identity_service.py) | `make_ip_bucket()` — hashes IP to /24 subnet, stores HMAC, never raw IP |
| Signals stored on the User model | [`src/gateway/models/user.py`](../src/gateway/models/user.py) | `verified_email`, `verified_phone`, `account_age_days` (computed property) |
| Failed auth tracking | [`src/gateway/services/auth_service.py`](../src/gateway/services/auth_service.py) | `record_failed_auth()` — brute-force signal |

**The signals used in this project:**
```python
# src/gateway/services/identity_service.py
# Signal 1: Account age (new accounts = higher risk)
# Signal 2: Email verification flag
# Signal 3: Phone verification flag (trust bonus)
# Signal 4: Failed auth attempts from Redis
# Signal 5: Hourly request rate
```

---

### § 5 — Rate Limiting Gets More Interesting With LLMs

> *"RPM alone isn't enough... One request might be 100 tokens, another 100,000."*

| What | File | What to look at |
|---|---|---|
| Full rate limit service | [`src/gateway/services/rate_limit_service.py`](../src/gateway/services/rate_limit_service.py) | `check_rpm()`, `check_tpm()`, `acquire_slot()` — three separate checks |
| Why sliding window vs token bucket | [`src/gateway/services/rate_limit_service.py`](../src/gateway/services/rate_limit_service.py) | Module docstring — tradeoff explanation |
| The 429 response with headers | [`src/gateway/services/rate_limit_service.py`](../src/gateway/services/rate_limit_service.py) | `raise_429()` — `Retry-After`, `X-RateLimit-Limit`, `X-RateLimit-Remaining` |
| Plan definitions (free/pro/enterprise) | [`src/gateway/services/quota_service.py`](../src/gateway/services/quota_service.py) | `_plan_defaults()` — RPM, TPM, daily_tokens, monthly_tokens, max_concurrent |

---

### § 6 — Redis + PostgreSQL: Fast Enforcement + Durable Accounting

> *"Redis → Fast enforcement. PostgreSQL → Durable accounting."*

| What | File | What to look at |
|---|---|---|
| Redis Lua scripts (atomic, race-condition-safe) | [`src/gateway/infrastructure/redis.py`](../src/gateway/infrastructure/redis.py) | `_SLIDING_WINDOW_SCRIPT`, `_ACQUIRE_SLOT_SCRIPT` — why Lua, explained inline |
| Dual-write rationale | [`src/gateway/models/usage_event.py`](../src/gateway/models/usage_event.py) | Module docstring — "Why both?" |
| Redis fast counters | [`src/gateway/services/quota_service.py`](../src/gateway/services/quota_service.py) | `consume_tokens()` — increments Redis counters with TTL |
| PostgreSQL durable events | [`src/gateway/services/usage_service.py`](../src/gateway/services/usage_service.py) | `UsageService.record()` — writes `UsageEvent` row after every request |

**`UsageEvent` columns (the article's usage ledger):**
```python
# src/gateway/models/usage_event.py
request_id, user_id, api_key_id, model,
input_tokens, output_tokens, total_tokens,
latency_ms, ttft_ms, status_code, created_at
```

---

### § 7 — The GPU Is Your New Bottleneck (Backpressure)

> *"You need: Admission Control → Queue → Scheduler → GPU"*

| What | File | What to look at |
|---|---|---|
| Concurrency slot acquisition | [`src/gateway/services/rate_limit_service.py`](../src/gateway/services/rate_limit_service.py) | `acquire_slot()` — atomic Redis INCR, returns False if full |
| Slot release in finally block | [`src/gateway/services/inference_service.py`](../src/gateway/services/inference_service.py) | `_handle_stream()` → the `finally:` block — always releases even on disconnect |
| Atomic Lua script for slot management | [`src/gateway/infrastructure/redis.py`](../src/gateway/infrastructure/redis.py) | `_ACQUIRE_SLOT_SCRIPT` — why this must be atomic |

**The critical pattern — slot always released:**
```python
# src/gateway/services/inference_service.py
# sse_generator() finally block
finally:
    # Slot release ALWAYS happens — even on client disconnect.
    # This is the most important lifecycle guarantee.
    await self._rate_limiter.release_slot(user_id)
```

---

### § 8 — TTFT Is Not Just API Latency

> *"TTFT — Time To First Token"*

| What | File | What to look at |
|---|---|---|
| TTFT measurement in streaming | [`src/gateway/services/inference_service.py`](../src/gateway/services/inference_service.py) | `sse_generator()` — `ttft_ms` set on first non-empty token |
| TTFT in non-streaming | [`src/gateway/infrastructure/vllm_client.py`](../src/gateway/infrastructure/vllm_client.py) | `generate()` — `ttft_ms = time elapsed to full response` |
| TTFT stored per-request | [`src/gateway/models/usage_event.py`](../src/gateway/models/usage_event.py) | `ttft_ms` field — with inline comment explaining why it's LLM-specific |
| Mock server simulates realistic TTFT | [`mock_inference/server.py`](../mock_inference/server.py) | `_stream_response()` — 150ms simulated TTFT, 30ms inter-token |

---

### § 9 — Streaming Is a First-Class Requirement

> *"The API is managing a long-lived inference stream."*

| What | File | What to look at |
|---|---|---|
| The streaming generator | [`src/gateway/services/inference_service.py`](../src/gateway/services/inference_service.py) | `_handle_stream()` + `sse_generator()` — full lifecycle |
| vLLM SSE stream reader | [`src/gateway/infrastructure/vllm_client.py`](../src/gateway/infrastructure/vllm_client.py) | `stream()` — async generator over SSE lines |
| Client disconnect + cleanup | [`src/gateway/services/inference_service.py`](../src/gateway/services/inference_service.py) | `sse_generator()` `finally:` block — slot + usage recorded regardless |
| Mock streaming with realistic timing | [`mock_inference/server.py`](../mock_inference/server.py) | `_stream_response()` — role delta → tokens → [DONE] sequence |

**Try streaming:**
```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "local-model", "messages": [{"role": "user", "content": "Explain KV cache."}], "stream": true}'
```

---

### § 14 — Observability / Structured Logging

> *"Log: request_id, user_id, model, input_tokens, ttft_ms, status..."*

| What | File | What to look at |
|---|---|---|
| structlog configuration + secret redaction | [`src/gateway/core/logging.py`](../src/gateway/core/logging.py) | `_redact_secrets()` processor — runs on every log line automatically |
| Request-ID injected into every log line | [`src/gateway/middleware/request_id.py`](../src/gateway/middleware/request_id.py) | `structlog.contextvars.bind_contextvars(request_id=...)` |
| Prometheus metrics endpoint | [`src/gateway/api/routes/health.py`](../src/gateway/api/routes/health.py) | `GET /metrics` |
| Readiness probe checks all 3 systems | [`src/gateway/api/routes/health.py`](../src/gateway/api/routes/health.py) | `ready()` — checks Postgres + Redis + inference backend |

---

### § 15 — The Full Architecture

> *"FastAPI → Redis → PostgreSQL → vLLM → GPU"*

The full wiring lives in [`src/gateway/main.py`](../src/gateway/main.py).

**Start the stack:**
```bash
docker compose up
```

**Services:**
| Service | Port | What |
|---|---|---|
| `api-gateway` | `8001` | Public LLM API gateway (this project) |
| `gateway` | `8000` | Enterprise control plane (existing) |
| `postgres` | `5432` | Usage events, keys, quotas |
| `redis` | `6379` | Rate limits, concurrency counters |
| `mock-inference` | `8002` | Fake vLLM (no GPU required) |

---

## Complete File Index

```
src/gateway/
├── main.py                          ← App factory, startup, dev key seeder
├── core/
│   ├── config.py                    ← All env vars in one place
│   ├── security.py                  ← Key generation + PBKDF2 hashing
│   └── logging.py                   ← structlog + secret redaction
├── infrastructure/
│   ├── database.py                  ← Async SQLAlchemy 2 engine
│   ├── redis.py                     ← Connection pool + Lua scripts
│   └── vllm_client.py               ← The ONLY code that touches vLLM
├── models/
│   ├── user.py                      ← User (verification flags, plan)
│   ├── api_key.py                   ← Key (hash only, never raw)
│   ├── usage_event.py               ← Per-request accounting record
│   ├── quota.py                     ← RPM/TPM/daily/monthly/concurrent limits
│   └── audit_event.py               ← Immutable security audit log
├── schemas/
│   ├── auth.py                      ← CreateKeyRequest/Response, APIKeyInfo
│   ├── chat.py                      ← OpenAI-compatible request/response
│   ├── usage.py                     ← UsageSummary, QuotaStatus
│   └── common.py                    ← ErrorDetail, HealthResponse, ModelInfo
├── services/
│   ├── auth_service.py              ← Key CRUD + validation (§3)
│   ├── identity_service.py          ← Risk scoring (§4)
│   ├── rate_limit_service.py        ← RPM/TPM/concurrency (§5, §7)
│   ├── quota_service.py             ← Daily/monthly token budgets (§5, §6)
│   ├── usage_service.py             ← PostgreSQL event writer (§6)
│   ├── model_router.py              ← Model name → backend URL
│   └── inference_service.py         ← Full request lifecycle (§7, §8, §9)
├── api/routes/
│   ├── auth.py                      ← POST/DELETE /v1/auth/keys
│   ├── chat.py                      ← POST /v1/chat/completions (§1, §9)
│   └── health.py                    ← /health, /ready, /metrics, /v1/models
└── middleware/
    ├── request_id.py                ← UUID injection + structlog binding
    └── auth.py                      ← Bearer key validation (§2)

mock_inference/
└── server.py                        ← Fake vLLM (150ms TTFT, SSE stream)

tests/gateway/
├── conftest.py                      ← SQLite fixtures, mocked Redis
├── test_auth.py                     ← Valid/invalid/revoked/missing key
├── test_rate_limit.py               ← Under/over limit, slot acquire/release
├── test_quota.py                    ← Daily/monthly exceeded, plan defaults
└── test_security.py                 ← Keys never in logs, auth rejects unauth
```

---

## Run the Tests

```bash
# Existing tests (must stay green)
pytest tests/test_capacity_math.py tests/test_control_plane.py -v

# New gateway tests (no Docker needed — uses SQLite + mocked Redis)
pytest tests/gateway/ -v
```

---

> **Production vs Educational Note:** This project demonstrates production architecture patterns. The mock inference backend returns scripted tokens — it is not real LLM inference. To connect a real vLLM node, set `INFERENCE_BACKEND=vllm` and `VLLM_BASE_URL=http://your-vllm-host:8000` in `.env`.
