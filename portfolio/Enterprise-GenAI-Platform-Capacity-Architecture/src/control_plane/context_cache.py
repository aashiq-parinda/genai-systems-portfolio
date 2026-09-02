"""Context Caching Layer — Prefix KV-Cache & Cross-Model Context Handoff.

Implements a three-layer caching hierarchy for enterprise LLM serving:

  Layer 1 — Semantic Response Cache (Redis, external):
    Full response deduplication for identical or near-identical queries.

  Layer 2 — Prefix KV-Cache (this module):
    SHA-256 keyed cache of transformer KV-state references for shared
    system prompt prefixes. After the first request, subsequent requests
    sharing the same bot system prompt only pay for incremental user
    query tokens — equivalent to Anthropic Prompt Caching, Google
    Context Caching, and vLLM's enable_prefix_caching flag.

  Layer 3 — Cross-Model Context Handoff (this module):
    Compresses multi-turn SLM conversation history into a concise summary
    when a request escalates to the Frontier model, preventing full cold
    re-processing of the prior context at frontier token prices.

References:
  - Anthropic Prompt Caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
  - Google Context Caching: https://ai.google.dev/gemini-api/docs/caching
  - vLLM Prefix Caching: https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html
  - PagedAttention: Kwon et al. 2023, arXiv:2309.06180
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class PrefixCacheEntry:
    """A single prefix KV-cache entry keyed by (model_name, system_prompt) hash.

    In a real vLLM / TGI deployment, ``kv_state_ref`` is an opaque handle
    (e.g., a PagedAttention block-table ID) returned by the inference engine
    after the first prefill of the system prompt.  In this API-layer
    implementation it stores a compact representation of the cached state so
    that the gateway can signal the inference backend to skip re-processing.

    Attributes:
        prefix_hash:     SHA-256 hex digest of ``model_name + system_prompt``.
        model_name:      The serving model this entry belongs to.
        system_prompt:   The original system prompt text that was cached.
        prefix_tokens:   Estimated token count of the cached prefix.
        kv_state_ref:    Opaque reference to the KV-state block (engine handle).
        hit_count:       Number of requests that reused this entry.
        created_at:      Unix timestamp when the entry was first populated.
        last_hit_at:     Unix timestamp of the most recent cache hit.
        ttl_seconds:     Sliding TTL — entry is evicted if idle for this long.
    """
    prefix_hash: str
    model_name: str
    system_prompt: str
    prefix_tokens: int
    kv_state_ref: str  # Opaque engine handle or "SIMULATED:<hash>"
    hit_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_hit_at: float = field(default_factory=time.time)
    ttl_seconds: int = 600  # 10 minutes sliding window

    def is_expired(self) -> bool:
        """Return True if the sliding TTL has elapsed since last access."""
        return (time.time() - self.last_hit_at) > self.ttl_seconds

    def touch(self) -> None:
        """Reset the sliding TTL window on cache hit."""
        self.last_hit_at = time.time()
        self.hit_count += 1


@dataclass
class CacheResult:
    """Outcome of a prefix cache lookup, returned to the gateway.

    Attributes:
        hit:             True if the prefix KV-states were found in cache.
        entry:           The matched ``PrefixCacheEntry`` (None on miss).
        tokens_saved:    Tokens avoided by this cache hit (0 on miss).
        cost_saved_usd:  Estimated USD cost avoided (0.0 on miss).
    """
    hit: bool
    entry: Optional[PrefixCacheEntry]
    tokens_saved: int = 0
    cost_saved_usd: float = 0.0


@dataclass
class HandoffResult:
    """Result of a SLM→Frontier cross-model context handoff compression.

    Attributes:
        compressed_summary:   Condensed summary of SLM conversation turns.
        handoff_tokens:       Token count of the compressed summary.
        original_tokens:      Token count of the raw conversation history.
        tokens_saved:         Tokens avoided by compression.
        turns_compressed:     Number of SLM turns summarised.
    """
    compressed_summary: str
    handoff_tokens: int
    original_tokens: int
    tokens_saved: int
    turns_compressed: int


# ---------------------------------------------------------------------------
# Context Cache Manager
# ---------------------------------------------------------------------------

class ContextCacheManager:
    """Production-grade prefix KV-cache and cross-model context handoff engine.

    This class is stateless-safe for multi-worker FastAPI deployments when
    backed by a shared Redis store.  The current implementation uses an
    in-process dict (suitable for single-worker development / testing).
    Swap ``_cache`` for a Redis client in production.

    Args:
        default_ttl_seconds:       Sliding TTL for all cache entries.
        input_token_cost_usd_1k:   Cost per 1K input tokens (non-cached tier).
        cached_token_cost_usd_1k:  Cost per 1K cached input tokens.
        tokens_per_word:           Heuristic token/word ratio (GPT-style BPE).
    """

    def __init__(
        self,
        default_ttl_seconds: int = 600,
        input_token_cost_usd_1k: float = 0.003,
        cached_token_cost_usd_1k: float = 0.0003,
        tokens_per_word: float = 1.3,
    ) -> None:
        self._cache: Dict[str, PrefixCacheEntry] = {}
        self.default_ttl = default_ttl_seconds
        self.full_cost = input_token_cost_usd_1k
        self.cached_cost = cached_token_cost_usd_1k
        self.tokens_per_word = tokens_per_word

        # Lifetime counters for observability
        self._total_hits: int = 0
        self._total_misses: int = 0
        self._total_tokens_saved: int = 0
        self._total_cost_saved_usd: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, model_name: str, system_prompt: str) -> str:
        """Deterministic SHA-256 key from model name + system prompt content."""
        raw = f"{model_name}::{system_prompt}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _estimate_tokens(self, text: str) -> int:
        """Heuristic word-to-token estimator (~1.3 tokens per word for BPE)."""
        return max(1, int(len(text.split()) * self.tokens_per_word))

    def _estimate_cost_saved(self, tokens: int) -> float:
        """Cost delta between full and cached token pricing for ``tokens`` tokens."""
        cost_full = (tokens / 1000.0) * self.full_cost
        cost_cached = (tokens / 1000.0) * self.cached_cost
        return round(cost_full - cost_cached, 8)

    def _make_kv_ref(self, prefix_hash: str, model_name: str) -> str:
        """Construct a simulated KV block reference handle.

        In production this is replaced by the opaque block-table ID returned
        by vLLM's ``AsyncLLMEngine`` or the cache token ID from the Anthropic /
        Google API response.
        """
        return f"SIMULATED::{model_name[:12]}::{prefix_hash[:16]}"

    # ------------------------------------------------------------------
    # Layer 2 — Prefix KV-Cache
    # ------------------------------------------------------------------

    def get_or_create_prefix_entry(
        self,
        model_name: str,
        system_prompt: str,
    ) -> CacheResult:
        """Look up or populate a prefix KV-cache entry.

        On a **cache miss** a new entry is created, simulating the first-request
        prefill cost (full tokens billed).  On a **cache hit** the sliding TTL
        is refreshed and the caller receives the KV-state reference so the
        inference backend can skip re-processing the system prompt prefix.

        Args:
            model_name:    Name of the serving model (part of the cache key).
            system_prompt: Full system prompt text for this bot/request.

        Returns:
            ``CacheResult`` with hit status, the matched entry, and savings.
        """
        self.evict_expired()

        key = self._make_key(model_name, system_prompt)

        if key in self._cache:
            entry = self._cache[key]
            entry.touch()

            tokens_saved = entry.prefix_tokens
            cost_saved = self._estimate_cost_saved(tokens_saved)

            self._total_hits += 1
            self._total_tokens_saved += tokens_saved
            self._total_cost_saved_usd += cost_saved

            return CacheResult(
                hit=True,
                entry=entry,
                tokens_saved=tokens_saved,
                cost_saved_usd=cost_saved,
            )

        # Cache MISS — create and store a new entry
        prefix_tokens = self._estimate_tokens(system_prompt)
        kv_ref = self._make_kv_ref(key, model_name)

        entry = PrefixCacheEntry(
            prefix_hash=key,
            model_name=model_name,
            system_prompt=system_prompt,
            prefix_tokens=prefix_tokens,
            kv_state_ref=kv_ref,
            ttl_seconds=self.default_ttl,
        )
        self._cache[key] = entry
        self._total_misses += 1

        return CacheResult(hit=False, entry=entry, tokens_saved=0, cost_saved_usd=0.0)

    # ------------------------------------------------------------------
    # Layer 3 — Cross-Model Context Handoff
    # ------------------------------------------------------------------

    def handoff_context(
        self,
        conversation_history: List[Dict[str, str]],
        target_model: str,
        max_summary_words: int = 80,
    ) -> HandoffResult:
        """Compress multi-turn SLM conversation history for Frontier escalation.

        When a request escalates from the SLM tier to the Frontier tier
        mid-session, re-processing the full conversation history at frontier
        token prices is wasteful.  This method produces a compressed summary
        of prior SLM turns that can be prepended to the Frontier context,
        achieving context coherence at a fraction of the token cost.

        Args:
            conversation_history: List of ``{"role": str, "content": str}`` dicts
                                  from the SLM session.
            target_model:         Name of the Frontier model receiving the handoff.
            max_summary_words:    Maximum word budget for the compressed summary.

        Returns:
            ``HandoffResult`` with compressed summary text and savings metrics.
        """
        if not conversation_history:
            return HandoffResult(
                compressed_summary="",
                handoff_tokens=0,
                original_tokens=0,
                tokens_saved=0,
                turns_compressed=0,
            )

        # Calculate original token cost
        original_text = " ".join(m["content"] for m in conversation_history)
        original_tokens = self._estimate_tokens(original_text)

        # Build a template-driven extractive summary
        # (In production: call SLM with a summarisation instruction)
        turns = [
            f"{m['role'].capitalize()}: {m['content'][:120]}"
            for m in conversation_history
            if m.get("role") in ("user", "assistant")
        ]
        raw_summary = " | ".join(turns)

        # Truncate to word budget
        words = raw_summary.split()
        if len(words) > max_summary_words:
            words = words[:max_summary_words]
            raw_summary = " ".join(words) + "..."

        compressed_summary = (
            f"[CONTEXT HANDOFF from SLM to {target_model}] "
            f"Prior {len(conversation_history)} turns summarised: {raw_summary}"
        )

        handoff_tokens = self._estimate_tokens(compressed_summary)
        tokens_saved = max(0, original_tokens - handoff_tokens)

        return HandoffResult(
            compressed_summary=compressed_summary,
            handoff_tokens=handoff_tokens,
            original_tokens=original_tokens,
            tokens_saved=tokens_saved,
            turns_compressed=len(conversation_history),
        )

    # ------------------------------------------------------------------
    # Cache management & observability
    # ------------------------------------------------------------------

    def evict_expired(self) -> int:
        """Evict all entries whose sliding TTL has elapsed.

        Returns:
            Number of entries evicted.
        """
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for k in expired_keys:
            del self._cache[k]
        return len(expired_keys)

    def invalidate_bot(self, model_name: str, system_prompt: str) -> bool:
        """Manually invalidate a specific prefix cache entry.

        Used when a bot's system prompt is updated — the stale KV-state
        reference must be flushed before the next request.

        Args:
            model_name:    Model name for the entry to invalidate.
            system_prompt: System prompt text for the entry to invalidate.

        Returns:
            True if an entry was found and removed, False otherwise.
        """
        key = self._make_key(model_name, system_prompt)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def flush_all(self) -> int:
        """Flush the entire prefix cache (admin operation).

        Returns:
            Number of entries removed.
        """
        count = len(self._cache)
        self._cache.clear()
        return count

    def get_cache_stats(self) -> Dict:
        """Return aggregated lifetime and current cache statistics.

        Returns:
            Dict with hit rate, token savings, cost savings, and entry count.
        """
        total_requests = self._total_hits + self._total_misses
        hit_rate = (
            round(self._total_hits / total_requests, 4)
            if total_requests > 0
            else 0.0
        )

        active_entries = [
            {
                "prefix_hash": e.prefix_hash[:16] + "...",
                "model": e.model_name,
                "prefix_tokens": e.prefix_tokens,
                "hit_count": e.hit_count,
                "age_seconds": round(time.time() - e.created_at, 1),
                "idle_seconds": round(time.time() - e.last_hit_at, 1),
            }
            for e in self._cache.values()
        ]

        return {
            "prefix_cache_hit_rate": hit_rate,
            "total_requests": total_requests,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "total_tokens_saved": self._total_tokens_saved,
            "total_cost_saved_usd": round(self._total_cost_saved_usd, 6),
            "active_prefix_entries": len(self._cache),
            "active_entries": active_entries,
        }
