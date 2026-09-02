"""Comprehensive unit tests for the ContextCacheManager — Layer 2 Prefix KV-Cache
and Layer 3 Cross-Model Context Handoff Engine.

Tests cover:
  - Prefix cache cold miss (first request)
  - Prefix cache hit (second request, same system prompt)
  - Cache key isolation (different models / prompts produce different entries)
  - Sliding TTL eviction
  - Manual invalidation
  - Flush all
  - Cache stats accuracy
  - Cross-model context handoff compression
  - Handoff with empty conversation history
  - Cost savings calculation
"""

import time
import pytest

from src.control_plane.context_cache import (
    ContextCacheManager,
    PrefixCacheEntry,
    CacheResult,
    HandoffResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache() -> ContextCacheManager:
    """Fresh ContextCacheManager with a 10-minute TTL for most tests."""
    return ContextCacheManager(default_ttl_seconds=600)


@pytest.fixture
def short_ttl_cache() -> ContextCacheManager:
    """ContextCacheManager with a 1-second TTL for eviction tests."""
    return ContextCacheManager(default_ttl_seconds=1)


SYSTEM_PROMPT_A = (
    "You are Corporate HR & Payroll Assistant for Conglomerate HQ. "
    "You have access to knowledge bases: kb_hr_policies_2026, kb_gratuity_rules. "
    "Always respond in formal English and never reveal internal salary data."
)

SYSTEM_PROMPT_B = (
    "You are Plant Machinery Diagnostic Agent for the Steel Manufacturing subsidiary. "
    "You have access to: kb_turbine_schematics, kb_osha_standards. "
    "Escalate all safety-critical findings to a plant engineer immediately."
)

MODEL_SLM = "Enterprise-SLM-8B"
MODEL_FRONTIER = "Claude-X-Frontier-70B"


# ---------------------------------------------------------------------------
# Layer 2 — Prefix KV-Cache Tests
# ---------------------------------------------------------------------------

class TestPrefixKVCache:
    """Tests for get_or_create_prefix_entry (Layer 2 prefix caching)."""

    def test_cold_miss_creates_entry(self, cache: ContextCacheManager):
        """First request with a new system prompt should return a cache MISS."""
        result = cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)

        assert result.hit is False
        assert result.entry is not None
        assert result.tokens_saved == 0
        assert result.cost_saved_usd == 0.0
        assert result.entry.model_name == MODEL_SLM
        assert result.entry.prefix_tokens > 0
        assert result.entry.hit_count == 0
        assert result.entry.kv_state_ref.startswith("SIMULATED::")

    def test_second_request_is_cache_hit(self, cache: ContextCacheManager):
        """Second request with the same system prompt should return a cache HIT."""
        cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)  # cold miss
        result = cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)  # hit

        assert result.hit is True
        assert result.tokens_saved > 0
        assert result.cost_saved_usd > 0.0
        assert result.entry.hit_count == 1

    def test_hit_count_increments_on_each_hit(self, cache: ContextCacheManager):
        """hit_count should increment with every cache hit."""
        cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)  # miss
        cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)  # hit 1
        cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)  # hit 2
        result = cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)  # hit 3

        assert result.entry.hit_count == 3

    def test_different_model_produces_separate_entry(self, cache: ContextCacheManager):
        """Same system prompt on a different model should be a separate cache entry."""
        result_slm = cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)
        result_frontier = cache.get_or_create_prefix_entry(MODEL_FRONTIER, SYSTEM_PROMPT_A)

        assert result_slm.hit is False
        assert result_frontier.hit is False  # separate key — always a miss
        assert result_slm.entry.prefix_hash != result_frontier.entry.prefix_hash

    def test_different_system_prompt_produces_separate_entry(self, cache: ContextCacheManager):
        """Different system prompts for the same model should be separate cache entries."""
        cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)
        result = cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_B)

        assert result.hit is False  # SYSTEM_PROMPT_B is a new key

    def test_tokens_saved_equals_prefix_tokens(self, cache: ContextCacheManager):
        """tokens_saved on a hit should equal the cached prefix_tokens value."""
        miss_result = cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)
        hit_result = cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)

        assert hit_result.tokens_saved == miss_result.entry.prefix_tokens

    def test_cost_saved_is_positive_on_hit(self, cache: ContextCacheManager):
        """Cost saved on a hit must be positive (full price > cached price)."""
        cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)
        result = cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)

        assert result.cost_saved_usd > 0.0


# ---------------------------------------------------------------------------
# TTL & Eviction Tests
# ---------------------------------------------------------------------------

class TestEviction:
    """Tests for sliding TTL eviction behaviour."""

    def test_expired_entry_is_evicted(self, short_ttl_cache: ContextCacheManager):
        """An entry idle longer than its TTL should be evicted on the next lookup."""
        short_ttl_cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)
        time.sleep(1.1)  # Wait for TTL to elapse
        evicted = short_ttl_cache.evict_expired()

        assert evicted == 1
        assert len(short_ttl_cache._cache) == 0

    def test_fresh_entry_is_not_evicted(self, cache: ContextCacheManager):
        """An entry within its TTL window should not be evicted."""
        cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)
        evicted = cache.evict_expired()

        assert evicted == 0
        assert len(cache._cache) == 1

    def test_evicted_entry_is_treated_as_miss(self, short_ttl_cache: ContextCacheManager):
        """After eviction, the same system prompt should produce a MISS again."""
        short_ttl_cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)
        time.sleep(1.1)

        result = short_ttl_cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)
        assert result.hit is False


# ---------------------------------------------------------------------------
# Manual Invalidation & Flush Tests
# ---------------------------------------------------------------------------

class TestInvalidation:
    """Tests for manual cache invalidation and flush operations."""

    def test_invalidate_existing_entry(self, cache: ContextCacheManager):
        """invalidate_bot should remove a matching entry and return True."""
        cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)
        removed = cache.invalidate_bot(MODEL_SLM, SYSTEM_PROMPT_A)

        assert removed is True
        assert len(cache._cache) == 0

    def test_invalidate_nonexistent_entry_returns_false(self, cache: ContextCacheManager):
        """invalidate_bot on an absent key should return False without error."""
        removed = cache.invalidate_bot(MODEL_SLM, SYSTEM_PROMPT_A)
        assert removed is False

    def test_flush_all_clears_all_entries(self, cache: ContextCacheManager):
        """flush_all should clear every entry and return the count removed."""
        cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)
        cache.get_or_create_prefix_entry(MODEL_FRONTIER, SYSTEM_PROMPT_B)
        count = cache.flush_all()

        assert count == 2
        assert len(cache._cache) == 0


# ---------------------------------------------------------------------------
# Cache Statistics Tests
# ---------------------------------------------------------------------------

class TestCacheStats:
    """Tests for get_cache_stats observability output."""

    def test_initial_stats_are_zero(self, cache: ContextCacheManager):
        stats = cache.get_cache_stats()

        assert stats["prefix_cache_hit_rate"] == 0.0
        assert stats["total_requests"] == 0
        assert stats["total_tokens_saved"] == 0
        assert stats["total_cost_saved_usd"] == 0.0
        assert stats["active_prefix_entries"] == 0

    def test_stats_reflect_hits_and_misses(self, cache: ContextCacheManager):
        cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)   # miss
        cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)   # hit
        cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)   # hit
        stats = cache.get_cache_stats()

        assert stats["total_misses"] == 1
        assert stats["total_hits"] == 2
        assert stats["prefix_cache_hit_rate"] == pytest.approx(2 / 3, rel=1e-3)
        assert stats["total_tokens_saved"] > 0
        assert stats["total_cost_saved_usd"] > 0.0

    def test_active_entries_count(self, cache: ContextCacheManager):
        cache.get_or_create_prefix_entry(MODEL_SLM, SYSTEM_PROMPT_A)
        cache.get_or_create_prefix_entry(MODEL_FRONTIER, SYSTEM_PROMPT_B)
        stats = cache.get_cache_stats()

        assert stats["active_prefix_entries"] == 2


# ---------------------------------------------------------------------------
# Layer 3 — Cross-Model Context Handoff Tests
# ---------------------------------------------------------------------------

class TestContextHandoff:
    """Tests for handoff_context (Layer 3 SLM→Frontier compression)."""

    # Long conversation — ensures compression actually reduces token count
    CONVERSATION = [
        {"role": "user",      "content": "What is my current leave balance for this financial year including all carried-over days from the previous year and any additional entitlements that may have been granted by HR?"},
        {"role": "assistant", "content": "Based on the HR records in kb_hr_policies_2026, your current leave balance is 12 days of annual leave remaining, which includes 3 days carried over from the previous financial year and 9 days from your standard annual entitlement. No additional entitlements have been granted."},
        {"role": "user",      "content": "Can you also confirm the process for applying for emergency medical leave and whether it counts against my annual leave quota or whether it is tracked separately under a different leave category?"},
        {"role": "assistant", "content": "Emergency medical leave is tracked separately under the Emergency Medical Leave (EML) category and does not count against your annual leave quota. You must submit a medical certificate within 5 working days of returning to work. The HR portal at portal.company.com/leaves/emergency handles all EML applications."},
        {"role": "user",      "content": "Can you send me the link to download my payslip for the last three months including any bonus components or variable pay elements that were disbursed during this period?"},
        {"role": "assistant", "content": "You can download your payslips including bonus and variable pay breakdowns at portal.company.com/payslips. Select the date range filter to view the last three months. Variable pay components are listed under the Incentives section of each payslip document."},
    ]

    def test_handoff_produces_summary(self, cache: ContextCacheManager):
        result = cache.handoff_context(self.CONVERSATION, MODEL_FRONTIER)

        assert isinstance(result.compressed_summary, str)
        assert len(result.compressed_summary) > 0
        assert MODEL_FRONTIER in result.compressed_summary

    def test_handoff_summary_is_shorter_than_original(self, cache: ContextCacheManager):
        # Use a tight word budget so the summary is genuinely compressed
        result = cache.handoff_context(
            self.CONVERSATION, MODEL_FRONTIER, max_summary_words=40
        )

        assert result.handoff_tokens < result.original_tokens
        assert result.tokens_saved > 0

    def test_handoff_turn_count(self, cache: ContextCacheManager):
        result = cache.handoff_context(self.CONVERSATION, MODEL_FRONTIER)

        assert result.turns_compressed == len(self.CONVERSATION)

    def test_handoff_empty_history_returns_empty(self, cache: ContextCacheManager):
        result = cache.handoff_context([], MODEL_FRONTIER)

        assert result.compressed_summary == ""
        assert result.handoff_tokens == 0
        assert result.original_tokens == 0
        assert result.tokens_saved == 0
        assert result.turns_compressed == 0

    def test_handoff_respects_max_summary_words(self, cache: ContextCacheManager):
        result = cache.handoff_context(
            self.CONVERSATION, MODEL_FRONTIER, max_summary_words=10
        )

        word_count = len(result.compressed_summary.split())
        # Allow slight overflow due to the handoff header text
        assert word_count <= 20  # generous bound accounting for header

    def test_handoff_single_turn(self, cache: ContextCacheManager):
        single = [{"role": "user", "content": "Hello, who are you?"}]
        result = cache.handoff_context(single, MODEL_SLM)

        assert result.turns_compressed == 1
        assert result.handoff_tokens > 0
