"""
Tests for the response cache module.
Uses a mock Redis client — no actual Redis server needed.
"""

import pytest
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def cache():
    """Create a ResponseCache with a mock Redis client."""
    from inference.cache import ResponseCache
    c = ResponseCache(redis_url="redis://localhost:6379", ttl=3600)
    return c


@pytest.fixture
def connected_cache():
    """Create a ResponseCache with a mocked connected Redis client."""
    from inference.cache import ResponseCache
    c = ResponseCache(redis_url="redis://localhost:6379", ttl=3600)
    c._client = AsyncMock()
    c._client.ping = AsyncMock(return_value=True)
    return c


class TestResponseCache:

    def test_is_connected_when_client_none(self, cache):
        assert cache.is_connected is False

    def test_is_connected_when_client_set(self, connected_cache):
        assert connected_cache.is_connected is True

    def test_make_key_is_deterministic(self, cache):
        key1 = cache._make_key("article text", 256, 0.1)
        key2 = cache._make_key("article text", 256, 0.1)
        assert key1 == key2

    def test_make_key_differs_for_different_docs(self, cache):
        key1 = cache._make_key("article one", 256, 0.1)
        key2 = cache._make_key("article two", 256, 0.1)
        assert key1 != key2

    def test_make_key_differs_for_different_temps(self, cache):
        key1 = cache._make_key("same article", 256, 0.1)
        key2 = cache._make_key("same article", 256, 0.5)
        assert key1 != key2

    def test_make_key_has_prefix(self, cache):
        key = cache._make_key("text", 256, 0.1)
        assert key.startswith("summarize:")

    @pytest.mark.asyncio
    async def test_get_returns_none_when_disconnected(self, cache):
        result = await cache.get("doc", 256, 0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_none_on_cache_miss(self, connected_cache):
        connected_cache._client.get = AsyncMock(return_value=None)
        result = await connected_cache.get("doc", 256, 0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_data_on_cache_hit(self, connected_cache):
        cached = json.dumps({"summary": "Test summary", "cached_at": time.time(), "original_latency_ms": 300.0})
        connected_cache._client.get = AsyncMock(return_value=cached)
        result = await connected_cache.get("doc", 256, 0.1)
        assert result is not None
        assert result["summary"] == "Test summary"

    @pytest.mark.asyncio
    async def test_set_returns_false_when_disconnected(self, cache):
        result = await cache.set("doc", "summary", 256, 0.1, 300.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_set_calls_redis_setex(self, connected_cache):
        connected_cache._client.setex = AsyncMock(return_value=True)
        result = await connected_cache.set("doc", "summary", 256, 0.1, 300.0)
        assert result is True
        connected_cache._client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_connected(self, connected_cache):
        connected_cache._client.ping = AsyncMock(return_value=True)
        ok = await connected_cache.health_check()
        assert ok is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_disconnected(self, cache):
        result = await cache.health_check()
        assert result is False

    def test_get_metrics_initial_zero(self, cache):
        m = cache.get_metrics()
        assert m["hits"] == 0
        assert m["misses"] == 0
        assert m["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_metrics_track_hits(self, connected_cache):
        cached = json.dumps({"summary": "s", "cached_at": 0, "original_latency_ms": 0})
        connected_cache._client.get = AsyncMock(return_value=cached)
        await connected_cache.get("doc1", 256, 0.1)
        await connected_cache.get("doc1", 256, 0.1)
        m = connected_cache.get_metrics()
        assert m["hits"] == 2

    @pytest.mark.asyncio
    async def test_metrics_track_misses(self, connected_cache):
        connected_cache._client.get = AsyncMock(return_value=None)
        await connected_cache.get("doc1", 256, 0.1)
        await connected_cache.get("doc2", 256, 0.1)
        m = connected_cache.get_metrics()
        assert m["misses"] == 2

    @pytest.mark.asyncio
    async def test_hit_rate_calculation(self, connected_cache):
        cached = json.dumps({"summary": "s", "cached_at": 0, "original_latency_ms": 0})
        # 1 hit
        connected_cache._client.get = AsyncMock(return_value=cached)
        await connected_cache.get("doc1", 256, 0.1)
        # 1 miss
        connected_cache._client.get = AsyncMock(return_value=None)
        await connected_cache.get("doc2", 256, 0.1)

        m = connected_cache.get_metrics()
        assert m["hit_rate"] == pytest.approx(0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_flush_returns_zero_when_disconnected(self, cache):
        count = await cache.flush()
        assert count == 0

    @pytest.mark.asyncio
    async def test_flush_deletes_keys(self, connected_cache):
        connected_cache._client.keys = AsyncMock(return_value=["k1", "k2"])
        connected_cache._client.delete = AsyncMock(return_value=2)
        count = await connected_cache.flush()
        assert count == 2
