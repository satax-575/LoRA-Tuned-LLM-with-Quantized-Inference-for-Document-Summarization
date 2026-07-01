"""Tests for the Redis caching layer."""

import json
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def cache():
    from inference.cache import ResponseCache
    c = ResponseCache(redis_url="redis://localhost:6379", ttl=3600)
    return c


def test_make_key_deterministic(cache):
    """Same input should always produce the same cache key."""
    doc = "This is a test document for summarization."
    key1 = cache._make_key(doc, 256, 0.1)
    key2 = cache._make_key(doc, 256, 0.1)
    assert key1 == key2
    assert key1.startswith("summarize:")


def test_make_key_different_params(cache):
    """Different parameters should produce different keys."""
    doc = "This is a test document for summarization."
    key1 = cache._make_key(doc, 256, 0.1)
    key2 = cache._make_key(doc, 512, 0.1)   # Different max_length
    key3 = cache._make_key(doc, 256, 0.5)   # Different temperature
    assert key1 != key2
    assert key1 != key3


def test_make_key_whitespace_normalized(cache):
    """Whitespace normalization should produce the same key."""
    doc1 = "  Test document.  "
    doc2 = "Test document."
    key1 = cache._make_key(doc1, 256, 0.1)
    key2 = cache._make_key(doc2, 256, 0.1)
    assert key1 == key2


@pytest.mark.asyncio
async def test_get_returns_none_when_disconnected(cache):
    """Should return None gracefully when Redis is not connected."""
    cache._client = None
    result = await cache.get("some document", 256, 0.1)
    assert result is None


@pytest.mark.asyncio
async def test_set_returns_false_when_disconnected(cache):
    """Should return False gracefully when Redis is not connected."""
    cache._client = None
    result = await cache.set("doc", "summary", 256, 0.1, 300.0)
    assert result is False


@pytest.mark.asyncio
async def test_get_cache_hit(cache):
    """Should return cached data on a hit."""
    cached_payload = json.dumps({
        "summary": "Test summary.",
        "cached_at": time.time(),
        "original_latency_ms": 500.0,
    })
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=cached_payload)
    cache._client = mock_client

    result = await cache.get("Test document about AI.", 256, 0.1)
    assert result is not None
    assert result["summary"] == "Test summary."
    assert cache._hits == 1
    assert cache._misses == 0


@pytest.mark.asyncio
async def test_get_cache_miss(cache):
    """Should return None and increment miss counter on a miss."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=None)
    cache._client = mock_client

    result = await cache.get("Test document about AI.", 256, 0.1)
    assert result is None
    assert cache._hits == 0
    assert cache._misses == 1


def test_get_metrics_empty(cache):
    """Metrics should return zero values when no requests made."""
    metrics = cache.get_metrics()
    assert metrics["hits"] == 0
    assert metrics["misses"] == 0
    assert metrics["hit_rate"] == 0.0


def test_get_metrics_with_data(cache):
    """Metrics should correctly compute hit rate."""
    cache._hits = 4
    cache._misses = 6
    metrics = cache.get_metrics()
    assert metrics["hits"] == 4
    assert metrics["misses"] == 6
    assert metrics["total"] == 10
    assert abs(metrics["hit_rate"] - 0.4) < 1e-6


@pytest.mark.asyncio
async def test_health_check_connected(cache):
    """Health check should return True when Redis is reachable."""
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)
    cache._client = mock_client

    assert await cache.health_check() is True


@pytest.mark.asyncio
async def test_health_check_disconnected(cache):
    """Health check should return False when client is None."""
    cache._client = None
    assert await cache.health_check() is False
