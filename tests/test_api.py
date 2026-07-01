"""Tests for the FastAPI inference backend."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def mock_summarizer():
    """Mock summarizer that returns a predefined summary."""
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(return_value={
        "summary": "Scientists developed a new battery technology with 4x energy density.",
        "cached": False,
        "latency_ms": 342.5,
        "document_length": 800,
        "summary_length": 70,
        "compression_ratio": 0.0875,
    })
    summarizer.get_latency_percentiles = MagicMock(return_value={
        "p50_ms": 320.0, "p90_ms": 500.0, "p95_ms": 650.0,
        "p99_ms": 900.0, "mean_ms": 380.0, "total_requests": 10,
    })
    return summarizer


@pytest.fixture
def mock_loader():
    loader = MagicMock()
    loader.is_loaded = True
    loader.get_gpu_stats.return_value = {
        "gpu_available": True, "gpu_memory_used_gb": 5.2, "gpu_memory_total_gb": 16.0,
    }
    loader.engine_type = "transformers"
    return loader


@pytest.fixture
def mock_cache():
    cache = MagicMock()
    cache.is_connected = True
    cache.health_check = AsyncMock(return_value=True)
    cache.get_metrics.return_value = {
        "hits": 5, "misses": 5, "total": 10, "hit_rate": 0.5,
    }
    return cache


SAMPLE_DOCUMENT = (
    "Scientists at MIT have developed a revolutionary new battery technology "
    "that could transform electric vehicles. The new design uses lithium-sulfur "
    "chemistry with a novel cathode that prevents the common dissolution problem. "
    "In lab tests, batteries maintained 80% capacity after 1,500 charge cycles. "
    "The technology could enable electric vehicles with over 600 miles of range "
    "and could reach commercial production within three to five years. "
    "The research was funded by the Department of Energy and published in Nature Energy."
)


@pytest.mark.asyncio
async def test_summarize_endpoint(mock_summarizer, mock_loader, mock_cache):
    """Test the /summarize endpoint returns correct response structure."""
    with (
        patch("inference.main.get_summarizer", return_value=mock_summarizer),
        patch("inference.main.get_model_loader", return_value=mock_loader),
        patch("inference.main.get_cache", return_value=mock_cache),
        patch("inference.main._summarizer", mock_summarizer),
    ):
        from inference.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/summarize",
                json={"document": SAMPLE_DOCUMENT, "max_length": 256, "temperature": 0.1},
            )

    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "latency_ms" in data
    assert "cached" in data
    assert "compression_ratio" in data
    assert isinstance(data["summary"], str)
    assert len(data["summary"]) > 0


@pytest.mark.asyncio
async def test_summarize_short_document(mock_summarizer):
    """Test that documents < 50 chars are rejected."""
    with patch("inference.main._summarizer", mock_summarizer):
        from inference.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/summarize",
                json={"document": "Too short", "max_length": 256},
            )
    # Pydantic validation error → 422
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_endpoint(mock_loader, mock_cache, mock_summarizer):
    """Test the /health endpoint."""
    with (
        patch("inference.main.get_model_loader", return_value=mock_loader),
        patch("inference.main.get_cache", return_value=mock_cache),
        patch("inference.main._summarizer", mock_summarizer),
    ):
        from inference.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "redis_connected" in data


@pytest.mark.asyncio
async def test_batch_summarize(mock_summarizer, mock_loader, mock_cache):
    """Test the /summarize/batch endpoint."""
    with (
        patch("inference.main._summarizer", mock_summarizer),
        patch("inference.main.get_summarizer", return_value=mock_summarizer),
    ):
        from inference.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/summarize/batch",
                json={
                    "documents": [SAMPLE_DOCUMENT, SAMPLE_DOCUMENT],
                    "max_length": 256,
                },
            )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["summaries"]) == 2


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test the root / endpoint."""
    from inference.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "service" in data
