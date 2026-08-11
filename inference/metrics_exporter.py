"""
Prometheus Metrics Exporter
Exposes production observability metrics for the FastAPI inference backend.

Metrics exported:
  summarize_requests_total        — Counter: total requests by endpoint + status
  summarize_latency_seconds       — Histogram: end-to-end request latency
  cache_operations_total          — Counter: Redis cache hits and misses
  model_load_time_seconds         — Gauge: how long the model took to load
  gpu_memory_used_bytes           — Gauge: current GPU VRAM usage
  active_requests                 — Gauge: in-flight request count

Endpoint: GET /metrics/prometheus
Returns Prometheus text format (scraped by Prometheus server).
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Lazy import prometheus_client ─────────────────────────────────────────────
# Avoids hard crash if prometheus_client is not installed
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed — Prometheus metrics disabled.")


# ── Registry & Metrics ────────────────────────────────────────────────────────

if _PROMETHEUS_AVAILABLE:
    _REGISTRY = CollectorRegistry()

    REQUESTS_TOTAL = Counter(
        "summarize_requests_total",
        "Total number of summarization requests",
        ["endpoint", "status"],
        registry=_REGISTRY,
    )

    LATENCY_HISTOGRAM = Histogram(
        "summarize_latency_seconds",
        "End-to-end request latency in seconds",
        ["endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        registry=_REGISTRY,
    )

    CACHE_OPS_TOTAL = Counter(
        "cache_operations_total",
        "Redis cache hits and misses",
        ["operation"],  # "hit" or "miss"
        registry=_REGISTRY,
    )

    MODEL_LOAD_TIME = Gauge(
        "model_load_time_seconds",
        "Time taken to load the model at startup",
        registry=_REGISTRY,
    )

    GPU_MEMORY_USED = Gauge(
        "gpu_memory_used_bytes",
        "Current GPU VRAM usage in bytes",
        registry=_REGISTRY,
    )

    GPU_MEMORY_TOTAL = Gauge(
        "gpu_memory_total_bytes",
        "Total GPU VRAM in bytes",
        registry=_REGISTRY,
    )

    ACTIVE_REQUESTS = Gauge(
        "active_requests",
        "Number of requests currently being processed",
        registry=_REGISTRY,
    )

    BATCH_SIZE_HISTOGRAM = Histogram(
        "batch_request_size",
        "Number of documents in batch summarization requests",
        buckets=[1, 2, 4, 8, 16],
        registry=_REGISTRY,
    )
else:
    # Stub objects so import never fails
    class _Stub:
        def labels(self, *a, **kw):
            return self
        def inc(self, *a, **kw): pass
        def observe(self, *a, **kw): pass
        def set(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    _stub = _Stub()
    REQUESTS_TOTAL = _stub
    LATENCY_HISTOGRAM = _stub
    CACHE_OPS_TOTAL = _stub
    MODEL_LOAD_TIME = _stub
    GPU_MEMORY_USED = _stub
    GPU_MEMORY_TOTAL = _stub
    ACTIVE_REQUESTS = _stub
    BATCH_SIZE_HISTOGRAM = _stub


# ── Public API ────────────────────────────────────────────────────────────────

def record_request(endpoint: str, status: str, latency_seconds: float):
    """Record a completed request with status and latency."""
    REQUESTS_TOTAL.labels(endpoint=endpoint, status=status).inc()
    LATENCY_HISTOGRAM.labels(endpoint=endpoint).observe(latency_seconds)


def record_cache_hit():
    """Record a Redis cache hit."""
    CACHE_OPS_TOTAL.labels(operation="hit").inc()


def record_cache_miss():
    """Record a Redis cache miss."""
    CACHE_OPS_TOTAL.labels(operation="miss").inc()


def set_model_load_time(seconds: float):
    """Record the model load time at startup."""
    MODEL_LOAD_TIME.set(seconds)


def update_gpu_metrics(used_bytes: int, total_bytes: int):
    """Update GPU memory gauges."""
    GPU_MEMORY_USED.set(used_bytes)
    GPU_MEMORY_TOTAL.set(total_bytes)


def get_prometheus_output() -> Optional[bytes]:
    """
    Generate the Prometheus text format metrics payload.
    Returns None if prometheus_client is not available.
    """
    if not _PROMETHEUS_AVAILABLE:
        return None
    return generate_latest(_REGISTRY)


def get_content_type() -> str:
    """Return the correct Content-Type for Prometheus responses."""
    if _PROMETHEUS_AVAILABLE:
        return CONTENT_TYPE_LATEST
    return "text/plain"
