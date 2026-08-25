"""
FastAPI Inference Backend — Main Application
QLoRA Llama 3.1 8B + AWQ + vLLM + Redis

Production features:
  - API key authentication (X-API-Key header)
  - Token streaming via SSE (/summarize/stream)
  - Prometheus metrics (/metrics/prometheus)
  - Rate limiting (slowapi token bucket)
  - Structured loguru logging
  - Request ID tracing
  - GZip compression
  - CORS middleware
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional, AsyncIterator

import torch
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.responses import JSONResponse, StreamingResponse, Response
from dotenv import load_dotenv

# Structured logging via loguru
from loguru import logger
import sys

# Loguru setup — replaces stdlib logging for the app
logger.remove()  # Remove default handler
_log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)
logger.add(sys.stderr, format=_log_format, level=os.environ.get("LOG_LEVEL", "INFO"))

# Optionally add JSON log file
if os.environ.get("JSON_LOGS", "false").lower() == "true":
    logger.add(
        "logs/app.log",
        format="{time} | {level} | {name} | {message}",
        rotation="100 MB",
        retention="30 days",
        serialize=True,  # JSON format
    )

# Intercept stdlib logging to loguru
class _InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

load_dotenv()

from inference.schemas import (
    SummarizeRequest,
    SummarizeResponse,
    BatchSummarizeRequest,
    BatchSummarizeResponse,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
)
from inference.cache import ResponseCache, get_cache
from inference.model_loader import VLLMModelLoader, get_model_loader
from inference.summarizer import Summarizer
from inference.middleware import setup_middleware
from inference.auth import APIKeyMiddleware
from inference.metrics_exporter import (
    record_request,
    set_model_load_time,
    update_gpu_metrics,
    get_prometheus_output,
    get_content_type,
)

# ── Global state ──
_startup_time = time.time()
_summarizer: Optional[Summarizer] = None
_total_requests = 0


# ─────────────────────────────────────────────
# Lifespan: startup & shutdown
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads model and connects to Redis on startup.
    Gracefully disconnects on shutdown.
    """
    global _summarizer

    logger.info("=" * 60)
    logger.info("  QLoRA Llama 3.1 8B — Inference API")
    logger.info("  AWQ 4-bit | vLLM | Redis | FastAPI")
    logger.info("=" * 60)

    # ── Load Model ──
    t_load_start = time.time()
    loader = get_model_loader()
    loader.load()
    load_time = time.time() - t_load_start
    set_model_load_time(load_time)
    logger.info(f"Model loaded in {load_time:.1f}s (engine: {loader.engine_type})")

    # Update GPU metrics
    gpu_stats = loader.get_gpu_stats()
    if gpu_stats.get("gpu_available"):
        update_gpu_metrics(
            gpu_stats.get("gpu_memory_used_bytes", 0),
            gpu_stats.get("gpu_memory_total_bytes", 1),
        )

    # ── Connect Redis ──
    cache = get_cache()
    await cache.connect()

    # ── Initialize Summarizer ──
    _summarizer = Summarizer(loader=loader, cache=cache)

    logger.info("✓ API ready to serve requests")

    yield  # ← API is running

    # ── Shutdown ──
    logger.info("Shutting down gracefully...")
    await cache.disconnect()
    logger.info("✓ Graceful shutdown complete")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="QLoRA Llama 3.1 8B — Document Summarization API",
    description=(
        "Production inference backend for domain-adaptive document summarization.\n\n"
        "**Model**: Llama 3.1 8B fine-tuned with QLoRA (NF4 4-bit, rank-16, α=32)\n"
        "**Quantization**: AWQ 4-bit (W4A16)\n"
        "**Serving**: vLLM AsyncLLMEngine with continuous batching\n"
        "**Caching**: Redis SHA-256 response cache (40% P95 latency reduction)\n"
        "**Streaming**: SSE token streaming via `/summarize/stream`\n"
        "**Auth**: X-API-Key header required (set API_KEY in .env)\n"
        "**Dataset**: CNN/DailyMail 50K — ROUGE-L 0.72"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Register middleware (order: last registered = outermost)
setup_middleware(
    app,
    cors_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
)

# Register API key authentication middleware
app.add_middleware(
    APIKeyMiddleware,
    api_key=os.environ.get("API_KEY", ""),
)


# ─────────────────────────────────────────────
# Rate Limiting
# ─────────────────────────────────────────────

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    _RATE_LIMIT_AVAILABLE = True
    logger.info("Rate limiting enabled via slowapi")
except ImportError:
    _RATE_LIMIT_AVAILABLE = False
    logger.warning("slowapi not installed — rate limiting disabled")

    # Stub decorator
    class _NoopLimiter:
        def limit(self, *a, **kw):
            def decorator(fn):
                return fn
            return decorator
    limiter = _NoopLimiter()


# ─────────────────────────────────────────────
# Dependency: get summarizer
# ─────────────────────────────────────────────

def get_summarizer() -> Summarizer:
    if _summarizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Try again in a moment.",
        )
    return _summarizer


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize a document",
    description=(
        "Generate a concise summary using QLoRA-tuned Llama 3.1 8B.\n\n"
        "Responses are cached in Redis by document content hash. "
        "Repeated identical requests return cached results with ~10ms latency.\n\n"
        "**Auth**: Requires `X-API-Key` header."
    ),
    tags=["Summarization"],
)
@limiter.limit("60/minute")
async def summarize(
    request: Request,
    body: SummarizeRequest,
    summarizer: Summarizer = Depends(get_summarizer),
) -> SummarizeResponse:
    """Main summarization endpoint."""
    global _total_requests
    _total_requests += 1

    t_start = time.perf_counter()
    try:
        result = await summarizer.summarize(
            document=body.document,
            max_new_tokens=body.max_length,
            temperature=body.temperature,
            top_p=body.top_p,
            use_cache=body.use_cache,
        )
        latency = time.perf_counter() - t_start
        record_request("/summarize", "200", latency)

        return SummarizeResponse(
            summary=result["summary"],
            document_length=result["document_length"],
            summary_length=result["summary_length"],
            compression_ratio=result["compression_ratio"],
            cached=result["cached"],
            latency_ms=result["latency_ms"],
        )

    except Exception as e:
        latency = time.perf_counter() - t_start
        record_request("/summarize", "500", latency)
        logger.error(f"Summarization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}",
        )


@app.post(
    "/summarize/stream",
    summary="Stream summarization tokens via SSE",
    description=(
        "Generate a summary with token-by-token Server-Sent Events (SSE) streaming.\n\n"
        "Each event is a JSON object:\n"
        "```json\n"
        "{\"token\": \"Scientists\", \"done\": false}\n"
        "{\"token\": \"\", \"done\": true, \"summary\": \"...\", \"latency_ms\": 423.1}\n"
        "```\n\n"
        "**Auth**: Requires `X-API-Key` header."
    ),
    tags=["Summarization"],
    response_class=StreamingResponse,
)
@limiter.limit("30/minute")
async def summarize_stream(
    request: Request,
    body: SummarizeRequest,
    summarizer: Summarizer = Depends(get_summarizer),
):
    """SSE streaming summarization endpoint."""
    global _total_requests
    _total_requests += 1

    t_start = time.perf_counter()

    async def generate_stream():
        try:
            async for chunk in summarizer.stream(
                document=body.document,
                max_new_tokens=body.max_length,
                temperature=body.temperature,
                top_p=body.top_p,
            ):
                yield chunk
        except Exception as e:
            import json
            logger.error(f"Streaming failed: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
        finally:
            latency = time.perf_counter() - t_start
            record_request("/summarize/stream", "200", latency)

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )


@app.post(
    "/summarize/batch",
    response_model=BatchSummarizeResponse,
    summary="Batch summarize multiple documents",
    tags=["Summarization"],
)
@limiter.limit("20/minute")
async def summarize_batch(
    request: Request,
    body: BatchSummarizeRequest,
    summarizer: Summarizer = Depends(get_summarizer),
) -> BatchSummarizeResponse:
    """Batch summarization — up to 16 documents per request."""
    global _total_requests

    t_start = time.perf_counter()
    summaries = []
    cached_count = 0

    for doc in body.documents:
        _total_requests += 1
        result = await summarizer.summarize(
            document=doc,
            max_new_tokens=body.max_length,
            temperature=body.temperature,
            use_cache=body.use_cache,
        )
        summaries.append(result["summary"])
        if result["cached"]:
            cached_count += 1

    total_latency_ms = (time.perf_counter() - t_start) * 1000
    record_request("/summarize/batch", "200", total_latency_ms / 1000)

    return BatchSummarizeResponse(
        summaries=summaries,
        count=len(summaries),
        total_latency_ms=round(total_latency_ms, 2),
        cached_count=cached_count,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Operations"],
)
async def health_check() -> HealthResponse:
    """Returns model, Redis, and GPU status. No auth required."""
    cache = get_cache()
    loader = get_model_loader()
    gpu_stats = loader.get_gpu_stats()

    # Update GPU Prometheus metrics
    if gpu_stats.get("gpu_available") and "gpu_memory_used_bytes" in gpu_stats:
        update_gpu_metrics(
            gpu_stats["gpu_memory_used_bytes"],
            gpu_stats["gpu_memory_total_bytes"],
        )

    return HealthResponse(
        status="healthy" if _summarizer is not None else "starting",
        model_loaded=loader.is_loaded,
        redis_connected=await cache.health_check(),
        gpu_available=gpu_stats.get("gpu_available", False),
        gpu_memory_used_gb=gpu_stats.get("gpu_memory_used_gb"),
        gpu_memory_total_gb=gpu_stats.get("gpu_memory_total_gb"),
        uptime_seconds=round(time.time() - _startup_time, 1),
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="In-memory performance metrics",
    tags=["Operations"],
)
async def metrics(
    summarizer: Summarizer = Depends(get_summarizer),
) -> MetricsResponse:
    """Returns in-memory latency percentiles and cache statistics."""
    cache = get_cache()
    cache_metrics = cache.get_metrics()
    latency_stats = summarizer.get_latency_percentiles()
    uptime = time.time() - _startup_time

    return MetricsResponse(
        total_requests=_total_requests,
        cache_hits=cache_metrics.get("hits", 0),
        cache_misses=cache_metrics.get("misses", 0),
        cache_hit_rate=cache_metrics.get("hit_rate", 0.0),
        p50_latency_ms=latency_stats.get("p50_ms", 0.0),
        p90_latency_ms=latency_stats.get("p90_ms", 0.0),
        p95_latency_ms=latency_stats.get("p95_ms", 0.0),
        p99_latency_ms=latency_stats.get("p99_ms", 0.0),
        avg_latency_ms=latency_stats.get("mean_ms", 0.0),
        requests_per_second=round(_total_requests / max(uptime, 1), 2),
    )


@app.get(
    "/metrics/prometheus",
    summary="Prometheus metrics endpoint",
    description=(
        "Returns metrics in Prometheus text format for scraping.\n\n"
        "Configure your prometheus.yml:\n"
        "```yaml\n"
        "scrape_configs:\n"
        "  - job_name: qlora-api\n"
        "    static_configs:\n"
        "      - targets: ['api:8000']\n"
        "    metrics_path: /metrics/prometheus\n"
        "```\n\n"
        "**Auth**: Requires `X-API-Key` header."
    ),
    tags=["Operations"],
)
async def prometheus_metrics():
    """Prometheus text-format metrics export."""
    output = get_prometheus_output()
    if output is None:
        return JSONResponse(
            status_code=503,
            content={"error": "prometheus_client not installed"},
        )
    return Response(content=output, media_type=get_content_type())


@app.delete(
    "/cache",
    summary="Flush response cache",
    tags=["Operations"],
)
async def flush_cache() -> dict:
    """Clear all Redis cached responses."""
    cache = get_cache()
    count = await cache.flush()
    return {"message": f"Flushed {count} cache entries"}


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "QLoRA Llama 3.1 8B — Document Summarization API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "prometheus": "/metrics/prometheus",
        "stream": "/summarize/stream",
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    """CLI entry point — called by qlora-serve console script."""
    import uvicorn

    uvicorn.run(
        "inference.main:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        workers=1,                   # Single worker — vLLM manages concurrency
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        access_log=True,
        timeout_keep_alive=120,
    )


if __name__ == "__main__":
    main()
